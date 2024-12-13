"""
coding: utf-8
@author: Yongsu Huang, Yehe Yan
@time: 2024/10/17 18:56
@Description: code during study in NEU for CS 7180 Advanced Perception
"""
from os.path import join, basename
import copy
import time

from tqdm.auto import tqdm

import cv2
import numpy as np
import os
import torch
import torch.nn.functional as F
from torchvision.transforms.functional import normalize

from accelerate import Accelerator
from accelerate.utils import set_seed

from kornia.morphology import dilation

from basicsr.utils import img2tensor, tensor2img

from guided_diffusion.script_util import create_gaussian_diffusion

from diffusion_arch import DensePosteriorConditionalUNet

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True

BATCH_SIZE = 1
TOTAL_ITERS = 800000
ACCUM = 16


def adjust_color_using_original(original_image, shadow_removed_image, shadow_mask):
    """
       Adjust the color of the shadow-removed image to match the original image more closely
       in non-shadowed regions.

       Args:
           original_image (ndarray): The original image before shadow removal.
           shadow_removed_image (ndarray): The image after shadow removal.
           shadow_mask (ndarray): The shadow mask used for separating shadow and non-shadow regions.

       Returns:
           corrected_image (ndarray): The color-corrected image after adjustments.
    """
    # Ensure both images have the same size
    if original_image.shape != shadow_removed_image.shape:
        shadow_removed_image = cv2.resize(shadow_removed_image, (original_image.shape[1], original_image.shape[0]))

    # Ensure shadow mask has the same size as the images
    if shadow_mask.shape[:2] != original_image.shape[:2]:
        shadow_mask = cv2.resize(shadow_mask, (original_image.shape[1], original_image.shape[0]))

    # Convert the mask to binary if necessary (255 for shadow, 0 for non-shadow)
    _, binary_mask = cv2.threshold(shadow_mask, 127, 255, cv2.THRESH_BINARY)
    inverse_mask = cv2.bitwise_not(binary_mask)

    # Convert both images to LAB color space
    original_lab = cv2.cvtColor(original_image, cv2.COLOR_BGR2LAB)
    shadow_removed_lab = cv2.cvtColor(shadow_removed_image, cv2.COLOR_BGR2LAB)

    # Split the LAB channels
    l_orig, a_orig, b_orig = cv2.split(original_lab)
    l_shadow, a_shadow, b_shadow = cv2.split(shadow_removed_lab)

    # Ensure that all channels have the same data type
    l_shadow = l_shadow.astype(np.uint8)
    a_orig = a_orig.astype(np.uint8)
    b_orig = b_orig.astype(np.uint8)

    # Adjust color using L channel of the shadow-removed image and a, b channels of the original
    corrected_lab = cv2.merge((l_shadow, a_orig, b_orig))

    # Restore color only in non-shadow regions using the mask
    corrected_lab_non_shadow = cv2.bitwise_and(corrected_lab, corrected_lab, mask=inverse_mask)

    # For the shadow areas, leave the color from the shadow_removed image
    shadow_removed_lab_shadow = cv2.bitwise_and(shadow_removed_lab, shadow_removed_lab, mask=binary_mask)

    # Combine the shadow and non-shadow regions
    final_corrected_lab = cv2.add(corrected_lab_non_shadow, shadow_removed_lab_shadow)

    # Convert the result back to BGR color space
    corrected_image = cv2.cvtColor(final_corrected_lab, cv2.COLOR_LAB2BGR)

    return corrected_image


def main_single_image(image_path, mask_path, output_dir):
    """
        Main function for processing a single image and its corresponding shadow mask using
        a pre-trained diffusion model.

        Args:
            image_path (str): Path to the input image file.
            mask_path (str): Path to the shadow mask image file.
    """
    accelerator = Accelerator(
        gradient_accumulation_steps=ACCUM,
        mixed_precision='fp16',
        project_dir="experiments",
    )
    set_seed(10666)

    if accelerator:
        print = accelerator.print

    print(f"=> Inited Accelerator")

    # initalize model
    model = DensePosteriorConditionalUNet(
        in_channels=3 + 3 + 1,
        out_channels=6,
        model_channels=192,
        num_res_blocks=2,
        attention_resolutions=[8, 16, 32],
        num_heads=4,
        num_head_channels=64,
        num_heads_upsample=-1,
        channel_mult=[1, 1, 2, 2, 2, 4],
        dropout=0.0,
        use_scale_shift_norm=True,
        resblock_updown=True,
        use_new_attention_order=True,
    )
    ema_model = copy.deepcopy(model)

    feature_encoder = DensePosteriorConditionalUNet(
        in_channels=3 + 3,
        out_channels=1,
        model_channels=96,
        num_res_blocks=1,
        attention_resolutions=[8, 16],
        num_heads=4,
        num_head_channels=-1,
        num_heads_upsample=-1,
        channel_mult=[1, 1, 2, 2, 4],
        dropout=0.0,
        use_scale_shift_norm=True,
        resblock_updown=True,
        use_new_attention_order=True,
    )
    ema_feature_encoder = copy.deepcopy(feature_encoder)

    print(f"=> Inited models")

    device = accelerator.device
    ema_model.to(device)
    ema_feature_encoder.to(device)

    model, ema_model, feature_encoder, ema_feature_encoder = accelerator.prepare(
        model, ema_model, feature_encoder, ema_feature_encoder
    )

    eval_gaussian_diffusion = create_gaussian_diffusion(
        steps=1000,
        learn_sigma=True,
        noise_schedule='linear',
        use_kl=False,
        timestep_respacing="ddim50",
        predict_xstart=False,
        rescale_timesteps=False,
        rescale_learned_sigmas=False,
        p2_gamma=0.5,
        p2_k=1,
    )

    # load model
    #accelerator.load_state('experiments/26k_state_279999.bin')
    #accelerator.load_state(r"C:\Work\checkpoints\26k_mix_checkpoints\origin_state_244999.bin")
    accelerator.load_state(r"C:\Work\checkpoints\50k\state_109999.bin")

    # load image and mask
    img_lq = cv2.imread(image_path)  # image
    original_img = img_lq
    img_lq = img_lq.astype(np.float32) / 255.0
    if img_lq is None:
        raise ValueError(f"Cannot load image: {image_path}")

    img_mask = cv2.imread(mask_path)  # mask
    img_mask = img_mask.astype(np.float32) / 255.0
    if img_mask is None:
        raise ValueError(f"Cannot load image: {mask_path}")

    # adjust size
    img_lq = cv2.resize(img_lq, (512, 512), cv2.INTER_CUBIC)
    img_mask = cv2.resize(img_mask, (512, 512), cv2.INTER_NEAREST)

    # convert to tensor
    img_lq, img_mask = img2tensor([img_lq, img_mask], bgr2rgb=True, float32=True)

    # send to gpu
    img_lq = img_lq.unsqueeze(0).to(device, non_blocking=True)
    img_mask = img_mask.unsqueeze(0).to(device, non_blocking=True)
    img_mask = dilation(img_mask, torch.ones(21, 21).to(img_mask.device))

    # Normalize
    img_lq = normalize(img_lq, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])

    print(f"img_lq shape: {img_lq.shape}")  # 应为 (1, 3, 256, 256)
    print(f"img_mask shape: {img_mask.shape}")  # 应为 (1, 1, 256, 256)

    # check and generate folder './removed'
    output_dir = output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    image_filename = os.path.basename(image_path)

    intrinsic_feature1 = None
    img_lq1 = img_lq
    # First iteration to get latent feature and first output
    with torch.no_grad():
        print(f"Iteration 1 - Initial processing")

        # Get intrinsic feature from the feature encoder
        intrinsic_feature = ema_feature_encoder(img_lq, torch.tensor([0], device=device), latent=img_mask)
        intrinsic_feature1 = intrinsic_feature
        latent = torch.cat((img_lq, intrinsic_feature), dim=1)  # Combine the image and latent feature

        # Predict using the diffusion model
        pred_gt = eval_gaussian_diffusion.ddim_sample_loop(
            ema_model,
            shape=img_lq.shape,  # Shape should match input
            model_kwargs={'latent': latent},
            progress=True
        )

        # Process the predicted result (denormalize and save it as output)
        pred_gt = pred_gt.clip(-1, 1)
        pred_gt = 255. * (pred_gt / 2 + 0.5)
        iter1_img = tensor2img(pred_gt, min_max=(0, 255))

        # Save the first output
        iter1_img = cv2.resize(iter1_img, (original_img.shape[1], original_img.shape[0]))
        iter1_img_path = os.path.join(output_dir, os.path.splitext(image_filename)[0] + "iter1.png")
        cv2.imwrite(iter1_img_path, iter1_img)
        print(f"First iteration image saved as {iter1_img_path}")

        # Perform color adjustment with shadow mask
        original_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        output_image_single = tensor2img(pred_gt, min_max=(0, 255))
        corrected_image = adjust_color_using_original(original_img, output_image_single, original_mask)

        # Save and display the corrected image
        corrected_image_path = os.path.join(output_dir,
                                            os.path.splitext(image_filename)[0] + "_iter1_color_correct.png")
        # Save the second output
        ##iter2_img = cv2.resize(iter2_img, (original_img.shape[1], original_img.shape[0]))
        # iter2_img_path = os.path.join(output_dir, os.path.splitext(image_filename)[0] + "iter2.png")
        # cv2.imwrite(iter2_img_path, iter2_img)
        cv2.imwrite(corrected_image_path, corrected_image)

        return corrected_image_path


if __name__ == "__main__":
    image_path = r"C:\Work\Instance-Shadow-Diffusion\data\aistd_test\shadow\133-16.png"
    mask_path = r"C:\Work\Instance-Shadow-Diffusion\data\aistd_test\mask\133-16.png"
    output_path = r"C:\Work\Instance-Shadow-Diffusion\sam_mask\removed"
    main_single_image(image_path, mask_path, output_path)

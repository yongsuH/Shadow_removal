"""
coding: utf-8
@author: Yongsu Huang
@time: 2024/10/30 10:23
@Description: code during study in NEU
"""
import argparse
import torch
from torchvision import transforms
from PIL import Image
import models
import utils
import yaml
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_image(image_path):
    """加载单张图片并预处理"""
    image = Image.open(image_path).convert("RGB")
    preprocess = transforms.Compose([
        transforms.Resize((1024, 1024)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    return preprocess(image).unsqueeze(0).to(device)


def batched_predict(model, inp, coord, bsize):
    """单张图片的预测"""
    with torch.no_grad():
        model.gen_feat(inp)
        n = coord.shape[1]
        ql = 0
        preds = []
        while ql < n:
            qr = min(ql + bsize, n)
            pred = model.query_rgb(coord[:, ql: qr, :])
            preds.append(pred)
            ql = qr
        pred = torch.cat(preds, dim=1)
    return pred, preds


def tensor2PIL(tensor):
    """将 Tensor 转换为 PIL 格式"""
    toPIL = transforms.ToPILImage()
    return toPIL(tensor)


def predict_single_image(model, image_path, data_norm=None, eval_type=None):
    model.eval()

    # 加载并处理图片
    inp = load_image(image_path)

    # 预测
    pred = torch.sigmoid(model.infer(inp))
    inp = inp.mean(dim=1, keepdim=True)
    print("Prediction shape:", pred.shape)
    print("Input shape:", inp.shape)

    # 计算评估指标
    if eval_type:
        metric_fn = getattr(utils, f"calc_{eval_type}")
        result1, result2, result3, result4 = metric_fn(pred, inp)
        print(f'{eval_type} metrics:')
        print('metric1: {:.4f}'.format(result1.item()))
        print('metric2: {:.4f}'.format(result2.item()))
        print('metric3: {:.4f}'.format(result3.item()))
        print('metric4: {:.4f}'.format(result4.item()))

    return pred


def getMask(image_path):
    # hardcode
    config_path = r'G:\Khoury\SAM-Adapter-PyTorch\best_results\istd\config2.yaml'
    model_path = r'G:\Khoury\SAM-Adapter-PyTorch\best_results\istd\model_epoch_best.pth'


    # 加载配置文件（如果有必要）
    data_norm = None
    eval_type = None
    if config_path:
        with open(config_path, 'r') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
            data_norm = config.get('data_norm')
            eval_type = config.get('eval_type')

    # 加载模型
    model = models.make(config['model']).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint, strict=True)

    # 单张图片预测
    pred = predict_single_image(model, image_path, data_norm=data_norm, eval_type=eval_type)

    # 将预测结果保存为图像
    pred_image = tensor2PIL(pred.squeeze().cpu())
    file_name = os.path.basename(image_path)
    output_dir = r"C:\Work\Instance-Shadow-Diffusion\sam_mask"
    new_path = os.path.join(output_dir, file_name)
    pred_image.save(new_path)
    print("Prediction saved as "+new_path)

    return new_path


if __name__ == '__main__':
    image_path = r"C:\Work\Instance-Shadow-Diffusion\data\aistd_test\shadow\95-5.png"
    getMask(image_path)

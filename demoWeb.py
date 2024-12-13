import gradio as gr
from SAM_Adapter_PyTorch.test_1img import getMask
from shadow_removal_singleOutput import main_single_image
import os
from PIL import Image


# function to process img
def process_image(image):
    input_image_path = "temp_input_image.jpg"
    if isinstance(image, str):
        # copy if path
        input_image_path = image
    else:
        image.save(input_image_path)

    # mask path
    mask_path = getMask(input_image_path)

    # output dir
    outdir = "temp_output"
    os.makedirs(outdir, exist_ok=True)

    # processing
    result_path = main_single_image(input_image_path, mask_path, outdir)

    # return mask and result
    return mask_path, result_path

# interface
with gr.Blocks() as demo:
    gr.Markdown("## Shadow Removal and Mask Generation")

    with gr.Row():
        with gr.Column():
            input_image = gr.Image(label="Upload Image", type="pil")
            submit_button = gr.Button("Process")

        with gr.Column():
            mask_output = gr.Image(label="Generated Mask")
            result_output = gr.Image(label="Final Result")

    # interact
    submit_button.click(
        process_image,
        inputs=[input_image],
        outputs=[mask_output, result_output]
    )

# start app
if __name__ == "__main__":
    demo.launch()
from tkinter import filedialog

import gradio as gr
from torchvision.utils import save_image

from deepx.nn import registered_models
from deepx.tasks import ImageGen


def generate(ckpt_path, model_name):
    model = registered_models[model_name]
    model = model(backbone="resnet18", tgt_shape=(1, 8, 8))
    model = ImageGen.load_from_checkpoint(ckpt_path, model=model)
    model.eval()
    z = model.generate_noize(16)
    generated = model.model.generator(z)
    save_image(generated, "generated.png", normalize=True)
    return "generated.png"


with gr.Blocks("Model") as app:
    gr.Markdown(
        """
        # Image Classification App
    """
    )
    with gr.Row():
        with gr.Column():
            with gr.Box():
                with gr.Row():
                    model_name = gr.Dropdown(list(registered_models.keys()), label="Model")

                ckpt_path = gr.Textbox(label="Checkpoint path")
                ckpt_btn = gr.Button(value="Select checkpoint")

                def set_ckpt_path():
                    path = filedialog.askopenfilename(
                        initialdir="/workspace/experiments",
                        title="Select checkpoint file",
                    )
                    return path

                ckpt_btn.click(fn=set_ckpt_path, outputs=ckpt_path)

            genarate_btn = gr.Button(value="Generate")

        with gr.Column():
            result = gr.Image(label="Result")
    genarate_btn.click(fn=generate, inputs=[ckpt_path, model_name], outputs=result)

app.launch()

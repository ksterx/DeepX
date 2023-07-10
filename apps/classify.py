from tkinter import filedialog

import gradio as gr
import torch

from deepx import registered_tasks
from deepx.dms import CIFAR10DM, MNISTDM
from deepx.nn import ResNet18, registered_models
from deepx.tasks import Classification

registered_dms = registered_tasks['classification']['datamodule']

def load_model(checkpoint_path, model_name, dm_name):
    model_class = registered_models[model_name]
    dm_class = registered_dms[dm_name]
    dm = dm_class()

    model = model_class(num_classes=dm.NUM_CLASSES, in_channels=dm.NUM_CHANNELS)
    model = Classification.load_from_checkpoint(checkpoint_path, model=model, num_classes=dm.NUM_CLASSES)
    model.eval()
    return model, dm


def predict(ckpt_path, model_name, dm_name, image):
    model = registered_models[model_name]
    dm = registered_dms[dm_name]
    model = model(num_classes=dm.NUM_CLASSES, in_channels=dm.NUM_CHANNELS)
    model = Classification.load_from_checkpoint(ckpt_path, model=model, num_classes=dm.NUM_CLASSES)
    model.eval()

    # Preprocess
    transform = dm.transform()
    image = transform(image).unsqueeze(0)

    # Predict
    image = image.to(model.device)
    output = model(image)
    # _, predicted = torch.max(output, 1)
    predicted = torch.argmax(output, dim=1)

    # Result
    class_names = dm.CLASSES
    return class_names[predicted.item()]


with gr.Blocks("Model") as app:
    gr.Markdown("""
        # Image Classification App
    """)
    with gr.Row():
        with gr.Column():
            with gr.Box():
                image = gr.Image(live=True, label="Image")

                with gr.Row():
                    model_name = gr.Dropdown(list(registered_models.keys()), label="Model")
                    dm_name = gr.Dropdown(list(registered_dms.keys()), label="Dataset")

                ckpt_path = gr.Textbox(label="Checkpoint path")
                ckpt_btn = gr.Button(value="Select checkpoint", size="sm")

                def set_ckpt_path():
                    path = filedialog.askopenfilename(initialdir="/workspace/experiments", title="Select checkpoint file",)
                    return path

                ckpt_btn.click(fn=set_ckpt_path, outputs=ckpt_path)

            predict_btn = gr.Button(value="Predict")

        with gr.Column():
            result = gr.Label(label="Result")
    predict_btn.click(fn=predict, inputs=[ckpt_path, model_name, dm_name, image], outputs=result)

app.launch()

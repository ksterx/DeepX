import cv2
import gradio as gr
import torch
from utils import set_ckpt_path

from deepx import registered_algos
from deepx.algos import Segmentation
from deepx.nn import registered_models

registered_dms = registered_algos["segmentation"]["datamodule"]

# data_dir = "../experiments/data/images"
# examples = [
#     [osp.join(data_dir, "car.png")],
#     [osp.join(data_dir, "cat.png")],
#     [osp.join(data_dir, "plane.jpg")],
# ]


def segment(ckpt_path, model_name, dm_name, image):
    dm = registered_dms[dm_name]

    model = registered_models[model_name]
    model = model(num_classes=dm.NUM_CLASSES, in_channels=dm.NUM_CHANNELS)
    model = Segmentation.load_from_checkpoint(
        ckpt_path, model=model, num_classes=dm.NUM_CLASSES
    )
    model.eval()

    # Preprocess
    transform = dm.transform()
    transformed_image = transform(image).unsqueeze(0)

    # Predict
    transformed_image = transformed_image.to(model.device)
    output = model(transformed_image)
    predicted = torch.argmax(output, dim=1).squeeze()  # (H, W)

    sections = []
    for i, cls in enumerate(dm.CLASSES):
        mask = torch.where(predicted == i, 1, 0)
        mask = mask.detach().cpu().numpy()
        sections.append((mask, cls))

    image = cv2.resize(image, (predicted.shape[0], predicted.shape[1]))

    return (image, sections)


with gr.Blocks("Model") as app:
    gr.Markdown(
        """
        <center><strong><font size='8'>Image Segmentation App</font></strong></center>
    """
    )
    with gr.Row():
        model_name = gr.Dropdown(list(registered_models.keys()), label="Model")
        dm_name = gr.Dropdown(list(registered_dms.keys()), label="Dataset")

        with gr.Column():
            ckpt_path = gr.Textbox(label="Checkpoint path")
            ckpt_btn = gr.Button(value="Select checkpoint")

            ckpt_btn.click(fn=set_ckpt_path, outputs=ckpt_path)

    with gr.Row():
        img_input = gr.Image(label="Image")
        img_output = gr.AnnotatedImage(label="Result")

    seg_btn = gr.Button(value="Segment")
    seg_btn.click(
        fn=segment,
        inputs=[
            ckpt_path,
            model_name,
            dm_name,
            img_input,
        ],
        outputs=img_output,
    )

app.launch()

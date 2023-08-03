import gradio as gr
import torch
from app import App

from deepx import registered_tasks
from deepx.nn import registered_models
from deepx.tasks import ClassificationTask


class ClassificationApp(App):
    def __init__(self, initialdir):
        super().__init__(initialdir)

        self.registered_dms = registered_tasks["classification"]["datamodule"]

        with gr.Blocks("Model") as self.app:
            gr.Markdown(
                """
                <center><strong><font size='8'>Image Classification App</font></strong></center>
            """
            )
            with gr.Row():
                with gr.Column():
                    with gr.Box():
                        image = gr.Image(label="Image")

                        with gr.Row():
                            model_name = gr.Dropdown(
                                list(registered_models.keys()), label="Model"
                            )
                            dm_name = gr.Dropdown(
                                list(self.registered_dms.keys()), label="Dataset"
                            )

                        ckpt_path = gr.Textbox(label="Checkpoint path")
                        ckpt_btn = gr.Button(value="Select checkpoint", size="sm")

                        ckpt_btn.click(
                            fn=self.set_ckpt_path,
                            outputs=ckpt_path,
                        )

                    predict_btn = gr.Button(value="Predict")

                with gr.Column():
                    result = gr.Label(label="Result")

            predict_btn.click(
                fn=self.predict,
                inputs=[ckpt_path, model_name, dm_name, image],
                outputs=result,
            )

    def launch(self):
        self.app.launch()

    def load_model(self, checkpoint_path, model_name, dm_name):
        model_class = registered_models[model_name]
        dm_class = self.registered_dms[dm_name]
        dm = dm_class()

        model = model_class(num_classes=dm.NUM_CLASSES, in_channels=dm.NUM_CHANNELS)
        model = ClassificationTask.load_from_checkpoint(
            checkpoint_path, model=model, num_classes=dm.NUM_CLASSES
        )
        model.eval()
        return model, dm

    def predict(self, ckpt_path, model_name, dm_name, image):
        model = registered_models[model_name]
        dm = self.registered_dms[dm_name]
        model = model(num_classes=dm.NUM_CLASSES, in_channels=dm.NUM_CHANNELS)
        model = ClassificationTask.load_from_checkpoint(
            ckpt_path, model=model, num_classes=dm.NUM_CLASSES
        )
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

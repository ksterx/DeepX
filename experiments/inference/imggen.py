import gradio as gr
from app import App
from torchvision.utils import save_image

from deepx.nn import registered_models
from deepx.tasks import GAN


class ImageGenerationApp(App):
    def __init__(self, initialdir):
        super().__init__(initialdir)

        with gr.Blocks("Model") as self.app:
            gr.Markdown(
                """
                <center><strong><font size='8'>Image Generation App</font></strong></center>
            """
            )
            with gr.Row():
                with gr.Column():
                    with gr.Row():
                        model_name = gr.Dropdown(
                            list(registered_models.keys()), label="Model"
                        )

                        with gr.Column():
                            ckpt_path = gr.Textbox(label="Checkpoint path")
                            ckpt_btn = gr.Button(value="Select checkpoint")

                    with gr.Row():
                        target_shape = gr.Dropdown(
                            [
                                "1x28x28",
                                "3x32x32",
                                "3x64x64",
                                "3x128x128",
                                "3x256x256",
                            ],
                            label="Target shape",
                        )
                        # set a dimention (latent_dim)
                        latent_dim = gr.Number(label="Latent dim", value=100)
                        base_dim_g = gr.Number(label="Generator base dim", value=128)
                        base_dim_d = gr.Number(
                            label="Discriminator base dim", value=128
                        )

                    ckpt_btn.click(fn=self.set_ckpt_path, outputs=ckpt_path)

                    genarate_btn = gr.Button(value="Generate")

                with gr.Column():
                    result = gr.Image(label="Result")
            genarate_btn.click(
                fn=self.generate,
                inputs=[
                    ckpt_path,
                    model_name,
                    target_shape,
                    latent_dim,
                    base_dim_g,
                    base_dim_d,
                ],
                outputs=result,
            )

    def launch(self):
        self.app.launch()

    def generate(
        self, ckpt_path, model_name, tgt_shape, latent_dim, base_dim_g, base_dim_d
    ):
        tgt_shape = tuple(map(int, tgt_shape.split("x")))

        model = registered_models[model_name]
        model = model(
            tgt_shape=tgt_shape,
            latent_dim=int(latent_dim),
            base_dim_g=int(base_dim_g),
            base_dim_d=int(base_dim_d),
            dropout=0.0,
            negative_slope=0.0,
        )
        model = GAN.load_from_checkpoint(ckpt_path, model=model)
        model.eval()
        z = model.generate_noize(16)
        generated = model.model.generator(z)  # type: ignore
        save_image(generated, "generated.png", normalize=True, nrow=4)
        return "generated.png"

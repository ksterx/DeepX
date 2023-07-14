import tempfile

import torch
from torch import nn
from torchvision.utils import save_image

from .task import TaskX


class ImageGen(TaskX):
    NAME = "imagegen"

    def __init__(
        self,
        model: str | nn.Module,
        lr: float = 1e-3,
        loss_fn: nn.Module | str = "bce",
        optimizer: str | torch.optim.Optimizer = "adam",
        **kwargs,
    ):
        super().__init__(model=model, lr=lr, loss_fn=loss_fn, optimizer=optimizer, **kwargs)

        self.automatic_optimization = False

        self.generator = self.model.generator
        self.discriminator = self.model.discriminator

    def forward(self, x):
        return self.generator(x)

    def _mode_step(self, batch, batch_idx, mode):
        img, _ = batch
        self.type_ = img

        opt_g, opt_d = self.optimizers()

        # Train discriminator
        if mode == "train":
            self.toggle_optimizer(opt_d)
        # for real images
        preds = self.discriminator(img)
        loss_real = self.loss_fn(preds, torch.ones_like(preds))
        # for fake images
        z = self.generate_noize(img.shape[0])
        fake_img = self.generator(z)
        preds = self.discriminator(fake_img)
        loss_fake = self.loss_fn(preds, torch.zeros_like(preds))
        loss_d = (loss_real + loss_fake) / 2

        if mode == "train":
            self.manual_backward(loss_d)
            opt_d.step()
            opt_d.zero_grad()
            self.untoggle_optimizer(opt_d)

        # Train generator
        if mode == "train":
            self.toggle_optimizer(opt_g)
        fake_img = self.generator(z)
        preds = self.discriminator(fake_img)
        loss_g = self.loss_fn(preds, torch.ones_like(preds))

        self.log(f"{mode}_loss_g", loss_g, on_step=True, on_epoch=True, prog_bar=True)
        self.log(f"{mode}_loss_d", loss_d, on_step=True, on_epoch=True, prog_bar=True)
        self.log(f"{mode}_loss_real", loss_real, on_step=True, on_epoch=True)
        self.log(f"{mode}_loss_fake", loss_fake, on_step=True, on_epoch=True)

        if mode == "train":
            self.manual_backward(loss_g)
            opt_g.step()
            opt_g.zero_grad()
            self.untoggle_optimizer(opt_g)

    def configure_optimizers(self):
        opt_g = torch.optim.Adam(self.generator.parameters(), lr=self.lr)
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=self.lr)
        print("Generator optimizer:", opt_g)
        print("Discriminator optimizer:", opt_d)
        return opt_g, opt_d

    def on_validation_epoch_end(self):
        z = self.generate_noize(16)
        fake_img = self.generator(z)
        print("Generated images:", fake_img.shape)
        save_image(fake_img, "generated.png", nrow=4, normalize=False)
        save_image(fake_img, "generated_norm.png", nrow=4, normalize=True)

    def generate_noize(self, batch_size):
        z = torch.randn(batch_size, self.generator.latent_dim, 1, 1)
        z = z.to(self.device)
        return z
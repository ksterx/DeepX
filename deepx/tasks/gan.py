import tempfile

import numpy as np
import torch
from torch import Tensor, nn, optim

# from torchmetrics.classification import BinaryAccuracy
from torchmetrics.image.fid import FrechetInceptionDistance
from torchvision.utils import make_grid, save_image

from ..utils.vision import denormalize
from .core import DataModuleConfig, ModelConfig, Task, TaskConfig, Trainer


class GANModelConfig(ModelConfig):
    def __init__(
        self,
        latent_dim: int,
        generator: nn.Module,
        discriminator: nn.Module,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.latent_dim = latent_dim
        self.generator = generator
        self.discriminator = discriminator


class GANConfig(TaskConfig):
    def __init__(self, one_side_label_smoothing: float, **kwargs):
        super().__init__(**kwargs)
        self.one_side_label_smoothing = one_side_label_smoothing


class GAN(Task):
    NAME = "gan"
    SEED = 2525

    def __init__(
        self,
        model_cfg: GANModelConfig,
        task_cfg: GANConfig,
    ):
        super().__init__(
            model_cfg=model_cfg,
            task_cfg=task_cfg,
        )

        self.automatic_optimization = False

        # if not isinstance(self.model.generator, nn.Module):
        #     raise ValueError(f"Invalid model: {self.model}")

        self.generator = self.model.generator
        self.discriminator = self.model.discriminator

        # self.val_acc_real = BinaryAccuracy()
        # self.val_acc_fake = BinaryAccuracy()
        self.test_metric = FrechetInceptionDistance()

    def forward(self, x: Tensor) -> Tensor:
        return self.generator(x)

    def _mode_step(self, batch, batch_idx, mode):
        img, _ = batch
        self.type_ = img

        opt_g, opt_d = self.optimizers()

        # Train discriminator
        if mode == "train":
            self.toggle_optimizer(opt_d)

        # for real images
        preds = self.discriminator(img)  # [batch_size, 1]
        tgt = torch.ones_like(preds)
        loss_real = self.loss_fn(preds, tgt * self.tparams.one_side_label_smoothing)
        # if mode == "val":
        #     acc_real = self.val_acc_real(preds, tgt)

        #     self.log("val_acc_real", acc_real, on_step=True, on_epoch=True)

        # for fake images
        z = self.generate_noize(img.shape[0])
        fake_img = self.generator(z)
        preds = self.discriminator(fake_img)  # [batch_size, 1]
        tgt = torch.zeros_like(preds)
        loss_fake = self.loss_fn(preds, tgt)
        # if mode == "val":
        #     acc_fake = self.val_acc_fake(preds, tgt)
        #     acc = (acc_real + acc_fake) / 2

        #     self.log("val_acc_fake", acc_fake, on_step=True, on_epoch=True)
        #     self.log("val_acc", acc, on_step=True, on_epoch=True, prog_bar=True)

        # Discriminator loss
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

        # Generator loss
        loss_g = self.loss_fn(preds, torch.ones_like(preds))

        # Metrics
        if img.shape[1] == 3 and mode == "test":
            img = denormalize(img, mean=0.1307, std=0.3081, is_batch=True)
            fake_img = denormalize(fake_img, mean=0.1307, std=0.3081, is_batch=True)
            exec(f"self.{mode}_metric.update(img, real=True)")
            exec(f"self.{mode}_metric.update(fake_img, real=False)")
            fid = eval(f"self.{mode}_metric.compute()")
            self.log(f"{mode}_fid", fid)

        self.log(f"{mode}_loss_g", loss_g, on_step=True, on_epoch=True)
        self.log(f"{mode}_loss_d", loss_d, on_step=True, on_epoch=True)

        if mode != "train":
            self.log(f"{mode}_loss_real", loss_real, on_step=True, on_epoch=True)
            self.log(f"{mode}_loss_fake", loss_fake, on_step=True, on_epoch=True)

        if mode == "train":
            self.manual_backward(loss_g)
            opt_g.step()
            opt_g.zero_grad()
            self.untoggle_optimizer(opt_g)

    def configure_optimizers(self):
        opt_g = torch.optim.Adam(
            self.generator.parameters(),
            lr=self.hparams.lr,
            betas=(self.hparams.beta1, self.hparams.beta2),
        )
        opt_d = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=self.hparams.lr,
            betas=(self.hparams.beta1, self.hparams.beta2),
        )
        print("Generator optimizer:", opt_g)
        print("Discriminator optimizer:", opt_d)
        return opt_g, opt_d

    def on_train_end(self):
        self._make_gif_from_images("img_*.png", "epoch", "training.gif")

    def on_validation_epoch_end(self):
        z = self.generate_noize(16, seed=self.SEED)
        fake_img = self.generator(z)
        grid = make_grid(fake_img, nrow=4)

        with tempfile.TemporaryDirectory() as tmpdir:
            img_path = f"{tmpdir}/img_{self.trainer.current_epoch:03d}.png"
            save_image(grid, fp=img_path)
            self.logger.experiment.log_artifacts(self.logger.run_id, tmpdir)

    def generate_noize(self, batch_size: int, seed: int | None = None):
        if isinstance(seed, int):
            np.random.seed(seed)
            z = np.random.randn(batch_size, self.generator.latent_dim, 1, 1)  # type: ignore
            z = torch.from_numpy(z).float().to(self.device)
        else:
            z = torch.randn(
                batch_size, self.generator.latent_dim, 1, 1, device=self.device
            )  # type: ignore
        return z


class GANTrainer(Trainer):
    NAME = "gan"

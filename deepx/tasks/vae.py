import tempfile
from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F
from torchmetrics.classification import BinaryAccuracy

# from torchmetrics.image.fid import FrechetInceptionDistance
from torchvision.utils import make_grid, save_image

from ..utils.vision import denormalize
from .core import DataModuleConfig, ModelConfig, Task, TaskConfig, Trainer


class VAEModelConfig(ModelConfig):
    def __init__(
        self,
        backbone: str,
        latent_dim: int,
        base_dim_g: int,
        dropout: float,
        negative_slope: float,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.backbone = backbone
        self.latent_dim = latent_dim
        self.base_dim_g = base_dim_g
        self.dropout = dropout
        self.negative_slope = negative_slope


class VAETaskConfig(TaskConfig):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class VAEDMConfig(DataModuleConfig):
    def __init__(
        self,
        train_ratio: float,
        mean: tuple[float, ...],
        std: tuple[float, ...],
        download: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.train_ratio = train_ratio
        self.mean = mean
        self.std = std
        self.download = download


class VAETask(Task):
    NAME = "vae"
    SEED = 2525

    def __init__(
        self,
        model_cfg: VAEModelConfig,
        task_cfg: VAETaskConfig,
    ):
        super().__init__(
            model_cfg=model_cfg,
            task_cfg=task_cfg,
        )

        self.automatic_optimization = False

        self.decoder = self.model.decoder
        self.encoder = self.model.encoder

        self.train_acc = BinaryAccuracy()
        self.val_acc = BinaryAccuracy()
        self.test_acc = BinaryAccuracy()

    def forward(self, x: Tensor) -> Tensor:
        return self.decoder(x)

    def _mode_step(self, batch, batch_idx, mode):
        tgt, _ = batch
        logits, z, mu, logvar = self.model(tgt)
        pred = torch.sigmoid(logits)
        loss, rec_loss, kl_loss = self.criterion(pred, tgt, mu, logvar)

        self.log(f"{mode}_loss", loss, prog_bar=True)

        return loss

    def criterion(self, pred, tgt, mu, logvar):
        rec_loss = F.binary_cross_entropy(pred, tgt)
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        loss = rec_loss + kl_loss
        return loss, rec_loss, kl_loss

    def on_train_end(self):
        self._make_gif_from_images("img_*.png", "epoch", "training.gif")

    def on_validation_epoch_end(self):
        z = self.generate_noize(16, seed=self.SEED)
        fake_img = self.decoder(z)
        grid = make_grid(fake_img, nrow=4)

        with tempfile.TemporaryDirectory() as tmpdir:
            img_path = f"{tmpdir}/img_{self.trainer.current_epoch:03d}.png"
            save_image(grid, fp=img_path)
            self.logger.experiment.log_artifacts(self.logger.run_id, tmpdir)

    def generate_noize(self, batch_size: int, seed: int | None = None):
        if isinstance(seed, int):
            np.random.seed(seed)
            z = np.random.randn(batch_size, self.decoder.latent_dim, 1, 1)  # type: ignore
            z = torch.from_numpy(z).float().to(self.device)
        else:
            z = torch.randn(
                batch_size, self.decoder.latent_dim, 1, 1, device=self.device
            )  # type: ignore
        return z


class VAETrainer(Trainer):
    NAME = "vae"

    def _update_configs(self):
        h, w = self.dm.SIZE
        self.model_cfg.update(tgt_shape=(self.dm.NUM_CHANNELS, h, w))

    def _build_task(self, model_cfg: ModelConfig, task_cfg: TaskConfig) -> Task:
        return VAETask(model_cfg=model_cfg, task_cfg=task_cfg)


@dataclass
class VAE:
    model_cfg: ModelConfig = VAEModelConfig
    task_cfg: TaskConfig = VAETaskConfig
    dm_cfg: DataModuleConfig = VAEDMConfig
    trainer: Trainer = VAETrainer

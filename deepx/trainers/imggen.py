import math
import warnings

import torch
from lightning import LightningDataModule, LightningModule
from torch import nn, optim

from .trainer import TrainerX


class ImageGenerationTrainer(TrainerX):
    NAME = "imagegeneration"

    def __init__(
        self,
        backbone: str | nn.Module,
        model: str | LightningModule,
        datamodule: str | LightningDataModule,
        batch_size: int = 32,
        train_ratio: float = 0.8,
        num_workers: int = 2,
        download: bool = False,
        lr: float = 1e-3,
        loss_fn: str | nn.Module = "bce",
        optimizer: str | optim.Optimizer = "adam",
        scheduler: str | optim.lr_scheduler._LRScheduler = "cos",
        root_dir: str = "/workspace",
        data_dir: str = "/workspace/experiments/data",
        log_dir: str = "/workspace/experiments/runs",
        hidden_dim: int = 1024,
        negative_slope: float = 0.01,
        dropout: float = 0.0,
        latent_dim: int = 1024,
        base_channels: int = 32,
        **kwargs,
    ):
        super().__init__(
            model=model,
            datamodule=datamodule,
            batch_size=batch_size,
            train_ratio=train_ratio,
            num_workers=num_workers,
            download=download,
            lr=lr,
            loss_fn=loss_fn,
            optimizer=optimizer,
            scheduler=scheduler,
            root_dir=root_dir,
            data_dir=data_dir,
            log_dir=log_dir,
            **kwargs,
        )

        if loss_fn != "bce":
            warnings.warn(
                f"Loss function {loss_fn} might cause problems. Use 'bce' instead."
            )

        torch.manual_seed(2525)

        # self.dm_cfg.update({})
        self.datamodule = self.get_datamodule(datamodule=datamodule, **self.dm_cfg)

        h, _ = self.datamodule.SIZE
        h = 2 ** math.ceil(math.log2(h))
        tgt_shape = (self.datamodule.NUM_CHANNELS, h, h)
        self.model_cfg.update(
            {
                "backbone": backbone,
                "tgt_shape": tgt_shape,
                "hidden_dim": hidden_dim,
                "negative_slope": negative_slope,
                "dropout": dropout,
                "latent_dim": latent_dim,
                "base_channels": base_channels,
            }
        )
        self.model = self.get_model(model, **self.model_cfg)

        self.algo_cfg.update({"model": self.model})
        self.algo = self.get_algo(algo=self.NAME, **self.algo_cfg)

        self.hparams.update(self.dm_cfg)
        self.hparams.update(self.model_cfg)
        self.hparams.update(self.algo_cfg)

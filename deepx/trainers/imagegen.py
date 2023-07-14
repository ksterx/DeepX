from lightning import LightningDataModule, LightningModule
from torch import nn, optim

from .trainer import TrainerX


class ImageGenTrainer(TrainerX):
    NAME = "imagegen"

    def __init__(
        self,
        backbone: str,
        model: str | LightningModule,
        datamodule: str | LightningDataModule,
        batch_size: int = 32,
        train_ratio: float = 0.8,
        num_workers: int = 2,
        download: bool = False,
        lr: float = 1e-3,
        loss_fn: str | nn.Module = "bce",
        optimizer: str | optim.Optimizer = "adam",
        root_dir: str = "/workspace",
        data_dir: str = "/workspace/data",
        log_dir: str = "/workspace/experiments",
        hidden_dim: int = 1024,
        negative_slope: float = 0.01,
        p_dropout: float = 0.0,
        latent_dim: int = 100,
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
            root_dir=root_dir,
            data_dir=data_dir,
            log_dir=log_dir,
            **kwargs,
        )

        # self.dm_cfg.update({})
        self.datamodule = self.get_datamodule(datamodule=datamodule, **self.dm_cfg)

        tgt_shape = (self.datamodule.NUM_CHANNELS, *self.datamodule.SIZE)
        self.model_cfg.update(
            {
                "backbone": backbone,
                "tgt_shape": tgt_shape,
                "hidden_dim": hidden_dim,
                "negative_slope": negative_slope,
                "p_dropout": p_dropout,
                "latent_dim": latent_dim,
                "base_channels": base_channels,
            }
        )
        self.model = self.get_model(model, **self.model_cfg)

        self.task_cfg.update({"model": self.model})
        self.task = self.get_task(task=self.NAME, **self.task_cfg)

        self.hparams.update(self.dm_cfg)
        self.hparams.update(self.model_cfg)
        self.hparams.update(self.task_cfg)

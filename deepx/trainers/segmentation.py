import torch
from lightning import LightningDataModule, LightningModule
from torch import nn

from .trainer import TrainerX


class SegmentationTrainer(TrainerX):
    NAME = "segmentation"

    def __init__(
        self,
        model: str | LightningModule,
        datamodule: str | LightningDataModule,
        batch_size: int = 32,
        train_ratio: float = 0.8,
        num_workers: int = 2,
        download: bool = False,
        lr: float = 1e-3,
        loss_fn: str | nn.Module = "ce",
        optimizer: str | torch.optim.Optimizer = "adam",
        scheduler: str | torch.optim.lr_scheduler._LRScheduler = "cos",
        root_dir: str = "/workspace",
        data_dir: str = "/workspace/experiments/data",
        log_dir: str = "/workspace/experiments/mlruns",
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

        # self.dm_cfg.update({})
        self.datamodule = self.get_datamodule(datamodule=datamodule, **self.dm_cfg)

        num_classes = self.datamodule.NUM_CLASSES
        num_channels = self.datamodule.NUM_CHANNELS

        self.model_cfg.update({"num_classes": num_classes, "in_channels": num_channels})
        self.model = self.get_model(model, **self.model_cfg)

        self.algo_cfg.update({"model": self.model, "num_classes": num_classes})
        self.algo = self.get_algo(algo=self.NAME, **self.algo_cfg)

        self.hparams.update(self.dm_cfg)
        self.hparams.update(self.model_cfg)
        self.hparams.update(self.algo_cfg)

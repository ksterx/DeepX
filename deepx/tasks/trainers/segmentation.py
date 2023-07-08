import torch
from lightning import LightningDataModule, LightningModule
from torch import nn

from .trainer import TrainerX


class SegmentationTrainer(TrainerX):
    TASK_TYPE = "classification"

    def __init__(
        self,
        model: str | LightningModule,
        datamodule: str | LightningDataModule,
        batch_size: int = 32,
        train_ratio: float = 0.8,
        num_workers: int = 2,
        download: bool = False,
        lr: float = 1e-3,
        loss_fn: str | nn.Module = nn.CrossEntropyLoss(),
        optimizer: str | torch.optim.Optimizer = "adam",
        root_dir: str = "/workspace",
        data_dir: str = "/workspace/data",
        log_dir: str = "/workspace/experiments",
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

        self.hparams.update({"lr": lr, "loss_fn": loss_fn, "optimizer": optimizer})

        # Set up model
        num_classes = self.datamodule.NUM_CLASSES
        in_channels = self.datamodule.NUM_CHANNELS
        self.model = self.get_model(model, num_classes=num_classes, in_channels=in_channels)

        # Set up task
        self.task = self.get_task(
            self.TASK_TYPE,
            model=self.model,
            num_classes=num_classes,
            lr=lr,
            loss_fn=loss_fn,
            optimizer=optimizer,
        )

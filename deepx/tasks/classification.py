import torch
from torch import nn, optim
from torchmetrics import Accuracy

from .core import Task, Trainer


class Classification(Task):
    NAME = "classification"

    def __init__(
        self,
        model: str,
        num_classes: int,
        lr: float = 1e-4,
        loss_fn: str | nn.Module = "ce",
        optimizer: str | optim.Optimizer = "adam",
        scheduler: str | optim.lr_scheduler.LRScheduler = "cos",
        **kwargs,
    ):
        super().__init__(
            model=model,
            lr=lr,
            loss_fn=loss_fn,
            optimizer=optimizer,
            scheduler=scheduler,
            **kwargs,
        )

        self.initialize()

        self.train_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.test_acc = Accuracy(task="multiclass", num_classes=num_classes)

    def initialize(self):
        super().initialize()

    def predict_step(self, batch, batch_idx, dataloader_idx):
        x = batch
        logits = self(x)
        preds = torch.argmax(logits, dim=1)
        return preds

    def _mode_step(self, batch, batch_idx, mode):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        preds = torch.argmax(logits, dim=1)

        acc = eval(f"self.{mode}_acc")
        acc(preds, y)
        self.log(f"{mode}_acc", eval(f"self.{mode}_acc"), prog_bar=True)
        self.log(f"{mode}_loss", loss, prog_bar=True)

        return loss


class ClassificationTrainer(Trainer):
    def __init__(
        self,
        model: str,
        datamodule: str,
        batch_size: int = 32,
        train_ratio: float = 0.8,
        num_workers: int = 2,
        download: bool = False,
        lr: float = 1e-4,
        loss_fn: str | nn.Module = "ce",
        optimizer: str | optim.Optimizer = "adam",
        scheduler: str | optim.lr_scheduler._LRScheduler = "cos",
        root_dir: str = "/workspace",
        data_dir: str = "/workspace/experiments/data",
        log_dir: str = "/workspace/experiments/mlruns",
        dropout: float = 0.0,
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

        self.datamodule = self.get_datamodule(datamodule=datamodule, **self.dm_cfg)

        num_classes = self.datamodule.NUM_CLASSES
        num_channels = self.datamodule.NUM_CHANNELS

        self.task_cfg.update({"model": model, "num_classes": num_classes})
        self.task = Classification(**self.task_cfg)

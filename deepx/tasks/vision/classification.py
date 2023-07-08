from abc import ABC, abstractmethod

import torch
from torch import nn
from torchmetrics import Accuracy
from torchvision import transforms

from deepx.tasks import DataModuleX, TaskX


class Classification(TaskX):
    def __init__(
        self,
        model: str | nn.Module,
        num_classes: int,
        lr: float = 1e-3,
        loss_fn: nn.Module | str = nn.CrossEntropyLoss(),
        optimizer: str | torch.optim.Optimizer = "adam",
        **kwargs,
    ):
        super().__init__(model=model, lr=lr, loss_fn=loss_fn, optimizer=optimizer, **kwargs)

        self.train_accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        self.test_accuracy = Accuracy(task="multiclass", num_classes=num_classes)

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

        exec(f"self.{mode}_accuracy.update(preds, y)")
        self.log(f"{mode}_loss", loss, prog_bar=True)
        self.log(f"{mode}_acc", eval(f"self.{mode}_accuracy"), prog_bar=True)

        return loss

    @property
    def name(self):
        return "classification"


class ClassificationDM(DataModuleX, ABC):
    def __init__(
        self,
        data_dir: str,
        batch_size: int,
        train_ratio: float,
        num_workers: int,
        download: bool = False,
        **kwargs,
    ):
        super().__init__(
            data_dir=data_dir,
            batch_size=batch_size,
            train_ratio=train_ratio,
            num_workers=num_workers,
            download=download,
        )

    def transform(self, size, antialias=True, mean=(0.1307,), std=(0.3081,)):
        return transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize(size, antialias=antialias),
                transforms.Normalize(mean, std),
            ]
        )

    def train_transform(self, size, antialias=True, mean=(0.1307,), std=(0.3081,)):
        return transforms.Compose(
            [
                transforms.RandomCrop(size, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Resize(size, antialias=antialias),
                transforms.Normalize(mean, std),
            ]
        )

    @property
    @abstractmethod
    def name(self):
        return "None"

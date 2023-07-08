import pathlib
from abc import ABC, abstractmethod

import torch
from lightning import LightningDataModule, LightningModule
from torch import nn
from torch.utils.data import DataLoader, random_split

from deepx.nn import registered_losses


class TaskX(LightningModule, ABC):
    def __init__(
        self,
        model,
        lr: float = 1e-3,
        loss_fn: nn.Module | str = nn.CrossEntropyLoss(),
        optimizer: str | torch.optim.Optimizer = "adam",
    ):
        LightningModule.__init__(self)
        ABC.__init__(self)

        self.model = model
        self.lr = lr
        self.optimizer = optimizer

        # Loss function
        if isinstance(loss_fn, str):
            self.loss_fn = registered_losses[loss_fn]()
        elif isinstance(loss_fn, nn.Module):
            self.loss_fn = loss_fn
        else:
            raise ValueError(f"Invalid loss function: {loss_fn}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def training_step(self, batch, batch_idx):
        return self._mode_step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        return self._mode_step(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        return self._mode_step(batch, batch_idx, "test")

    def _mode_step(self, batch, batch_idx, mode):
        return NotImplementedError

    def configure_optimizers(self):
        if isinstance(self.optimizer, str):
            if self.optimizer == "adam":
                return torch.optim.Adam(self.parameters(), lr=self.lr)
            elif self.optimizer == "sgd":
                return torch.optim.SGD(self.parameters(), lr=self.lr)
            else:
                raise ValueError(f"Invalid optimizer: {self.optimizer}")
        else:
            return self.optimizer(self.parameters(), lr=self.lr)

    @property
    @abstractmethod
    def name(self):
        return "None"


class DataModuleX(LightningDataModule, ABC):
    TASK_TYPE = "None"

    def __init__(
        self,
        data_dir: str | pathlib.Path,
        batch_size: int = 32,
        train_ratio: float = 0.9,
        num_workers: int = 2,
        download: bool = False,
    ):
        LightningDataModule.__init__(self)
        ABC.__init__(self)

        if not isinstance(data_dir, str):
            data_dir = str(data_dir)

        self.data_dir = data_dir
        self.batch_size = batch_size
        self.train_ratio = train_ratio
        self.num_workers = num_workers
        self.download = download

    @abstractmethod
    def setup(self, stage=None):
        self.train_data = None
        self.val_data = None
        self.test_data = None
        self.predict_data = None

    def train_dataloader(self):
        return DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_data,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_data,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def predict_dataloader(self):
        return DataLoader(
            self.predict_data,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    @property
    @abstractmethod
    def name(self):
        return "None"

    def _random_split(self, data):
        num_data = len(data)
        len_train = int(num_data * self.train_ratio)
        len_val = num_data - len_train
        return random_split(dataset=data, lengths=[len_train, len_val])

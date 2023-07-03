import pathlib

import torch
from lightning import LightningDataModule, LightningModule
from torch import nn
from torch.utils.data import DataLoader, random_split

from vision import tasks
from vision.nn import available_models


class Task(LightningModule):
    TASK_TYPE = "None"

    def __init__(
        self,
        model: str | nn.Module,
        dataset_name: str,
        lr: float = 1e-3,
    ):
        super().__init__()

        self.lr = lr

        self.dataset = tasks.available_datasets[dataset_name]
        if isinstance(model, str):
            self.model = available_models[model](
                num_classes=self.dataset["num_classes"],
                in_channels=self.dataset["num_channels"],
            )
        elif issubclass(type(model), nn.Module):
            self.model = model
        else:
            raise ValueError(f"Model {model} is not found in available models or is not nn.Module.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class DataModule(LightningDataModule):
    TASK_TYPE = "None"

    def __init__(
        self,
        dataset_name: str,
        data_dir: str | pathlib.Path,
        batch_size: int = 32,
        train_ratio: float = 0.9,
        num_workers: int = 2,
        download: bool = False,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.train_ratio = train_ratio
        self.num_workers = num_workers
        self.download = download

        self.dataset = tasks.available_datasets[dataset_name]
        self.transform = self.dataset["transform"]

    def prepare_data(self):
        try:
            self.dataset["class"](self.data_dir, split="train", download=self.download)
            self.dataset["class"](self.data_dir, split="val", download=self.download)
            self.dataset["class"](self.data_dir, split="test", download=self.download)
        except TypeError:
            self.dataset["class"](self.data_dir, train=True, download=self.download)
            self.dataset["class"](self.data_dir, train=False, download=self.download)
        except ValueError:
            raise ValueError("Dataset not found.")

    def setup(self, stage: str):
        if stage == "fit" or stage is None:
            try:
                self.train_data = self.dataset["class"](
                    self.data_dir, split="train", transform=self.transform
                )
                self.val_data = self.dataset["class"](
                    self.data_dir, split="val", transform=self.transform
                )
            except TypeError:
                self.data = self.dataset["class"](
                    self.data_dir, train=True, transform=self.transform
                )
                len_train = int(len(self.data) * self.train_ratio)
                len_val = len(self.data) - len_train
                self.train_data, self.val_data = random_split(self.data, [len_train, len_val])

        if stage == "test" or stage is None:
            try:
                self.test_data = self.dataset["class"](
                    self.data_dir, split="test", transform=self.transform
                )
            except TypeError:
                self.test_data = self.dataset["class"](
                    self.data_dir, train=False, transform=self.transform
                )

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
            self.test_data,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
        )

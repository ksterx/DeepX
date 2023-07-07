import dataclasses
import pathlib

import torch
from lightning import LightningDataModule, LightningModule, Trainer
from torch import nn
from torch.utils.data import DataLoader, random_split

from deepx import (
    registered_datasets,
    registered_losses,
    registered_models,
    registered_tasks,
)


class TaskX(LightningModule):
    TASK_TYPE = "None"

    def __init__(
        self,
        lr: float = 1e-3,
        loss_fn: nn.Module | str = nn.CrossEntropyLoss(),
        optimizer: str | torch.optim.Optimizer = "adam",
    ):
        super().__init__()

        self.model = lambda x: x
        self.lr = lr
        self.optimizer = optimizer

        # self.dataset = registered_datasets[dataset_name]

        # # Model
        # if isinstance(model, str):
        #     self.model = registered_models[model](
        #         num_classes=self.dataset["num_classes"],
        #         in_channels=self.dataset["num_channels"],
        #     )
        # elif issubclass(type(model), nn.Module):
        #     self.model = model  # ko
        # else:
        #     raise ValueError(f"Model {model} is not found in available models or is not nn.Module.")

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


class DataModuleX(LightningDataModule):
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

        self.dataset = registered_datasets[dataset_name]
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


class Manager:
    def __init__(
        self,
        task: str | LightningModule,
        data: LightningDataModule | None = None,
        task_config: dict | None = None,
        dataset_config: dict | None = None,
        trainer_config: dict | None = None,
        **kwargs,
    ):
        self.task = task
        self.kwargs = kwargs
        if isinstance(task, str):


            if dataset_config is None:
                dataset_config = {}
            else:
                dataset_config = overwrite_config(kwargs, dataset_config)

            if task_config is None:
                task_config = {}
            else:
                task_config = overwrite_config(kwargs, task_config)

        elif issubclass(task, LightningModule):


        if isinstance(model, str):
            self.model = model

        self.trainer = Trainer(
            max_epochs=epochs,
            accelerator="auto",
            devices=1,
            logger=mlf_logger,
            enable_checkpointing=True,
            callbacks=[
                EarlyStopping(monitor="val_loss", patience=stopping_patience),
                ModelSummary(max_depth=max_depth),
            ],
            benchmark=benchmark,
            fast_dev_run=debug,
        )

    def train(self):
        self.trainer.fit(self.task, datamodule=self.dataset, ckpt_path=ckpt_path)
        if not debug:
            self.trainer.test(ckpt_path="best", datamodule=self.dataset)


def overwrite_config(a: dict, b: dict) -> dict:
    merged = a.copy()
    merged.update(b)
    return merged

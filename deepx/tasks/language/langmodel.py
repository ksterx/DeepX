import pathlib
from typing import Any

import torch
from lightning import LightningModule
from torch import nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from deepx.tasks import DataModuleX, TaskX


class LangModel(TaskX):
    NAME = "langmodel"

    def __init__(
        self,
        model: str | LightningModule,
        lr: float = 0.001,
        loss_fn: nn.Module | str = nn.CrossEntropyLoss(),
        optimizer: str | torch.optim.Optimizer = "adam",
        tokenizer: str | Any = "bert-base-uncased",
        max_length: int = 128,
        **kwargs,
    ):
        super().__init__(model=model, lr=lr, loss_fn=loss_fn, optimizer=optimizer, **kwargs)

        if isinstance(tokenizer, str):
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        else:
            self.tokenizer = tokenizer
        self.max_length = max_length

    def _mode_step(self, batch, batch_idx, mode: str):
        x = batch[:, :-1]
        y = batch[:, 1:]
        logits = self(x)
        loss = self.loss_fn(logits, y)
        self.log(f"{mode}_loss", loss)

        return loss

    def tokenize(self, text: str | list[str]):
        return self.tokenizer(
            text,
            return_tensors="pt",
            padding="max_length",
            max_length=self.max_length + 1,
            truncation=True,
        )


class LangModelDM(DataModuleX):
    def __init__(
        self,
        tokenizer: str,
        max_length: int,
        data_dir: str | pathlib.Path,
        batch_size: int = 32,
        train_ratio: float = 0.9,
        num_workers: int = 2,
        download: bool = False,
    ):
        super().__init__(
            data_dir=data_dir,
            batch_size=batch_size,
            train_ratio=train_ratio,
            num_workers=num_workers,
            download=download,
        )
        if isinstance(tokenizer, str):
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        else:
            self.tokenizer = tokenizer
        self.max_length = max_length

    def train_dataloader(self):
        return DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            collate_fn=self._collate_fn,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_data,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self._collate_fn,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_data,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self._collate_fn,
            pin_memory=True,
        )

    def predict_dataloader(self):
        return DataLoader(
            self.predict_data,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self._collate_fn,
            pin_memory=True,
        )

    def _collate_fn(self, batch):
        return tokenize(self.tokenizer, batch, max_length=self.max_length)["input_ids"]


def tokenize(tokenizer, text: str | list[str], max_length: int):
    return tokenizer(
        text,
        return_tensors="pt",
        padding="max_length",
        max_length=max_length + 1,
        truncation=True,
    )

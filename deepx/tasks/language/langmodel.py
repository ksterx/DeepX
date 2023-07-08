from typing import Any

import torch
from lightning import LightningModule
from torch import nn
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

    def _mode_step(self, batch, mode: str):
        src = self.tokenizer.encode(
            batch,
            return_tensors="pt",
            padding="max_length",
            max_length=self.max_length + 1,
            truncation=True,
        )
        x = src[:, :-1]
        y = src[:, 1:]
        logits = self(x)
        loss = self.loss_fn(logits, y)
        self.log(f"{mode}_loss", loss)


class LangModelDM(DataModuleX):
    pass

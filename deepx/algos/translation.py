from typing import Any

import torch
from lightning import LightningModule
from torch import nn
from torchmetrics import Accuracy
from transformers import AutoTokenizer

from .algo import Algorithm


class Translation(Algorithm):
    NAME = "langmodel"

    def __init__(
        self,
        model: str | LightningModule,
        lr: float = 0.001,
        loss_fn: nn.Module | str = "ce",
        optimizer: str | torch.optim.Optimizer = "adam",
        scheduler: str | torch.optim.lr_scheduler._LRScheduler = "cos",
        tokenizer: str | Any = "bert-base-uncased",
        max_length: int = 128,
        **kwargs,
    ):
        super().__init__(
            model=model, lr=lr, loss_fn=loss_fn, optimizer=optimizer, scheduler=scheduler, **kwargs
        )

        if isinstance(tokenizer, str):
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        else:
            self.tokenizer = tokenizer
        self.max_length = max_length

        self.train_acc = Accuracy(task="multiclass", num_classes=self.tokenizer.vocab_size)
        self.val_acc = Accuracy(task="multiclass", num_classes=self.tokenizer.vocab_size)
        self.test_acc = Accuracy(task="multiclass", num_classes=self.tokenizer.vocab_size)

    def _mode_step(self, batch, batch_idx, mode: str):
        x = batch[:, :-1]
        y = batch[:, 1:]
        y = y.contiguous().view(-1)
        logits, sim = self(x)

        # for debugging
        # text = self.tokenizer.decode(x[0])
        # print(f"Text: {text}")
        # pred = self.tokenizer.decode(logits[0].argmax(dim=-1))
        # print(f"Prediction: {pred}")

        logits = logits.view(-1, logits.size(-1))
        loss = self.loss_fn(logits, y)

        exec(f"self.{mode}_acc.update(logits, y)")
        self.log(f"{mode}_acc", eval(f"self.{mode}_acc"), prog_bar=True)
        self.log(f"{mode}_loss", loss)

        return loss

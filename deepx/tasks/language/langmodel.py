import pathlib

import numpy as np
import torch
from lightning import LightningModule
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchtext.datasets import WikiText103
from transformers import AutoTokenizer

from deepx.nn import Transformer
from deepx.tasks import DataModule, Task


class LangModelTask(Task):
    def __init__(
        self,
        model: str | LightningModule,
        dataset_name: str,
        lr: float = 0.001,
        loss_fn: nn.Module | str = nn.CrossEntropyLoss(),
        tokenizer: str = "bert-base-uncased",
        **kwargs,
    ):
        super().__init__(model=model, dataset_name=dataset_name, lr=lr, loss_fn=loss_fn)

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)

    def training_step(self, batch, batch_idx):
        x = self.tokenizer.encode(batch, return_tensors="pt")
        logits = self(x)
        loss = self.loss_fn(logits, x)


class WikiText103Dataset(DataModule):
    def setup(self, stage: str):
        self.train_data, self.val_data, self.test_data = WikiText103(
            root=self.data_dir, split=("train", "valid", "test")
        )

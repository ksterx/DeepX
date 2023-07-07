import pathlib

import numpy as np
import torch
from lightning import LightningModule
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchmetrics.classification import MulticlassJaccardIndex
from torchtext.datasets import WikiText103
from transformers import AutoTokenizer

from deepx.nn import LangModelTransformer
from deepx.tasks import DataModuleX, TaskX


class LangModelTask(TaskX):
    def __init__(
        self,
        model: str | LightningModule,
        dataset_name: str,
        lr: float = 0.001,
        loss_fn: nn.Module | str = nn.CrossEntropyLoss(),
        tokenizer: str = "bert-base-uncased",
        max_length: int = 128,
        embed_dim: int = 512,
        num_heads: int = 8,
        hidden_dim: int = 2048,
        num_blocks: int = 6,
        dropout: float = 0.0,
        **kwargs,
    ):
        super().__init__(model=model, dataset_name=dataset_name, lr=lr, loss_fn=loss_fn)

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        self.vocab_size = self.tokenizer.vocab_size
        self.max_length = max_length

        self.model = LangModelTransformer(
            vocab_size=self.vocab_size,
            embed_dim=embed_dim,
            num_heads=num_heads,
            hidden_dim=hidden_dim,
            num_blocks=num_blocks,
            dropout=dropout,
        )

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


class WikiText103Dataset(DataModuleX):
    def setup(self, stage=None):
        self.train_data, self.valid_data, self.test_data = WikiText103(
            self.data_dir, split=("train", "valid", "test")
        )


if __name__ == "__main__":
    from lightning import Trainer

    task = LangModelTask(model="resnet18", dataset_name="mnist")

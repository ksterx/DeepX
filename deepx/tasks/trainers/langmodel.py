import torch
from lightning import LightningDataModule, LightningModule
from torch import nn
from transformers import AutoTokenizer

from .trainer import TrainerX


class LangModelTrainer(TrainerX):
    TASK_TYPE = "langmodel"

    def __init__(
        self,
        model: str | LightningModule,
        datamodule: str | LightningDataModule,
        batch_size: int = 32,
        train_ratio: float = 0.8,
        num_workers: int = 2,
        download: bool = False,
        lr: float = 1e-3,
        loss_fn: str | nn.Module = nn.CrossEntropyLoss(),
        optimizer: str | torch.optim.Optimizer = "adam",
        root_dir: str = "/workspace",
        data_dir: str = "/workspace/data",
        log_dir: str = "/workspace/experiments",
        tokenizer: str = "bert-base-uncased",
        max_length: int = 128,
        embed_dim: int = 512,
        num_heads: int = 8,
        hidden_dim: int = 2048,
        num_blocks: int = 6,
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
            root_dir=root_dir,
            data_dir=data_dir,
            log_dir=log_dir,
            **kwargs,
        )

        self.hparams.update({})

        tokenizer = AutoTokenizer.from_pretrained(tokenizer)

        self.dm_cfg.update({"tokenizer": tokenizer, "max_length": max_length})
        self.datamodule = self.get_datamodule(datamodule=datamodule, **self.dm_cfg)

        self.model_cfg.update(
            {
                "vocab_size": tokenizer.vocab_size,
                "embed_dim": embed_dim,
                "num_heads": num_heads,
                "hidden_dim": hidden_dim,
                "num_blocks": num_blocks,
                "dropout": dropout,
            }
        )
        self.model = self.get_model(model, **self.model_cfg)

        self.task_cfg.update(
            {
                "model": self.model,
                "tokenizer": tokenizer,
                "max_length": max_length,
            }
        )
        self.task = self.get_task(task=self.TASK_TYPE, **self.task_cfg)

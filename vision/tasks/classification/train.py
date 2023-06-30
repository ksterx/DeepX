import pathlib
from typing import Any

import torch
from lightning import Trainer
from lightning.pytorch.callbacks import EarlyStopping, ModelSummary
from lightning.pytorch.loggers import TensorBoardLogger
from torch import nn
from torchmetrics import Accuracy

from vision.tasks import DataModule, Task


class ClassificationTask(Task):
    TASK_TYPE = "classification"

    def __init__(
        self,
        model: str | nn.Module,
        dataset_name: str,
        lr: float = 1e-3,
    ):
        super().__init__(model=model, dataset_name=dataset_name, lr=lr)

        self.loss_fn = torch.nn.CrossEntropyLoss()

        self.train_accuracy = Accuracy(task="multiclass", num_classes=self.dataset["num_classes"])
        self.val_accuracy = Accuracy(task="multiclass", num_classes=self.dataset["num_classes"])
        self.test_accuracy = Accuracy(task="multiclass", num_classes=self.dataset["num_classes"])

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        preds = torch.argmax(logits, dim=1)
        self.train_accuracy.update(preds, y)

        self.log("train_loss", loss)
        self.log("train_acc", self.train_accuracy)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        preds = torch.argmax(logits, dim=1)
        self.val_accuracy.update(preds, y)

        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", self.val_accuracy, prog_bar=True)

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        x = batch
        logits = self(x)
        preds = torch.argmax(logits, dim=1)
        return preds

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        preds = torch.argmax(logits, dim=1)
        self.test_accuracy.update(preds, y)

        self.log("test_loss", loss, prog_bar=True)
        self.log("test_acc", self.test_accuracy, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


class ClassificationDataset(DataModule):
    TASK_TYPE = "classification"


def train(
    model: str | nn.Module,
    dataset_name: str,
    root_dir: str | pathlib.Path,
    epochs: int = 2,
    batch_size: int = 32,
    is_test: bool = False,
    num_workers: int = 2,
    lr: float = 1e-3,
    stopping_patience: int = 5,
    max_depth: int = 1,
    download: bool = False,
) -> None:
    root_dir = pathlib.Path(root_dir)
    data_dir = root_dir / "data"
    if isinstance(model, str):
        model_name = model
    else:
        model_name = model.__class__.__name__

    print(f"Root directory: {root_dir.resolve()}")

    model = ClassificationTask(model=model, dataset_name=dataset_name, lr=lr)
    datamodule = ClassificationDataset(
        dataset_name=dataset_name,
        data_dir=data_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        download=download,
    )

    if is_test:
        log_dir = root_dir / f"experiments/tests/{dataset_name}"
    else:
        log_dir = root_dir / f"experiments/{dataset_name}"

    trainer = Trainer(
        max_epochs=epochs,
        accelerator="auto",
        devices=1,
        logger=TensorBoardLogger(log_dir, name=model_name),
        enable_checkpointing=True,
        callbacks=[
            EarlyStopping(monitor="val_loss", patience=stopping_patience),
            ModelSummary(max_depth=max_depth),
        ],
    )

    trainer.fit(model, datamodule=datamodule)
    trainer.test(ckpt_path="best", datamodule=datamodule)


if __name__ == "__main__":
    import argparse

    from vision.nn import available_models
    from vision.tasks import available_datasets

    # Argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="resnet50",
        choices=available_models.keys(),
        required=True,
    )
    parser.add_argument(
        "-d",
        "--dataset_name",
        type=str,
        default="mnist",
        choices=available_datasets["classification"].keys(),
        required=True,
    )
    parser.add_argument("-e", "--epochs", type=int, default=2)
    parser.add_argument("-b", "--batch_size", type=int, default=32)
    parser.add_argument("-t", "--is_test", action="store_true")
    parser.add_argument("-w", "--num_workers", type=int, default=2)
    parser.add_argument("-l", "--lr", type=float, default=1e-3)
    parser.add_argument("-p", "--stopping_patience", type=int, default=5)
    parser.add_argument("-r", "--root_dir", type=str, default="/workspace")
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()

    train(
        model=args.model,
        dataset_name=args.dataset_name,
        root_dir=args.root_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        is_test=args.is_test,
        num_workers=args.num_workers,
        lr=args.lr,
        stopping_patience=args.stopping_patience,
        download=args.download,
    )

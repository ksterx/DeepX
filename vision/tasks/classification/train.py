import pathlib
from typing import Any

import torch
from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import EarlyStopping, ModelSummary
from lightning.pytorch.loggers import TensorBoardLogger
from torch.utils.data import DataLoader, random_split
from torchmetrics import Accuracy
from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100, MNIST

from vision.nn import available_models


class ClassificationTask(LightningModule):
    DATASETS = {
        "mnist": {"class": MNIST, "num_classes": 10, "num_channels": 1},
        "cifar10": {"class": CIFAR10, "num_classes": 10, "num_channels": 3},
        "cifar100": {"class": CIFAR100, "num_classes": 100, "num_channels": 3},
    }

    def __init__(
        self,
        model_name: str,
        dataset_name: str,
        data_dir: str | pathlib.Path,
        batch_size: int = 32,
        train_ratio: float = 0.9,
    ):
        super().__init__()
        self.dataset = self.DATASETS[dataset_name]
        self.model = available_models[model_name](
            num_classes=self.dataset["num_classes"],
            in_channels=self.dataset["num_channels"],
        )
        self.dataset_name = dataset_name
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.train_ratio = train_ratio

        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ]
        )

        self.loss_fn = torch.nn.CrossEntropyLoss()

        self.train_accuracy = Accuracy(task="multiclass", num_classes=self.dataset["num_classes"])
        self.val_accuracy = Accuracy(task="multiclass", num_classes=self.dataset["num_classes"])
        self.test_accuracy = Accuracy(task="multiclass", num_classes=self.dataset["num_classes"])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

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
        return torch.optim.Adam(self.parameters(), lr=1e-3)

    def prepare_data(self):
        # download
        self.dataset["class"](self.data_dir, train=True, download=True)
        self.dataset["class"](self.data_dir, train=False, download=True)

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            data_full = self.dataset["class"](self.data_dir, train=True, transform=self.transform)
            len_train = int(len(data_full) * self.train_ratio)
            len_val = len(data_full) - len_train
            self.data_train, self.data_val = random_split(data_full, [len_train, len_val])

        if stage == "test" or stage is None:
            self.data_test = self.dataset["class"](
                self.data_dir, train=False, transform=self.transform
            )

    def train_dataloader(self):
        return DataLoader(self.data_train, batch_size=self.batch_size, num_workers=10)

    def val_dataloader(self):
        return DataLoader(self.data_val, batch_size=self.batch_size, num_workers=10)

    def test_dataloader(self):
        return DataLoader(self.data_test, batch_size=self.batch_size, num_workers=10)


def train(
    model_name: str,
    dataset_name: str,
    root_dir: str | pathlib.Path,
    epochs: int = 2,
    batch_size: int = 32,
    is_test: bool = False,
) -> None:
    root_dir = pathlib.Path(root_dir)
    data_dir = root_dir / "data"
    print(f"Root directory: {root_dir.resolve()}")
    model = ClassificationTask(
        model_name=model_name, dataset_name=dataset_name, data_dir=data_dir, batch_size=batch_size
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
        callbacks=[EarlyStopping(monitor="val_loss", patience=5), ModelSummary(max_depth=1)],
    )

    trainer.fit(model)
    trainer.test(ckpt_path="best")


if __name__ == "__main__":
    import argparse

    # Argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--model_name",
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
        choices=ClassificationTask.DATASETS.keys(),
        required=True,
    )
    parser.add_argument("-e", "--epochs", type=int, default=3)
    parser.add_argument("-b", "--batch_size", type=int, default=32)
    parser.add_argument("-t", "--is_test", action="store_true")
    args = parser.parse_args()

    train(
        model_name=args.model_name,
        dataset_name=args.dataset_name,
        root_dir="../../..",
        epochs=args.epochs,
        batch_size=args.batch_size,
        is_test=args.is_test,
    )

    # model_names = available_models.keys()
    # for model_name in model_names:
    #     train(model_name, "cifar10", epochs=100, batch_size=128, is_test=False)

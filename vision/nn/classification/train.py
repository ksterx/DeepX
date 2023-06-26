import lightning as L
import torch
from lightning.pytorch.callbacks import EarlyStopping, ModelSummary
from lightning.pytorch.loggers import TensorBoardLogger
from torch.utils.data import DataLoader, random_split
from torchmetrics import Accuracy
from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100, MNIST

from vision.nn import available_models


class ImageClassificationModel(L.LightningModule):
    DATA_DIR = "../../../data/"
    TRAIN_RATIO = 0.9
    DATASETS = {
        "mnist": {"class": MNIST, "num_classes": 10, "num_channels": 1},
        "cifar10": {"class": CIFAR10, "num_classes": 10, "num_channels": 3},
        "cifar100": {"class": CIFAR100, "num_classes": 100, "num_channels": 3},
    }
    MODELS = available_models

    def __init__(self, model_name: str, dataset_name: str, batch_size: int = 32):
        super().__init__()
        self.dataset = self.DATASETS[dataset_name]
        self.model = self.MODELS[model_name](
            num_classes=self.dataset["num_classes"],
            in_channels=self.dataset["num_channels"],
        )
        self.dataset_name = dataset_name
        self.batch_size = batch_size

        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ]
        )

        self.loss_fn = torch.nn.CrossEntropyLoss()

        self.train_accuracy = Accuracy(task="multiclass", num_classes=self.NUM_CLASSES)
        self.val_accuracy = Accuracy(task="multiclass", num_classes=self.NUM_CLASSES)
        self.test_accuracy = Accuracy(task="multiclass", num_classes=self.NUM_CLASSES)

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
        self.dataset["class"](self.DATA_DIR, train=True, download=True)
        self.dataset["class"](self.DATA_DIR, train=False, download=True)

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            data_full = self.dataset["class"](self.DATA_DIR, train=True, transform=self.transform)
            len_train = int(len(data_full) * self.TRAIN_RATIO)
            len_val = len(data_full) - len_train
            self.data_train, self.data_val = random_split(data_full, [len_train, len_val])

        if stage == "test" or stage is None:
            self.data_test = self.dataset["class"](
                self.DATA_DIR, train=False, transform=self.transform
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
    epochs: int = 3,
    batch_size: int = 32,
    is_test: bool = False,
) -> None:
    model = ImageClassificationModel(
        model_name=model_name, dataset_name=dataset_name, batch_size=batch_size
    )

    if is_test:
        log_dir = f"./experiments/tests/{dataset_name}"
    else:
        log_dir = f"./experiments/{dataset_name}"

    trainer = L.Trainer(
        max_epochs=epochs,
        accelerator="auto",
        devices=1,
        logger=TensorBoardLogger(log_dir, name=model_name),
        callbacks=[EarlyStopping(monitor="val_loss", patience=5), ModelSummary(max_depth=1)],
    )

    trainer.fit(model)
    trainer.test(ckpt_path="best")


if __name__ == "__main__":
    import argparse

    # Argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="resnet50")
    parser.add_argument("--dataset_name", type=str, default="cifar10")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--is_test", type=bool, default=False)
    args = parser.parse_args()

    train(args.model_name, args.dataset_name, args.epochs, args.batch_size, args.is_test)

    # model_names = available_models.keys()
    # for model_name in model_names:
    #     train(model_name, "cifar10", epochs=100, batch_size=128, is_test=False)

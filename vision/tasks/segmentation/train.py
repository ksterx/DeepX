import pathlib

import torch
from lightning import LightningModule
from torchmetrics import Dice, JaccardIndex, MetricCollection
from torchvision.datasets import VOCSegmentation

from vision.tasks import DataModule, Task


class SegmentationTask(Task):
    TASK_TYPE = "segmentation"

    def __init__(
        self,
        model: str | LightningModule,
        dataset_name: str,
        lr: float = 1e-3,
    ):
        super().__init__(model=model, dataset_name=dataset_name, lr=lr)

        self.loss_fn = torch.nn.CrossEntropyLoss()

        self.train_metrics = MetricCollection(
            [
                JaccardIndex(task="multiclass", num_classes=self.dataset["num_classes"]),
                Dice(task="multiclass", num_classes=self.dataset["num_classes"]),
            ],
            prefix="train_",
        )
        self.train_iou = JaccardIndex(task="multiclass", num_classes=self.dataset["num_classes"])
        self.val_iou = JaccardIndex(task="multiclass", num_classes=self.dataset["num_classes"])
        self.test_iou = JaccardIndex(task="multiclass", num_classes=self.dataset["num_classes"])

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        y = y.squeeze(1)  # (B, 1, H, W) -> (B, H, W)
        loss = self.loss_fn(logits, y)
        self.train_metrics(logits, y)
        self.train_iou(logits, y)
        self.log_dict(self.train_metrics, on_step=False, on_epoch=True, logger=True, prog_bar=True)
        self.log("train_iou", self.train_iou)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        y = y.squeeze(1)
        loss = self.loss_fn(logits, y)
        self.log("val_loss", loss, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        y = y.squeeze(1)
        loss = self.loss_fn(logits, y)
        self.log("test_loss", loss, prog_bar=True)

    def predict_step(self, batch, batch_idx):
        x, _ = batch
        logits = self(x)
        return torch.argmax(logits, dim=1)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


class SegmentationDataset(DataModule):
    TASK_TYPE = "segmentation"

    def __init__(
        self,
        dataset_name: str,
        data_dir: str | pathlib.Path,
        batch_size: int = 32,
        train_ratio: float = 0.9,
        num_workers: int = 2,
        download: bool = False,
    ):
        super().__init__(
            dataset_name=dataset_name,
            data_dir=data_dir,
            batch_size=batch_size,
            train_ratio=train_ratio,
            num_workers=num_workers,
            download=download,
        )

        self.target_transform = self.dataset["target_transform"]

    def prepare_data(self):
        VOCSegmentation(self.data_dir, image_set="train", download=self.download)
        VOCSegmentation(self.data_dir, image_set="val", download=self.download)

    def setup(self, stage: str):
        if stage == "fit":
            self.train_data = VOCSegmentation(
                self.data_dir,
                image_set="train",
                download=False,
                transform=self.transform,
                target_transform=self.target_transform,
            )
            self.val_data = VOCSegmentation(
                self.data_dir,
                image_set="val",
                download=False,
                transform=self.transform,
                target_transform=self.target_transform,
            )
        elif stage == "test":
            self.test_data = VOCSegmentation(
                self.data_dir,
                image_set="val",
                download=False,
                transform=self.transform,
                target_transform=self.target_transform,
            )

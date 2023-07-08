import pathlib

import numpy as np
import torch
from lightning import LightningModule
from PIL import Image
from torch import nn
from torchmetrics.classification import MulticlassJaccardIndex
from torchvision.datasets import VOCSegmentation

from deepx.tasks import DataModuleX, TaskX


class Segmentation(TaskX):
    TASK_TYPE = "segmentation"

    def __init__(
        self,
        model: str | LightningModule,
        dataset_name: str,
        lr: float = 1e-3,
        loss_fn: nn.Module | str = nn.CrossEntropyLoss(),
        **kwargs,
    ):
        super().__init__(model=model, dataset_name=dataset_name, lr=lr, loss_fn=loss_fn)

        self.loss_fn = torch.nn.CrossEntropyLoss(ignore_index=255)

        self.train_iou = MulticlassJaccardIndex(
            num_classes=self.dataset["num_classes"],
            ignore_index=255,
        )
        self.val_iou = MulticlassJaccardIndex(
            num_classes=self.dataset["num_classes"],
            ignore_index=255,
        )
        self.test_iou = MulticlassJaccardIndex(
            num_classes=self.dataset["num_classes"],
            ignore_index=255,
        )

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        y = y.squeeze(1).long()  # (B, 1, H, W) -> (B, H, W)
        loss = self.loss_fn(logits, y)
        self.train_iou.update(logits, y)

        self.log("train_iou", self.train_iou)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        y = y.squeeze(1).long()
        loss = self.loss_fn(logits, y)
        self.val_iou.update(logits, y)

        self.log("val_iou", self.val_iou, prog_bar=True)
        self.log("val_loss", loss, prog_bar=True)

    def on_validation_epoch_end(self):
        car_img = Image.open("/workspace/data/images/car.jpg")
        car_img = self.dataset["transform"](car_img)
        car_img = car_img.unsqueeze(0).to(self.device)
        car_pred = self.predict_step(car_img, 0).squeeze().cpu().numpy()
        np.save("/workspace/data/images/car_pred.npy", car_pred)

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        y = y.squeeze(1).long()
        loss = self.loss_fn(logits, y)
        self.test_iou.update(logits, y)

        self.log("test_iou", self.test_iou)
        self.log("test_loss", loss)

    def predict_step(self, batch, batch_idx):
        x = batch
        logits = self(x)
        return torch.argmax(logits, dim=1)


class SegmentationDM(DataModuleX):
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

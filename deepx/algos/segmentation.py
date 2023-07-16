import numpy as np
import torch
from lightning import LightningModule
from PIL import Image
from torch import nn
from torchmetrics.classification import MulticlassJaccardIndex

from .algo import Algorithm


class Segmentation(Algorithm):
    NAME = "segmentation"

    def __init__(
        self,
        model: str | LightningModule,
        num_classes: int,
        lr: float = 1e-3,
        loss_fn: nn.Module | str = "ce",
        optimizer: str | torch.optim.Optimizer = "adam",
        scheduler: str | torch.optim.lr_scheduler._LRScheduler = "cos",
        **kwargs,
    ):
        super().__init__(
            model=model, lr=lr, loss_fn=loss_fn, optimizer=optimizer, scheduler=scheduler, **kwargs
        )

        self.train_iou = MulticlassJaccardIndex(
            num_classes=num_classes,
            ignore_index=255,
        )
        self.val_iou = MulticlassJaccardIndex(
            num_classes=num_classes,
            ignore_index=255,
        )
        self.test_iou = MulticlassJaccardIndex(
            num_classes=num_classes,
            ignore_index=255,
        )

    def on_validation_epoch_end(self):
        car_img = Image.open("/workspace/experiments/data/images/car.jpg")
        car_img = self.dataset["transform"](car_img)
        car_img = car_img.unsqueeze(0).to(self.device)
        car_pred = self.predict_step(car_img, 0).squeeze().cpu().numpy()
        np.save("/workspace/experiments/data/images/car_pred.npy", car_pred)

    def predict_step(self, batch, batch_idx):
        x = batch
        logits = self(x)
        return torch.argmax(logits, dim=1)

    def _mode_step(self, batch, batch_idx, mode):
        x, y = batch
        logits = self(x)
        y = y.squeeze(1).long()
        loss = self.loss_fn(logits, y)

        exec(f"self.{mode}_iou.update(logits, y)")
        self.log(f"{mode}_iou", eval(f"self.{mode}_iou"), prog_bar=True)
        self.log(f"{mode}_loss", loss)

        return loss

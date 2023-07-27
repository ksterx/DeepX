import tempfile

import torch
from lightning import LightningModule
from torch import nn, optim
from torch.nn import functional as F
from torchmetrics.classification import MulticlassJaccardIndex
from torchvision.utils import save_image

from ..utils.vision import denormalize
from ..utils.wrappers import watch_kwargs
from .algo import Algorithm


class Segmentation(Algorithm):
    NAME = "segmentation"

    @watch_kwargs
    def __init__(
        self,
        model: str | LightningModule,
        num_classes: int,
        lr: float = 1e-4,
        loss_fn: nn.Module | str = "ce",
        optimizer: str | optim.Optimizer = "adam",
        scheduler: str | optim.lr_scheduler._LRScheduler = "cos",
        beta1: float = 0.9,
        beta2: float = 0.999,
        **kwargs,
    ):
        super().__init__(
            model=model,
            lr=lr,
            loss_fn=loss_fn,
            optimizer=optimizer,
            scheduler=scheduler,
            beta1=beta1,
            beta2=beta2,
            ignore_index=255,
            **kwargs,
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

    def on_train_end(self):
        self._make_gif_from_images("pred_masks_*.png", "epoch", "training.gif")

    def on_validation_epoch_end(self):
        x, logits = self.val_step_outputs
        pred_masks = F.softmax(logits, dim=0).unsqueeze(1)
        x = denormalize(
            x,
            mean=(0.5, 0.5, 0.5),
            std=(0.5, 0.5, 0.5),
            is_batch=False,
            levels=2,
            dtype=torch.float,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            save_image(
                pred_masks,
                fp=f"{tmpdir}/pred_masks_{self.trainer.current_epoch:03d}.png",
            )
            save_image(x, fp=f"{tmpdir}/image.png")
            self.logger.experiment.log_artifacts(self.logger.run_id, tmpdir)

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

        # Log x, logits at the last step of every training epoch
        if mode == "val":
            self.val_step_outputs = (x[0], logits[0])

        return loss

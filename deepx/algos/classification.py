import torch
from torch import nn
from torchmetrics import Accuracy

from .algo import Algorithm


class Classification(Algorithm):
    NAME = "classification"

    def __init__(
        self,
        model: str | nn.Module,
        num_classes: int,
        lr: float = 1e-3,
        loss_fn: nn.Module | str = "ce",
        optimizer: str | torch.optim.Optimizer = "adam",
        scheduler: str | torch.optim.lr_scheduler._LRScheduler = "cos",
        **kwargs,
    ):
        super().__init__(
            model=model,
            lr=lr,
            loss_fn=loss_fn,
            optimizer=optimizer,
            scheduler=scheduler,
            **kwargs,
        )

        self.train_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.test_acc = Accuracy(task="multiclass", num_classes=num_classes)

    def predict_step(self, batch, batch_idx, dataloader_idx):
        x = batch
        logits = self(x)
        preds = torch.argmax(logits, dim=1)
        return preds

    def _mode_step(self, batch, batch_idx, mode):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        preds = torch.argmax(logits, dim=1)

        exec(f"self.{mode}_acc.update(preds, y)")
        self.log(f"{mode}_acc", eval(f"self.{mode}_acc"), prog_bar=True)
        self.log(f"{mode}_loss", loss, prog_bar=True)

        return loss

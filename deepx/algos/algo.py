import torch
from lightning import LightningModule
from torch import nn

from ..nn import registered_losses


class Algorithm(LightningModule):
    NAME: str

    def __init__(
        self,
        model,
        lr: float,
        loss_fn: nn.Module | str,
        optimizer: str | torch.optim.Optimizer,
        scheduler: str | torch.optim.lr_scheduler._LRScheduler,
        **kwargs,
    ):
        super().__init__()

        self.model = model
        self.lr = lr
        self.optimizer = optimizer
        self.scheduler = scheduler

        # Loss function
        if isinstance(loss_fn, str):
            self.loss_fn = registered_losses[loss_fn]()
        elif isinstance(loss_fn, nn.Module):
            self.loss_fn = loss_fn
        else:
            raise ValueError(f"Invalid loss function: {loss_fn}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def training_step(self, batch, batch_idx):
        return self._mode_step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        return self._mode_step(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        return self._mode_step(batch, batch_idx, "test")

    def _mode_step(self, batch, batch_idx, mode):
        return NotImplementedError

    def configure_optimizers(self):
        if isinstance(self.optimizer, str):
            match self.optimizer:
                case "adam":
                    optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
                case "sgd":
                    optimizer = torch.optim.SGD(self.parameters(), lr=self.lr)
                case _:
                    raise ValueError(f"Invalid optimizer: {self.optimizer}")
        else:
            optimizer = self.optimizer(self.parameters(), lr=self.lr)

        if isinstance(self.scheduler, str):
            match self.scheduler:
                case "cos":
                    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                        optimizer, T_max=self.trainer.max_epochs
                    )
                case "step":
                    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
                case "plateau":
                    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                        optimizer, mode="min", factor=0.1, patience=10
                    )
                case "coswarm":
                    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                        optimizer, T_0=10, T_mult=2
                    )
                case _:
                    raise ValueError(f"Invalid scheduler: {self.scheduler}")
        else:
            scheduler = self.scheduler(optimizer)

        return [optimizer], [scheduler]

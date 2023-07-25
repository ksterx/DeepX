import torch
from lightning import LightningModule
from torch import nn, optim

from ..nn import registered_losses


class Algorithm(LightningModule):
    NAME: str

    def __init__(
        self,
        model: nn.Module,
        lr: float,
        loss_fn: nn.Module | str,
        optimizer: str | optim.Optimizer,
        scheduler: str | optim.lr_scheduler._LRScheduler,
        beta1: float = 0.9,
        beta2: float = 0.999,
        **kwargs,
    ):
        """Base class for all task algorithms.

        Args:
            model (nn.Module): The model to train.
            lr (float): Learning rate.
            loss_fn (nn.Module | str): Loss function.
            optimizer (str | optim.Optimizer): Optimizer.
            scheduler (str | optim.lr_scheduler._LRScheduler): Learning rate scheduler.
            beta1 (float, optional): Adam beta1. Defaults to 0.9
            beta2 (float, optional): Adam beta2. Defaults to 0.999

        Raises:
            ValueError: Invalid loss function.
        """
        super().__init__()

        self.model = model
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
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
                    optimizer = optim.Adam(
                        self.parameters(), lr=self.lr, betas=(self.beta1, self.beta2)
                    )
                case "sgd":
                    optimizer = optim.SGD(self.parameters(), lr=self.lr)
                case _:
                    raise ValueError(f"Invalid optimizer: {self.optimizer}")
        else:
            optimizer = self.optimizer(self.parameters(), lr=self.lr)

        if isinstance(self.scheduler, str):
            match self.scheduler:
                case "cos":
                    scheduler = optim.lr_scheduler.CosineAnnealingLR(
                        optimizer, T_max=self.trainer.max_epochs
                    )
                case "step":
                    scheduler = optim.lr_scheduler.StepLR(
                        optimizer, step_size=30, gamma=0.1
                    )
                case "plateau":
                    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                        optimizer, mode="min", factor=0.1, patience=10
                    )
                case "coswarm":
                    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                        optimizer, T_0=10, T_mult=2
                    )
                case "none":
                    scheduler = None
                case _:
                    raise ValueError(f"Invalid scheduler: {self.scheduler}")
        else:
            scheduler = self.scheduler(optimizer)

        return [optimizer], [scheduler]

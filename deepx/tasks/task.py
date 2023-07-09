import torch
from lightning import LightningModule
from torch import nn

from ..nn import registered_losses


class TaskX(LightningModule):
    NAME = ""

    def __init__(
        self,
        model,
        lr: float,
        loss_fn: nn.Module | str,
        optimizer: str | torch.optim.Optimizer,
        **kwargs,
    ):
        super().__init__()

        self.model = model
        self.lr = lr
        self.optimizer = optimizer

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
            if self.optimizer == "adam":
                return torch.optim.Adam(self.parameters(), lr=self.lr)
            elif self.optimizer == "sgd":
                return torch.optim.SGD(self.parameters(), lr=self.lr)
            else:
                raise ValueError(f"Invalid optimizer: {self.optimizer}")
        else:
            return self.optimizer(self.parameters(), lr=self.lr)

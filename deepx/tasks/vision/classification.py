import torch
from torch import nn
from torchmetrics import Accuracy

from deepx.tasks import DataModuleX, TaskX


class ClassificationTask(TaskX):
    TASK_TYPE = "classification"

    def __init__(
        self,
        model: str | nn.Module,
        dataset_name: str,
        lr: float = 1e-3,
        loss_fn: nn.Module | str = nn.CrossEntropyLoss(),
        **kwargs,
    ):
        super().__init__(model=model, dataset_name=dataset_name, lr=lr, loss_fn=loss_fn)

        self.train_accuracy = Accuracy(task="multiclass", num_classes=self.dataset["num_classes"])
        self.val_accuracy = Accuracy(task="multiclass", num_classes=self.dataset["num_classes"])
        self.test_accuracy = Accuracy(task="multiclass", num_classes=self.dataset["num_classes"])

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

    def predict_step(self, batch, batch_idx, dataloader_idx):
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
        return torch.optim.Adam(self.parameters(), lr=self.lr)


class ClassificationDataset(DataModuleX):
    TASK_TYPE = "classification"

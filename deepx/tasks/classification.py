from dataclasses import dataclass

import torch
from torchmetrics import Accuracy

from .core import DataModuleConfig, ModelConfig, Task, TaskConfig, Trainer


class ClassificationModelConfig(ModelConfig):
    def __init__(
        self,
        dropout: float = 0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.dropout = dropout


class ClassificationTaskConfig(TaskConfig):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class ClassificationDMConfig(DataModuleConfig):
    def __init__(
        self,
        train_ratio: float,
        mean: tuple[float, ...],
        std: tuple[float, ...],
        download: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.train_ratio = train_ratio
        self.mean = mean
        self.std = std
        self.download = download


class ClassificationTask(Task):
    NAME = "classification"

    def __init__(
        self,
        model_cfg: ClassificationModelConfig,
        task_cfg: ClassificationTaskConfig,
    ):
        super().__init__(
            model_cfg=model_cfg,
            task_cfg=task_cfg,
        )

        num_classes = model_cfg.num_classes

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

        acc = eval(f"self.{mode}_acc")
        acc(preds, y)
        self.log(f"{mode}_acc", eval(f"self.{mode}_acc"), prog_bar=True)
        self.log(f"{mode}_loss", loss, prog_bar=True)

        return loss


class ClassificationTrainer(Trainer):
    NAME = "classification"

    def _update_configs(self):
        self.model_cfg.update(
            num_classes=self.dm.NUM_CLASSES,
            in_channels=self.dm.NUM_CHANNELS,
        )

    def _build_task(self, model_cfg: ModelConfig, task_cfg: TaskConfig) -> Task:
        return ClassificationTask(model_cfg=model_cfg, task_cfg=task_cfg)


@dataclass
class Classification:
    model_cfg: ModelConfig = ClassificationModelConfig
    task_cfg: TaskConfig = ClassificationTaskConfig
    dm_cfg: DataModuleConfig = ClassificationDMConfig
    trainer: Trainer = ClassificationTrainer

import torch
from torchmetrics import Accuracy

from deepx.tasks.core import DataModuleConfig, ModelConfig, Task, TaskConfig, Trainer


class ClassificationModelConfig(ModelConfig):
    def __init__(
        self,
        num_classes: int,
        in_channels: int,
        dropout: float = 0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.dropout = dropout


class ClassificationConfig(TaskConfig):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class ClassificationDMConfig(DataModuleConfig):
    def __init__(self, train_ratio: float, download: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.train_ratio = train_ratio
        self.download = download


class Classification(Task):
    NAME = "classification"

    def __init__(
        self,
        model_cfg: ClassificationModelConfig,
        task_cfg: ClassificationConfig,
    ):
        super().__init__(
            model_cfg=model_cfg,
            task_cfg=task_cfg,
        )

        self.initialize()

        num_classes = model_cfg.num_classes

        self.train_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.test_acc = Accuracy(task="multiclass", num_classes=num_classes)

    def initialize(self):
        super().initialize()

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
    # def __init__(
    #     self,
    #     model_cfg: ClassificationModelConfig,
    #     task_cfg: ClassificationConfig,
    #     dm_cfg: DataModuleConfig,
    #     **kwargs,
    # ):
    #     super().__init__(
    #         model_cfg=model_cfg,
    #         task_cfg=task_cfg,
    #         dm_cfg=dm_cfg,
    #         **kwargs,
    #     )

    #     self._update_configs()

    #     self.task = Classification(
    #         model_cfg=self.model_cfg, task_cfg=self.task_cfg, **kwargs
    #     )

    def _update_configs(self):
        self.model_cfg.update(
            {
                "num_classes": self.dm.NUM_CLASSES,
                "in_channels": self.dm.NUM_CHANNELS,
            }
        )

        # self.task_cfg.update({})

        # self.dm_cfg.update()

    def _build_task(self, model_cfg: ModelConfig, task_cfg: TaskConfig) -> Task:
        return Classification(model_cfg=model_cfg, task_cfg=task_cfg)

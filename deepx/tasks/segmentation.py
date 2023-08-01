import tempfile

import torch
from lightning import LightningModule
from torch import nn, optim
from torch.nn import functional as F
from torchmetrics.classification import MulticlassJaccardIndex
from torchvision.utils import save_image

from deepx.tasks.core import ModelConfig, Task, TaskConfig

from ..utils.vision import denormalize
from ..utils.wrappers import watch_kwargs
from .core import DataModuleConfig, ModelConfig, Task, TaskConfig, Trainer


class SegmentationModelConfig(ModelConfig):
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


class SegmentationConfig(TaskConfig):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class SegmentationDMConfig(DataModuleConfig):
    def __init__(self, train_ratio: float, download: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.train_ratio = train_ratio
        self.download = download


class Segmentation(Task):
    NAME = "segmentation"

    def __init__(
        self,
        model_cfg: SegmentationModelConfig,
        task_cfg: SegmentationConfig,
    ):
        super().__init__(
            model_cfg=model_cfg,
            task_cfg=task_cfg,
        )

        num_classes = model_cfg.num_classes

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


class SegmentationTrainer(Trainer):
    NAME = "segmentation"

    def _update_configs(self):
        pass

    def _build_task(self, model_cfg: ModelConfig, task_cfg: TaskConfig) -> Task:
        return Segmentation(model_cfg=model_cfg, task_cfg=task_cfg)

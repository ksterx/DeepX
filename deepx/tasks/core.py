import glob
import os
import tempfile
from abc import ABC, abstractmethod
from dataclasses import dataclass
from logging import getLogger

import lightning as L
from lightning import LightningDataModule, LightningModule
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, ModelSummary
from lightning.pytorch.loggers import MLFlowLogger, TensorBoardLogger
from torch import Tensor, nn, optim

from ..dms import dm_aliases
from ..nn import registered_losses, registered_models
from ..utils.vision import make_gif_from_images
from ..utils.wrappers import watch_kwargs

process_logger = getLogger(__name__)


@dataclass
class ModelConfig:
    model: str


@dataclass
class TaskConfig:
    lr: float
    loss_fn: str
    optimizer: str
    scheduler: str
    beta1: float
    beta2: float
    ignore_index: int = -100


@dataclass
class DataModuleConfig:
    dm: str
    data_dir: str
    batch_size: int
    num_workers: int


@dataclass
class TrainingConfig:
    ckpt_path: str | None
    epochs: int
    patience: int
    max_depth: int
    benchmark: bool
    debug: bool
    monitor_metric: str
    monitor_mode: str
    logging: bool
    logger: str
    accelerator: str
    devices: str | None
    root_dir: str
    log_dir: str


class Task(LightningModule, ABC):
    NAME: str

    def __init__(
        self,
        model_cfg: ModelConfig,
        task_cfg: TaskConfig,
    ):
        """Base class for all task algorithms.

        Args:

        """
        LightningModule.__init__(self)
        ABC.__init__(self)

        self.save_hyperparameters()
        self.mparams = self.hparams.model_cfg
        self.tparams = self.hparams.task_cfg

        self.model = self._build_model(**vars(model_cfg))

        self.loss_fn = self.configure_loss_fn(
            self.tparams.loss_fn, ignore_index=self.tparams.ignore_index
        )

    # @abstractmethod
    # def initialize(self):
    #     self.save_hyperparameters()  # This must be called to load a checkpoint

    def forward(self, x: Tensor) -> Tensor:
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
        if isinstance(self.tparams.optimizer, str):
            match self.tparams.optimizer:
                case "adam":
                    optimizer = optim.Adam(
                        self.parameters(),
                        lr=self.tparams.lr,
                        betas=(self.tparams.beta1, self.tparams.beta2),
                    )
                case "sgd":
                    optimizer = optim.SGD(self.parameters(), lr=self.tparams.lr)
                case _:
                    raise ValueError(f"Invalid optimizer: {self.tparams.optimizer}")
        else:
            optimizer = self.tparams.optimizer(self.parameters(), lr=self.tparams.lr)

        if isinstance(self.tparams.scheduler, str):
            match self.tparams.scheduler:
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
                    raise ValueError(f"Invalid scheduler: {self.tparams.scheduler}")
        else:
            scheduler = self.tparams.scheduler(optimizer)

        return [optimizer], [scheduler]

    def configure_loss_fn(self, loss_fn, ignore_index=-100):
        if loss_fn == "ce" and ignore_index != -100:
            return nn.CrossEntropyLoss(ignore_index=ignore_index)
        elif isinstance(loss_fn, str):
            return registered_losses[loss_fn]()
        elif isinstance(loss_fn, nn.Module):
            return loss_fn  # type: ignore
        else:
            raise ValueError(f"Invalid loss function: {loss_fn}")

    def _make_gif_from_images(
        self,
        src_filename: str,
        metric: str,
        tgt_filename: str = "out.gif",
        duration: int = 100,
        loop: int = 0,
    ):
        with tempfile.TemporaryDirectory() as tmpdir:
            img_paths = sorted(
                glob.glob(
                    os.path.join(
                        self.logger.save_dir,
                        self.logger.experiment_id,
                        self.logger.run_id,
                        "artifacts",
                        src_filename,
                    )
                )
            )

            save_path = os.path.join(tmpdir, tgt_filename)

            make_gif_from_images(
                img_paths=img_paths,
                save_path=save_path,
                metric=metric.title(),
                duration=duration,
                loop=loop,
            )

            self.logger.experiment.log_artifact(self.logger.run_id, save_path)

    def _build_model(self, model: str, **kwargs) -> nn.Module:
        try:
            model_cls = registered_models[model]
            return model_cls(**kwargs)
        except KeyError:
            raise ValueError(
                f"Model {model} not supported. Please register it at deepx/nn/__init__.py"
            )


class Trainer(ABC):
    NAME: str

    @watch_kwargs
    def __init__(
        self,
        model_cfg: ModelConfig,
        task_cfg: TaskConfig,
        dm_cfg: DataModuleConfig,
        train_cfg: TrainingConfig | None = None,
        **kwargs,
    ):
        """Base class for all trainers.

        Args:

        """
        super().__init__()

        self.model_cfg = model_cfg
        self.task_cfg = task_cfg
        self.dm_cfg = dm_cfg
        self.train_cfg = train_cfg

        # Register hyperparameters to be logged
        self.hparams = kwargs
        self.hparams.update(**vars(model_cfg))
        self.hparams.update(**vars(task_cfg))
        self.hparams.update(**vars(dm_cfg))
        if train_cfg is not None:
            self.hparams.update(**vars(train_cfg))

        self.dm = self._get_datamodule(**vars(dm_cfg))
        self._update_configs()
        self.task = self._build_task(model_cfg=model_cfg, task_cfg=task_cfg)

    def _get_datamodule(self, dm: str, **kwargs) -> LightningDataModule:
        try:
            dm_cls = dm_aliases[dm]
            return dm_cls(**kwargs)
        except KeyError:
            raise ValueError(
                f"Datamodule {dm} not supported. Please register it at deepx/dms/__init__.py"
            )

    @abstractmethod
    def _build_task(self, model_cfg: ModelConfig, task_cfg: TaskConfig) -> Task:
        pass

    @abstractmethod
    def _update_configs(self):
        pass

    def train(self, train_cfg: TrainingConfig | None = None):
        if self.train_cfg is None:
            if train_cfg is None:
                raise ValueError("No training config provided.")
            else:
                tcfg = train_cfg
        else:
            if train_cfg is None:
                tcfg = self.train_cfg
            else:
                process_logger.warning("Overriding training config.")
                tcfg = self.train_cfg

        if tcfg.logging:
            match tcfg.logger:
                case "mlflow":
                    logger = MLFlowLogger(
                        experiment_name=f"{self.dm.NAME}-{self.NAME}",
                        tracking_uri=f"file://{tcfg.log_dir}",
                    )
                    print(f"Experiment ID: {logger.experiment_id}")
                    print(f"Run ID: {logger.run_id}")

                    logger.log_hyperparams(self.hparams)

                case "tensorboard":
                    logger = TensorBoardLogger(
                        save_dir=tcfg.log_dir, name=self.dm_cfg.dm
                    )
        else:
            logger = None

        if tcfg.monitor_mode:
            monitor_mode = "max"
        else:
            monitor_mode = "min"

        self.summarize()

        self.trainer = L.Trainer(
            limit_train_batches=2,
            limit_val_batches=2,
            limit_test_batches=2,
            limit_predict_batches=2,
            max_epochs=tcfg.epochs,
            accelerator=tcfg.accelerator,
            devices=tcfg.devices,
            logger=logger,
            enable_checkpointing=True,
            callbacks=[
                EarlyStopping(
                    monitor=tcfg.monitor_metric,
                    patience=tcfg.patience,
                    mode=monitor_mode,
                ),
                ModelSummary(max_depth=tcfg.max_depth),
                ModelCheckpoint(
                    monitor=tcfg.monitor_metric,
                    save_top_k=1,
                    mode=monitor_mode,
                    save_last=True,
                ),
            ],
            benchmark=tcfg.benchmark,
            fast_dev_run=tcfg.debug,
            num_sanity_val_steps=0,
        )

        # self.task = torch.compile(self.task)  # ERROR: NotImplementedError
        self.trainer.fit(
            model=self.task,
            datamodule=self.dm,
            ckpt_path=tcfg.ckpt_path,
        )
        if not tcfg.debug:
            self.trainer.test(ckpt_path="best", datamodule=self.dm)

    def test(self, ckpt_path: str | None = None):
        self.trainer.test(ckpt_path=ckpt_path, datamodule=self.dm)

    def summarize(self):
        cfgs = {
            "datamodule": self.dm_cfg,
            "model": self.model_cfg,
            "task": self.task_cfg,
            "training": self.train_cfg,
        }

        for k, v in cfgs.items():
            print("==============================")
            print(f"{k.title()} Config:")
            for kk, vv in vars(v).items():
                print(f"\t{kk}: {vv}")

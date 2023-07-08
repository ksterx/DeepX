from lightning import LightningDataModule, LightningModule, Trainer
from lightning.pytorch.callbacks import EarlyStopping, ModelSummary
from lightning.pytorch.loggers import MLFlowLogger
from torch import nn, optim

from deepx import tasks
from deepx.nn import registered_models


class TrainerX:
    TASK_TYPE = ""

    def __init__(
        self,
        model: str | LightningModule,
        datamodule: str | LightningDataModule,
        batch_size: int,
        train_ratio: float,
        num_workers: int,
        download: bool,
        lr: float,
        loss_fn: str | nn.Module,
        optimizer: str | optim.Optimizer,
        root_dir: str,
        data_dir: str,
        log_dir: str,
        **kwargs,
    ):
        self.hparams = kwargs
        self.hparams.update(
            {
                "model": model,
                "datamodule": datamodule,
                "batch_size": batch_size,
                "train_ratio": train_ratio,
                "lr": lr,
                "loss_fn": loss_fn,
                "optimizer": optimizer,
            }
        )

        self.root_dir = root_dir
        self.data_dir = data_dir
        self.log_dir = log_dir
        self.batch_size = batch_size
        self.train_ratio = train_ratio
        self.num_workers = num_workers
        self.download = download

        self.dm_cfg = {
            "data_dir": data_dir,
            "batch_size": batch_size,
            "train_ratio": train_ratio,
            "num_workers": num_workers,
            "download": download,
        }

        self.model_cfg = {}

        self.task_cfg = {
            "lr": lr,
            "loss_fn": loss_fn,
            "optimizer": optimizer,
        }

    def get_datamodule(self, datamodule, **kwargs):
        if isinstance(datamodule, str):
            dm_cls = tasks.registered_tasks[self.TASK_TYPE]["datamodule"][datamodule]
            return dm_cls(**kwargs)
        else:
            return datamodule

    def get_model(self, model, **kwargs):
        if isinstance(model, str):
            model_cls = registered_models[model]
            return model_cls(**kwargs)
        else:
            return model

    def get_task(self, task, **kwargs):
        if isinstance(task, str):
            task_cls = tasks.registered_tasks[task]["task"]
            return task_cls(**kwargs)
        else:
            return task

    def train(
        self,
        ckpt_path: str | None = None,
        epochs: int = 2,
        stopping_patience: int = 5,
        max_depth: int = 1,
        benchmark: bool = False,
        debug: bool = False,
        **kwargs,
    ):
        self.hparams.update(kwargs)
        self.hparams.update(
            {"epochs": epochs, "stopping_patience": stopping_patience, "ckpt_path": ckpt_path}
        )
        self.set(
            epochs=epochs,
            stopping_patience=stopping_patience,
            max_depth=max_depth,
            benchmark=benchmark,
            debug=debug,
            logging=True,
            **kwargs,
        )
        self.trainer.fit(self.task, datamodule=self.datamodule, ckpt_path=ckpt_path)
        if not debug:
            self.trainer.test(ckpt_path="best", datamodule=self.datamodule)

    def test(self, ckpt_path: str | None = None):
        try:
            self.trainer.test(ckpt_path=ckpt_path, datamodule=self.datamodule)
        except AttributeError:
            self.set(logging=False)
            self.trainer.test(ckpt_path=ckpt_path, datamodule=self.datamodule)

    def set(
        self,
        epochs: int = 2,
        stopping_patience: int = 5,
        max_depth: int = 1,
        benchmark: bool = False,
        debug: bool = False,
        logging: bool = True,
        **kwargs,
    ):
        if logging:
            logger = MLFlowLogger(
                experiment_name=self.datamodule.NAME,
                tags={"model": self.model.NAME},
                tracking_uri=f"file://{self.log_dir}",
            )
            logger.log_hyperparams(self.hparams)
        else:
            logger = None

        self.trainer = Trainer(
            max_epochs=epochs,
            accelerator="gpu",
            devices=1,
            logger=logger,
            enable_checkpointing=True,
            callbacks=[
                EarlyStopping(monitor="val_loss", patience=stopping_patience),
                ModelSummary(max_depth=max_depth),
            ],
            benchmark=benchmark,
            fast_dev_run=debug,
            **kwargs,
        )

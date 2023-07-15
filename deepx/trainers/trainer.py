import torch
from lightning import LightningDataModule, LightningModule, Trainer
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, ModelSummary
from lightning.pytorch.loggers import MLFlowLogger, TensorBoardLogger
from torch import nn, optim

from deepx import registered_algos
from deepx.nn import registered_models


class TrainerX:
    NAME: str

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
        scheduler: str | optim.lr_scheduler._LRScheduler,
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
                "scheduler": scheduler,
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

        self.algo_cfg = {
            "lr": lr,
            "loss_fn": loss_fn,
            "optimizer": optimizer,
            "scheduler": scheduler,
        }

    def get_datamodule(self, datamodule, **kwargs):
        if isinstance(datamodule, str):
            try:
                dm_cls = registered_algos[self.NAME]["datamodule"][datamodule]
                return dm_cls(**kwargs)
            except KeyError:
                raise ValueError(
                    f"Datamodule {datamodule} not supported. Please register it at deepx/__init__.py"
                )
        else:
            return datamodule

    def get_model(self, model, **kwargs):
        if isinstance(model, str):
            try:
                model_cls = registered_models[model]
                return model_cls(**kwargs)
            except KeyError:
                raise ValueError(
                    f"Model {model} not supported. Please register it at deepx/nn/__init__.py"
                )
        else:
            return model

    def get_algo(self, algo, **kwargs):
        if isinstance(algo, str):
            try:
                algo_cls = registered_algos[algo]["algo"]
                return algo_cls(**kwargs)
            except KeyError:
                raise ValueError(
                    f"Algorithm {algo} not supported. Please register it at deepx/__init__.py"
                )
        else:
            return algo

    def train(
        self,
        ckpt_path: str | None = None,
        epochs: int = 2,
        stopping_patience: int = 5,
        max_depth: int = 1,
        benchmark: bool = False,
        debug: bool = False,
        monitor: str = "val_loss",
        monitor_max: bool = False,
        logger: str = "mlflow",
        **kwargs,
    ):
        self.hparams.update(kwargs)
        self.hparams.update(
            {
                "epochs": epochs,
                "stopping_patience": stopping_patience,
                "ckpt_path": ckpt_path,
                "monitor": monitor,
                "monitor_max": monitor_max,
            }
        )
        self.set(
            epochs=epochs,
            stopping_patience=stopping_patience,
            max_depth=max_depth,
            benchmark=benchmark,
            debug=debug,
            logging=True,
            logger=logger,
            monitor=monitor,
            monitor_max=monitor_max,
            **kwargs,
        )
        # self.algo = torch.compile(self.algo)  # ERROR: NotImplementedError
        self.trainer.fit(
            self.algo,
            datamodule=self.datamodule,
            ckpt_path=ckpt_path,
        )
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
        logger: str = "mlflow",
        monitor: str = "val_loss",
        monitor_max: bool = False,
        **kwargs,
    ):
        if logging:
            match logger:
                case "mlflow":
                    logger = MLFlowLogger(
                        experiment_name=f"{self.datamodule.NAME}-{self.NAME}",
                        tags={"model": self.model.NAME},
                        tracking_uri=f"file://{self.log_dir}",
                    )
                    print(f"Experiment ID: {logger.experiment_id}")
                    print(f"Run ID: {logger.run_id}")

                    logger.log_hyperparams(self.hparams)

                case "tensorboard":
                    logger = TensorBoardLogger(save_dir=self.log_dir, name=self.datamodule.NAME)
        else:
            logger = None

        if monitor_max:
            monitor_mode = "max"
        else:
            monitor_mode = "min"

        self.trainer = Trainer(
            max_epochs=epochs,
            accelerator="gpu",
            devices=1,
            logger=logger,
            enable_checkpointing=True,
            callbacks=[
                EarlyStopping(monitor=monitor, patience=stopping_patience),
                ModelSummary(max_depth=max_depth),
                ModelCheckpoint(monitor=monitor, save_top_k=1, mode=monitor_mode, save_last=True),
            ],
            benchmark=benchmark,
            fast_dev_run=debug,
            num_sanity_val_steps=0,
            **kwargs,
        )

        self.algo_summary()

    def algo_summary(self):
        for k, v in self.hparams.items():
            print(f"{k}: {v}")

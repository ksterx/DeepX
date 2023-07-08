from lightning import LightningDataModule, LightningModule, Trainer
from lightning.pytorch.callbacks import EarlyStopping, ModelSummary
from lightning.pytorch.loggers import MLFlowLogger

from deepx import registered_models, registered_tasks


class TrainerX:
    TASK_TYPE = ""

    def __init__(
        self,
        model: str | LightningModule,
        datamodule: str | LightningDataModule,
        data_dir: str = "/workspace",
        batch_size: int = 32,
        train_ratio: float = 0.8,
        num_workers: int = 2,
        download: bool = False,
        **kwargs,
    ):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.train_ratio = train_ratio
        self.num_workers = num_workers
        self.download = download

        # Set up datamodule
        if isinstance(datamodule, str):
            self.datamodule = registered_tasks[self.TASK_TYPE]["datamodule"][datamodule](
                data_dir=data_dir,
                batch_size=batch_size,
                train_ratio=train_ratio,
                num_workers=num_workers,
                download=download,
            )
        else:
            self.datamodule = datamodule

    def get_model(self, model, **kwargs):
        if isinstance(model, str):
            model_cls = registered_models[model]
            return model_cls(**kwargs)
        else:
            return model

    def get_task(self, task, **kwargs):
        if isinstance(task, str):
            task_cls = registered_tasks[task]["task"]
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
    ):
        self.set(
            epochs=epochs,
            stopping_patience=stopping_patience,
            max_depth=max_depth,
            benchmark=benchmark,
            debug=debug,
            logging=True,
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
    ):
        if logging:
            logger = MLFlowLogger(
                experiment_name=self.datamodule.name,
                tags={"model": self.model.name},
                tracking_uri=f"file://{log_dir}",
            )
            logger.log_hyperparams(task_kwargs)
        else:
            logger = None

        self.trainer = Trainer(
            max_epochs=epochs,
            accelerator="auto",
            devices=1,
            logger=logger,
            enable_checkpointing=True,
            callbacks=[
                EarlyStopping(monitor="val_loss", patience=stopping_patience),
                ModelSummary(max_depth=max_depth),
            ],
            benchmark=benchmark,
            fast_dev_run=debug,
        )

    def summary(self):
        print("Task")
        for k, v in vars(self.task).items():
            print(f"{k}: {v}")
        print("\nDataModule")
        for k, v in vars(self.datamodule).items():
            print(f"{k}: {v}")
        print("\nTrainer")
        for k, v in vars(self.trainer).items():
            print(f"{k}: {v}")


def overwrite_config(a: dict | None, b: dict) -> dict:
    if a is None:
        return b
    else:
        merged = a.copy()
        merged.update(b)
        return merged

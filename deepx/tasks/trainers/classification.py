from lightning import LightningDataModule, LightningModule
from torch import nn

from .trainer import TrainerX


class ClassificationTrainer(TrainerX):
    TASK_TYPE = "classification"

    def __init__(
        self,
        model: str | LightningModule,
        datamodule: str | LightningDataModule,
        root_dir: str = "/workspace",
        data_dir: str = "/workspace/data",
        log_dir: str = "/workspace/experiments",
        batch_size: int = 32,
        train_ratio: float = 0.8,
        num_workers: int = 2,
        download: bool = False,
        lr: float = 1e-3,
        loss_fn: str | nn.Module = nn.CrossEntropyLoss(),
        **kwargs,
    ):
        super().__init__(
            model=model,
            datamodule=datamodule,
            data_dir=data_dir,
            batch_size=batch_size,
            train_ratio=train_ratio,
            num_workers=num_workers,
            download=download,
            **kwargs,
        )

        # Set up model
        num_classes = self.datamodule.NUM_CLASSES
        in_channels = self.datamodule.NUM_CHANNELS
        self.model = self.get_model(model, num_classes=num_classes, in_channels=in_channels)

        # Set up task
        self.task = self.get_task(
            self.TASK_TYPE, model=self.model, num_classes=num_classes, lr=lr, loss_fn=loss_fn
        )


if __name__ == "__main__":
    trainer = ClassificationTrainer(model="resnet18", datamodule="mnist")
    trainer.train(debug=True)

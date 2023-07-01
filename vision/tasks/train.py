import pathlib

from lightning import Trainer
from lightning.pytorch.callbacks import EarlyStopping, ModelSummary
from lightning.pytorch.loggers import TensorBoardLogger
from torch import nn

from vision.tasks import (
    ClassificationDataset,
    ClassificationTask,
    SegmentationDataset,
    SegmentationTask,
)


def train(
    task: str,
    model: str | nn.Module,
    dataset_name: str,
    root_dir: str | pathlib.Path,
    epochs: int = 2,
    batch_size: int = 32,
    debug: bool = False,
    num_workers: int = 2,
    lr: float = 1e-3,
    stopping_patience: int = 5,
    max_depth: int = 1,
    download: bool = False,
) -> None:
    root_dir = pathlib.Path(root_dir)
    data_dir = root_dir / "data"
    if isinstance(model, str):
        model_name = model
    else:
        model_name = model.__class__.__name__

    print(f"Root directory: {root_dir.resolve()}")

    task_kwargs = {"model": model, "dataset_name": dataset_name, "lr": lr}
    dataset_kwargs = {
        "dataset_name": dataset_name,
        "data_dir": data_dir,
        "batch_size": batch_size,
        "num_workers": num_workers,
        "download": download,
    }

    if task == "classification":
        model = ClassificationTask(**task_kwargs)  # type: ignore
        datamodule = ClassificationDataset(**dataset_kwargs)  # type: ignore

    elif task == "segmentation":
        model = SegmentationTask(**task_kwargs)  # type: ignore
        datamodule = SegmentationDataset(**dataset_kwargs)  # type: ignore

    if debug:
        log_dir = root_dir / f"experiments/tests/{dataset_name}"
    else:
        log_dir = root_dir / f"experiments/{dataset_name}"

    trainer = Trainer(
        max_epochs=epochs,
        accelerator="auto",
        devices=1,
        logger=TensorBoardLogger(log_dir, name=model_name),
        enable_checkpointing=True,
        callbacks=[
            EarlyStopping(monitor="val_loss", patience=stopping_patience),
            ModelSummary(max_depth=max_depth),
        ],
        fast_dev_run=debug,
    )

    trainer.fit(model, datamodule=datamodule)
    trainer.test(ckpt_path="best", datamodule=datamodule)


if __name__ == "__main__":
    import argparse

    from vision.nn import available_models
    from vision.tasks import available_datasets

    # Argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--task", type=str, default="classification", required=True)
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="resnet50",
        choices=available_models.keys(),
        required=True,
    )
    parser.add_argument(
        "-ds",
        "--dataset_name",
        type=str,
        default="mnist",
        choices=available_datasets.keys(),
        required=True,
    )
    parser.add_argument("-e", "--epochs", type=int, default=2)
    parser.add_argument("-b", "--batch_size", type=int, default=32)
    parser.add_argument("-d", "--debug", action="store_true")
    parser.add_argument("-w", "--num_workers", type=int, default=2)
    parser.add_argument("-l", "--lr", type=float, default=1e-3)
    parser.add_argument("-p", "--stopping_patience", type=int, default=5)
    parser.add_argument("-r", "--root_dir", type=str, default="/workspace")
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()

    train(
        task=args.task,
        model=args.model,
        dataset_name=args.dataset_name,
        root_dir=args.root_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        debug=args.debug,
        num_workers=args.num_workers,
        lr=args.lr,
        stopping_patience=args.stopping_patience,
        download=args.download,
    )

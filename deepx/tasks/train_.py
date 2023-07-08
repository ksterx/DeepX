import pathlib

import torch
from lightning import Trainer
from lightning.pytorch.callbacks import EarlyStopping, ModelSummary
from lightning.pytorch.loggers import MLFlowLogger
from torch import nn

from deepx.tasks import (
    Classification,
    ClassificationDM,
    LangModel,
    Segmentation,
    SegmentationDM,
    WikiText103DM,
)


def train(
    task: str,
    model: str | nn.Module,
    dataset_name: str,
    root_dir: str | pathlib.Path,
    epochs: int = 2,
    batch_size: int = 32,
    debug: bool = False,
    benchmark: bool = False,
    num_workers: int = 2,
    lr: float = 1e-3,
    loss_fn: nn.Module | str = nn.CrossEntropyLoss(),
    stopping_patience: int = 5,
    max_depth: int = 1,
    ckpt_path: str = "",
    download: bool = False,
) -> None:
    root_dir = pathlib.Path(root_dir)
    data_dir = root_dir / "data"
    if isinstance(model, str):
        model_name = model
    else:
        model_name = model.__class__.__name__

    print(f"Root directory: {root_dir.resolve()}")

    task_kwargs = {
        "model": model,
        "dataset_name": dataset_name,
        "lr": lr,
        "loss_fn": loss_fn,
        "task": task,
        "root_dir": root_dir,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": lr,
        "ckpt_path": ckpt_path,
    }

    dataset_kwargs = {
        "dataset_name": dataset_name,
        "data_dir": data_dir,
        "batch_size": batch_size,
        "num_workers": num_workers,
        "download": download,
    }

    # Task and dataset
    if task == "cls" or task == "classification":
        model = Classification(**task_kwargs)  # type: ignore
        datamodule = ClassificationDM(**dataset_kwargs)  # type: ignore
    elif task == "seg" or task == "segmentation":
        model = Segmentation(**task_kwargs)  # type: ignore
        datamodule = SegmentationDM(**dataset_kwargs)  # type: ignore
    elif task == "lm" or task == "langmodel":
        model = LangModel(**task_kwargs)  # type: ignore
        datamodule = WikiText103DM(**dataset_kwargs)  # type: ignore
    else:
        raise ValueError(f"Task {task} is not found in available tasks.")

    # Logging directory
    if debug:
        log_dir = root_dir / "experiments/tests"
    else:
        log_dir = root_dir / "experiments"

    mlf_logger = MLFlowLogger(
        experiment_name=dataset_name,
        tags={"model": model_name},
        tracking_uri=f"file://{log_dir}",
    )
    mlf_logger.log_hyperparams(task_kwargs)

    trainer = Trainer(
        max_epochs=epochs,
        accelerator="auto",
        devices=1,
        logger=mlf_logger,
        enable_checkpointing=True,
        callbacks=[
            EarlyStopping(monitor="val_loss", patience=stopping_patience),
            ModelSummary(max_depth=max_depth),
        ],
        benchmark=benchmark,
        fast_dev_run=debug,
    )

    torch.compile(model)
    trainer.fit(model, datamodule=datamodule, ckpt_path=ckpt_path)
    if not debug:
        trainer.test(ckpt_path="best", datamodule=datamodule)


if __name__ == "__main__":
    import argparse

    from deepx.nn import registered_models
    from deepx.tasks import registered_datasets

    # Argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--task", type=str, default="classification", required=True)
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="resnet50",
        choices=registered_models.keys(),
        required=True,
    )
    parser.add_argument(
        "-ds",
        "--dataset_name",
        type=str,
        default="mnist",
        choices=registered_datasets.keys(),
        required=True,
    )
    parser.add_argument("-e", "--epochs", type=int, default=2)
    parser.add_argument("-b", "--batch_size", type=int, default=32)
    parser.add_argument("-d", "--debug", action="store_true")
    parser.add_argument("-w", "--num_workers", type=int, default=1)
    parser.add_argument("-l", "--lr", type=float, default=1e-3)
    parser.add_argument("-lf", "--loss_fn", type=str, default="cross_entropy")
    parser.add_argument("-p", "--stopping_patience", type=int, default=5)
    parser.add_argument("-r", "--root_dir", type=str, default="/workspace")
    parser.add_argument("-bm", "--benchmark", action="store_true")
    parser.add_argument("--ckpt_path", default=None)
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()

    train(
        task=args.task,
        model=args.model,
        dataset_name=args.dataset_name,
        root_dir=args.root_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        benchmark=args.benchmark,
        debug=args.debug,
        num_workers=args.num_workers,
        lr=args.lr,
        stopping_patience=args.stopping_patience,
        ckpt_path=args.ckpt_path,
        download=args.download,
    )

from deepx.tasks.core import DataModuleX, TaskX
from deepx.tasks.language import LangModel, WikiText103DM
from deepx.tasks.trainers.trainer import TrainerX
from deepx.tasks.vision import (
    CIFAR10DM,
    CIFAR100DM,
    MNISTDM,
    Classification,
    ClassificationDM,
    FashionMNISTDM,
    Flowers102DM,
    Segmentation,
    SegmentationDM,
    VOCSegDM,
)

__all__ = [
    "DataModuleX",
    "TaskX",
    "ClassificationDM",
    "Classification",
    "SegmentationDM",
    "Segmentation",
    "LangModel",
    "WikiText103DM",
    "TrainerX",
]

registered_tasks = {
    "classification": {
        "task": Classification,
        "datamodule": {
            "mnist": MNISTDM,
            "fashionmnist": FashionMNISTDM,
            "cifar10": CIFAR10DM,
            "cifar100": CIFAR100DM,
            "flowers102": Flowers102DM,
        },
    },
    "segmentation": {
        "task": Segmentation,
        "datamodule": {
            "vocseg": VOCSegDM,
        },
    },
    "langmodel": {
        "task": LangModel,
        "datamodule": [
            WikiText103DM,
            # WikiText2Dataset,
        ],
    },
    # "translation": {
    #     "task": TranslationTask,
    #     "dataset": [
    #         Multi30kDataset,
    #         IWSLTDataset,
    #     ],
    # },
}

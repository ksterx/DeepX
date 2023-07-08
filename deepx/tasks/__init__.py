from .core import DataModuleX, TaskX
from .language import LangModel, PennTreebankDM, WikiText103DM
from .trainers.trainer import TrainerX
from .vision import (
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
        "datamodule": {
            "wiki103": WikiText103DM,
            "penn": PennTreebankDM,
            # WikiText2Dataset,
        },
    },
    # "translation": {
    #     "task": TranslationTask,
    #     "dataset": [
    #         Multi30kDataset,
    #         IWSLTDataset,
    #     ],
    # },
}

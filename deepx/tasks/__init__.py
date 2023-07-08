from torchvision import transforms
from torchvision.datasets import VOCSegmentation

from deepx.tasks.base import DataModuleX, TaskX
from deepx.tasks.language import LangModeling, WikiText103DM
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
)

from .trainers.trainer import TrainerX

__all__ = [
    "DataModuleX",
    "TaskX",
    "ClassificationDM",
    "Classification",
    "SegmentationDM",
    "Segmentation",
    "LangModeling",
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
        "datamodule": [SegmentationDM],
    },
    "langmodel": {
        "task": LangModeling,
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


registered_datasets = {
    "voc": {
        "class": VOCSegmentation,
        "num_classes": 21,
        "num_channels": 3,
        "transform": transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize(
                    (224, 224), interpolation=transforms.InterpolationMode.NEAREST_EXACT
                ),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        ),
        "target_transform": transforms.Compose(
            [
                transforms.PILToTensor(),
                transforms.Resize(
                    (224, 224), interpolation=transforms.InterpolationMode.NEAREST_EXACT
                ),
            ]
        ),
    },
}

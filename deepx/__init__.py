from .dms import (
    CIFAR10DM,
    CIFAR100DM,
    MNISTDM,
    FashionMNISTDM,
    Flowers102DM,
    PennTreebankDM,
    VOCSegDM,
    WikiText103DM,
)
from .tasks import Classification, LangModel, Segmentation

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

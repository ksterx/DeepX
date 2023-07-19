from .algos import Classification, ImageGeneration, LangModel, Segmentation
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

registered_algos = {
    "classification": {
        "algo": Classification,
        "datamodule": {
            "mnist": MNISTDM,
            "fashionmnist": FashionMNISTDM,
            "cifar10": CIFAR10DM,
            "cifar100": CIFAR100DM,
            "flowers102": Flowers102DM,
        },
    },
    "segmentation": {
        "algo": Segmentation,
        "datamodule": {
            "vocseg": VOCSegDM,
        },
    },
    "imagegeneration": {
        "algo": ImageGeneration,
        "datamodule": {
            "mnist": MNISTDM,
            "fashionmnist": FashionMNISTDM,
            "cifar10": CIFAR10DM,
            "cifar100": CIFAR100DM,
            "flowers102": Flowers102DM,
        },
    },
    "langmodel": {
        "algo": LangModel,
        "datamodule": {
            "wiki103": WikiText103DM,
            "penn": PennTreebankDM,
            # WikiText2Dataset,
        },
    },
    # "translation": {
    #     "algo": Translation,
    #     "dataset": [
    #         Multi30kDataset,
    #         IWSLTDataset,
    #     ],
    # },
}

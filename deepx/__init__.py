from .dms import (
    CIFAR10DM,
    CIFAR100DM,
    MNISTDM,
    AnimeDM,
    CelebADM,
    FashionMNISTDM,
    Flowers102DM,
    LFWPeopleDM,
    PennTreebankDM,
    VOCSegDM,
    WikiText103DM,
)
from .tasks import GAN, Classification, LangModel, Segmentation

registered_tasks = {
    "classification": {
        "name": Classification,
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
    "gan": {
        "name": GAN,
        "datamodule": {
            "mnist": MNISTDM,
            "fashionmnist": FashionMNISTDM,
            "cifar10": CIFAR10DM,
            "cifar100": CIFAR100DM,
            "flowers102": Flowers102DM,
            "celeba": CelebADM,
            "lfw": LFWPeopleDM,
            "anime": AnimeDM,
        },
    },
    "langmodel": {
        "name": LangModel,
        "datamodule": {
            "wiki103": WikiText103DM,
            "penn": PennTreebankDM,
            # WikiText2Dataset,
        },
    },
    # "translation": {
    #     "name": Translation,
    #     "dataset": [
    #         Multi30kDataset,
    #         IWSLTDataset,
    #     ],
    # },
}

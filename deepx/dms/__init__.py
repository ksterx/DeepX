from .classification import ClassificationDM
from .data import (
    CIFAR10DM,
    CIFAR100DM,
    KFTTDM,
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
from .dm import DataModule
from .langmodel import LangModelDM
from .segmentation import SegmentationDM
from .translation import TranslationDM

__all__ = [
    "DataModule",
    "ClassificationDM",
    "LangModelDM",
    "SegmentationDM",
    "TranslationDM",
]

dm_aliases = {
    "mnist": MNISTDM,
    "fashionmnist": FashionMNISTDM,
    "cifar10": CIFAR10DM,
    "cifar100": CIFAR100DM,
    "flowers102": Flowers102DM,
    "celeba": CelebADM,
    "lfw": LFWPeopleDM,
    "anime": AnimeDM,
    "vocseg": VOCSegDM,
    "wiki103": WikiText103DM,
    "penn": PennTreebankDM,
    "kftt": KFTTDM,
}

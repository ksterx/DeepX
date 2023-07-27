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
    "AnimeDM",
    "DataModule",
    "ClassificationDM",
    "LangModelDM",
    "SegmentationDM",
    "TranslationDM",
    "CIFAR10DM",
    "CIFAR100DM",
    "KFTTDM",
    "MNISTDM",
    "FashionMNISTDM",
    "Flowers102DM",
    "PennTreebankDM",
    "VOCSegDM",
    "WikiText103DM",
    "CelebADM",
    "LFWPeopleDM",
]

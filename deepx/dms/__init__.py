from .classification import ClassificationDM
from .data import (
    CIFAR10DM,
    CIFAR100DM,
    KFTTDM,
    MNISTDM,
    FashionMNISTDM,
    Flowers102DM,
    PennTreebankDM,
    VOCSegDM,
    WikiText103DM,
)
from .dm import DataModuleX
from .langmodel import LangModelDM
from .segmentation import SegmentationDM
from .translation import TranslationDM

__all__ = [
    "DataModuleX",
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
]

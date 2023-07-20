from .celeba import CelebADM
from .cifar10 import CIFAR10DM
from .cifar100 import CIFAR100DM
from .fashionmnist import FashionMNISTDM
from .flowers102 import Flowers102DM
from .kftt import KFTTDM
from .lfw import LFWPeopleDM
from .mnist import MNISTDM
from .penntreebank import PennTreebankDM
from .vocseg import VOCSegDM
from .wikitext103 import WikiText103DM

__all__ = [
    "CelebADM",
    "CIFAR10DM",
    "CIFAR100DM",
    "KFTTDM",
    "LFWPeopleDM",
    "MNISTDM",
    "FashionMNISTDM",
    "Flowers102DM",
    "PennTreebankDM",
    "VOCSegDM",
    "WikiText103DM",
]

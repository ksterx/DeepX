from .classification import Classification
from .core import Task
from .gan import GAN
from .langmodel import LangModel
from .segmentation import Segmentation
from .translation import Translation

__all__ = [
    "Task",
    "Classification",
    "LangModel",
    "Segmentation",
    "Translation",
    "GAN",
]

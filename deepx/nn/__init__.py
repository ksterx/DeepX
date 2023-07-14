from torch.nn import BCELoss, CrossEntropyLoss, MSELoss

from .gan import GAN
from .losses import DiceLoss, FocalLoss
from .mlp import MLP
from .resnet import ResNet, ResNet18, ResNet34, ResNet50, ResNet101, ResNet152
from .transformer import LangModelTransformer
from .unet import UNet

registered_models = {
    "resnet": ResNet,
    "resnet18": ResNet18,
    "resnet34": ResNet34,
    "resnet50": ResNet50,
    "resnet101": ResNet101,
    "resnet152": ResNet152,
    "unet": UNet,
    "mlp": MLP,
    "lmtransformer": LangModelTransformer,
    "gan": GAN,
}

registered_losses = {
    "dice": DiceLoss,
    "focal": FocalLoss,
    "ce": CrossEntropyLoss,
    "bce": BCELoss,
    "mse": MSELoss,
}


def register_model(name: str, model):
    registered_models[name] = model

from torchvision import transforms

from deepx.nn.core import MLP
from deepx.nn.losses import DiceLoss, FocalLoss
from deepx.nn.resnet import ResNet, ResNet18, ResNet34, ResNet50, ResNet101, ResNet152
from deepx.nn.transformer import (
    ClassificationTransformer,
    LangModelTransformer,
    Transformer,
)
from deepx.nn.unet import UNet

registered_models = {
    "resnet": ResNet,
    "resnet18": ResNet18,
    "resnet34": ResNet34,
    "resnet50": ResNet50,
    "resnet101": ResNet101,
    "resnet152": ResNet152,
    "unet": UNet,
    "mlp": MLP,
    "lang": LangModelTransformer,
}

registered_losses = {
    "dice": DiceLoss,
    "focal": FocalLoss,
}


def register_model(name: str, model):
    registered_models[name] = model

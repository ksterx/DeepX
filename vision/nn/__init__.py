from .core import MLP
from .resnet import ResNet, ResNet18, ResNet34, ResNet50, ResNet101, ResNet152
from .unet import UNet

available_models = {
    "resnet18": ResNet18,
    "resnet34": ResNet34,
    "resnet50": ResNet50,
    "resnet101": ResNet101,
    "resnet152": ResNet152,
    "unet": UNet,
}

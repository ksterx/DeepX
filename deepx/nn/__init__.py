from torch.nn import BCELoss, BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from .dcgan import DCGAN, Discriminator, DiscriminatorBlock, Generator, GeneratorBlock
from .losses import DiceLoss, FocalLoss
from .mlp import MLP
from .resnet import ResNet, ResNet18, ResNet34, ResNet50, ResNet101, ResNet152
from .transformer import LangModelTransformer
from .unet import UNet
from .vae import VAE

backbone_aliases = {
    "resnet": ResNet,
    "resnet18": ResNet18,
    "resnet34": ResNet34,
    "resnet50": ResNet50,
    "resnet101": ResNet101,
    "resnet152": ResNet152,
    "mlp": MLP,
}

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
    "dcgan": DCGAN,
    "vae": VAE,
}

registered_losses = {
    "dice": DiceLoss,
    "focal": FocalLoss,
    "ce": CrossEntropyLoss,
    "bce": BCELoss,
    "bce_logits": BCEWithLogitsLoss,
    "mse": MSELoss,
}


def register_model(name: str, model):
    registered_models[name] = model

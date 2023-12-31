import torch
from torch import nn

from ..utils.wrappers import watch_kwargs
from .dcgan import Generator
from .resnet import ResNet18


class Encoder(nn.Module):
    def __init__(
        self,
        backbone: str,
        in_channels: int,
        latent_dim: int,
        dropout: float,
    ):
        super().__init__()

        self.backbone = ResNet18(
            in_channels=in_channels, num_classes=1, dropout=dropout, has_head=False
        )
        self.head_mu = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(
                self.backbone.block.CHANNELS_IN_LAYERS[-1],
                latent_dim,
            ),
        )
        self.head_logvar = nn.Sequential(*self.head_mu)

    def forward(self, x):
        x = self.backbone(x)
        mu = self.head_mu(x)
        logvar = self.head_logvar(x)
        eps = torch.randn_like(mu)
        z = mu + torch.exp(0.5 * logvar) * eps
        return z, mu, logvar


class Decoder(Generator):
    pass


class VAE(nn.Module):
    NAME = "vae"

    @watch_kwargs
    def __init__(
        self,
        backbone: str,
        latent_dim: int,
        base_dim_g: int,
        tgt_shape: tuple[int, int, int],
        negative_slope: float = 0.01,
        dropout: float = 0.0,
        **kwargs,
    ) -> None:
        """Variational Autoencoder

        Args:
            tgt_shape (tuple[int, int, int]): (C, H, W)
        """
        super().__init__()

        self.encoder = Encoder(
            backbone=backbone,
            in_channels=tgt_shape[0],
            latent_dim=latent_dim,
            dropout=dropout,
        )
        self.decoder = Decoder(
            tgt_shape=tgt_shape,
            latent_dim=latent_dim,
            base_channels=base_dim_g,
            negative_slope=negative_slope,
            dropout=dropout,
        )

    def forward(self, x):
        z, mu, logvar = self.encoder(x)
        z = z.view(z.size(0), z.size(1), 1, 1)
        x = self.decoder(z)
        return x, z, mu, logvar

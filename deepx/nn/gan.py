import math

from torch import nn

import deepx


class GAN(nn.Module):
    NAME = "gan"

    def __init__(
        self,
        backbone: str,
        tgt_shape: tuple[int, int, int],
        hidden_dim: int = 1024,
        negative_slope: float = 0.01,
        dropout: float = 0.0,
        latent_dim: int = 100,
        base_channels: int = 32,
    ) -> None:
        """Generative Adversarial Network

        Args:
            backbone (str): Backbone architecture for discriminator
            tgt_shape (tuple[int, int, int]): (C, H, W)
        """
        super().__init__()

        self.generator = Generator(
            tgt_shape=tgt_shape,
            latent_dim=latent_dim,
            base_channels=base_channels,
            dropout=dropout,
        )
        self.discriminator = Discriminator(
            backbone=backbone,
            tgt_shape=tgt_shape,
            hidden_dim=hidden_dim,
            negative_slope=negative_slope,
            dropout=dropout,
        )

    def forward(self, x):
        return self.generator(x)


class Discriminator(nn.Module):
    def __init__(self, backbone, tgt_shape, hidden_dim=1024, negative_slope=0.01, dropout=0.0):
        super().__init__()
        c, h, w = tgt_shape

        if "resnet" in backbone:
            self.backbone = deepx.nn.registered_models[backbone](
                num_classes=0, in_channels=c, dropout=dropout, has_head=False
            )
            self.head = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(
                    self.backbone.block.CHANNELS_IN_LAYERS[-1] * self.backbone.block.EXPANSION,
                    hidden_dim,
                ),
                nn.LeakyReLU(negative_slope=negative_slope),
                nn.Linear(hidden_dim, 1),
                nn.Sigmoid(),
            )
        elif backbone == "mlp":
            input_dim = c * h * w
            self.backbone = deepx.nn.registered_models[backbone](
                channels_in_layers=[input_dim, 512],
                activation=nn.LeakyReLU(negative_slope=negative_slope),
                output_activation=nn.LeakyReLU(negative_slope=negative_slope),
                dropout=dropout,
                flatten=True,
            )
            self.head = nn.Sequential(nn.Linear(512, 1), nn.Sigmoid())

    def forward(self, x):
        x = self.backbone(x)
        x = self.head(x)
        return x


class Generator(nn.Module):
    def __init__(
        self, tgt_shape, latent_dim=100, base_channels=32, negative_slope=0.01, dropout=0.0
    ):
        super().__init__()

        self.tgt_shape = tgt_shape
        self.latent_dim = latent_dim
        self.base_channels = base_channels

        c, h, w = tgt_shape
        assert h == w, "Height and width must be equal"
        power = int(math.log2(h))
        channels = [base_channels * i for i in range(power - 2, 0, -1)]

        self.model = nn.ModuleList(
            [
                nn.ConvTranspose2d(latent_dim, base_channels * (power - 2), 4, 1, 0),
                nn.Dropout2d(dropout),
                nn.BatchNorm2d(base_channels * (power - 2)),
                nn.LeakyReLU(negative_slope=negative_slope, inplace=True),
            ]
        )

        for i in range(power - 3):
            self.model.append(
                nn.Sequential(
                    nn.ConvTranspose2d(channels[i], channels[i + 1], 4, 2, 1),
                    nn.Dropout2d(dropout),
                    nn.BatchNorm2d(channels[i + 1]),
                    nn.LeakyReLU(negative_slope=negative_slope, inplace=True),
                )
            )

        self.model.append(
            nn.Sequential(
                nn.ConvTranspose2d(channels[-1], c, 4, 2, 1),
                nn.Tanh(),
            )
        )

    def forward(self, x):
        for layer in self.model:
            x = layer(x)
        return x

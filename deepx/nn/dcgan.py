import math

from torch import nn


class DCGAN(nn.Module):
    NAME = "dcgan"

    def __init__(
        self,
        tgt_shape: tuple[int, int, int],
        hidden_dim: int = 1024,
        negative_slope: float = 0.01,
        dropout: float = 0.0,
        latent_dim: int = 1024,
        base_channels: int = 128,
        **kwargs,
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
            tgt_shape=tgt_shape,
            hidden_dim=hidden_dim,
            negative_slope=negative_slope,
            dropout=dropout,
        )

    def forward(self, x):
        return self.generator(x)


class Discriminator(nn.Module):
    def __init__(self, tgt_shape, hidden_dim, negative_slope, dropout):
        super().__init__()

        c, h, _ = tgt_shape
        num_layers = int(math.log2(h))

        self.model = nn.Sequential(
            DiscriminatorBlock(
                in_channels=c,
                out_channels=hidden_dim,
                kernel_size=4,
                stride=2,
                padding=1,
                negative_slope=negative_slope,
                dropout=dropout,
            ),
            *[
                DiscriminatorBlock(
                    in_channels=hidden_dim * 2 ** i,
                    out_channels=hidden_dim * 2 ** (i + 1),
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    negative_slope=negative_slope,
                    dropout=dropout,
                )
                for i in range(num_layers - 2)
            ],
            nn.Conv2d(hidden_dim * 2 ** (num_layers - 2), 1, 4, 1, 0),
            nn.Sigmoid(),
        )



class DiscriminatorBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        negative_slope,
        dropout,
    ):
        super().__init__()

        self.model = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=False,
            ),
            nn.Dropout2d(dropout),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(negative_slope=negative_slope, inplace=True),
        )

    def forward(self, x):
        return self.model(x)


class Generator(nn.Module):
    def __init__(
        self,
        tgt_shape,
        latent_dim=1024,
        base_channels=32,
        negative_slope=0.01,
        dropout=0.0,
    ):
        super().__init__()

        self.tgt_shape = tgt_shape
        self.latent_dim = latent_dim
        self.base_channels = base_channels

        c, h, w = tgt_shape
        assert h == w, "Height and width must be equal"
        power = int(math.log2(h))
        channels = [base_channels * 2 ** i for i in range(power - 3, -1, -1)]  # [..., base * 2^2, base * 2^1, base * 2^0]

        self.model = nn.ModuleList(
            [
                nn.ConvTranspose2d(latent_dim, channels[0], 4, 1, 0),
                nn.Dropout2d(dropout),
                nn.BatchNorm2d(channels[0]),
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

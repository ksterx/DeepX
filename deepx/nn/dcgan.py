import math

from torch import nn


class DCGAN(nn.Module):
    NAME = "dcgan"

    def __init__(
        self,
        tgt_shape: tuple[int, int, int],
        negative_slope: float = 0.01,
        dropout: float = 0.0,
        latent_dim: int = 100,
        base_dim_g: int = 128,
        base_dim_d: int = 128,
        **kwargs,
    ) -> None:
        """Generative Adversarial Network

        Args:
            tgt_shape (tuple[int, int, int]): (C, H, W)
        """
        super().__init__()

        self.generator = Generator(
            tgt_shape=tgt_shape,
            latent_dim=latent_dim,
            base_channels=base_dim_g,
            dropout=dropout,
        )
        self.discriminator = Discriminator(
            tgt_shape=tgt_shape,
            base_channels=base_dim_d,
            negative_slope=negative_slope,
            dropout=dropout,
        )

    def forward(self, x):
        return self.generator(x)


class Generator(nn.Module):
    def __init__(
        self,
        tgt_shape,
        latent_dim,
        base_channels,
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
        channels = [
            base_channels * 2**i for i in range(power - 3, -1, -1)
        ]  # [..., base * 2^2, base * 2^1, base * 2^0]

        self.model = nn.Sequential(
            GeneratorBlock(
                in_channels=latent_dim,
                out_channels=channels[0],
                kernel_size=4,
                stride=1,
                padding=0,
                negative_slope=negative_slope,
                dropout=dropout,
            ),
            *[
                GeneratorBlock(
                    in_channels=channels[i],
                    out_channels=channels[i + 1],
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    negative_slope=negative_slope,
                    dropout=dropout,
                )
                for i in range(power - 3)
            ],
            nn.ConvTranspose2d(channels[-1], c, 4, 2, 1),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.model(x)


class Discriminator(nn.Module):
    def __init__(self, tgt_shape, base_channels, negative_slope, dropout):
        super().__init__()

        c, h, _ = tgt_shape
        num_layers = int(math.log2(h)) - 1

        self.model = nn.Sequential(
            DiscriminatorBlock(
                in_channels=c,
                out_channels=base_channels,
                kernel_size=4,
                stride=2,
                padding=1,
                negative_slope=negative_slope,
                dropout=dropout,
            ),
            *[
                DiscriminatorBlock(
                    in_channels=base_channels * 2**i,
                    out_channels=base_channels * 2 ** (i + 1),
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    negative_slope=negative_slope,
                    dropout=dropout,
                )
                for i in range(num_layers - 2)
            ],
            nn.Conv2d(base_channels * 2 ** (num_layers - 2), 1, 4, 2, 0),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.model(x)


class GeneratorBlock(nn.Module):
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
            nn.ConvTranspose2d(
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
        out = self.model(x)
        print(out.shape)
        return self.model(x)

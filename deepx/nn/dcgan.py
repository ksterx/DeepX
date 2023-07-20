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
            negative_slope=negative_slope,
            dropout=dropout,
        )
        self.discriminator = Discriminator(
            tgt_shape=tgt_shape,
            base_channels=base_dim_d,
            negative_slope=negative_slope,
            dropout=dropout,
        )

        self.apply(self.init_weights)

    def forward(self, x):
        return self.generator(x)

    @staticmethod
    def init_weights(models):
        for m in models.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.normal_(m.weight, 0.0, 0.02)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.normal_(m.weight, 1.0, 0.02)
                nn.init.constant_(m.bias, 0)


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
        ksp_list = determine_ksp((h, w))
        ksp_list.reverse()
        channels = [base_channels * 2**i for i in range(len(ksp_list) - 1)]
        channels.append(latent_dim)
        channels.reverse()  # [latent_dim, ..., base * 2^1, base * 2^0]

        self.model = nn.Sequential(
            *[
                GeneratorBlock(
                    in_channels=channels[i],
                    out_channels=channels[i + 1],
                    kernel_size=k,
                    stride=s,
                    padding=p,
                    negative_slope=negative_slope,
                    dropout=dropout,
                )
                for i, (k, s, p) in enumerate(ksp_list[:-1])
            ],
            nn.ConvTranspose2d(channels[-1], c, *ksp_list[-1]),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.model(x)


class Discriminator(nn.Module):
    def __init__(self, tgt_shape, base_channels, negative_slope=0.01, dropout=0.0):
        super().__init__()

        c, h, w = tgt_shape
        ksp_list = determine_ksp((h, w))
        channels = [base_channels * 2**i for i in range(len(ksp_list) - 1)]
        channels.append(1)
        channels.insert(0, c)  # [c, base * 2^0, base * 2^1, ..., 1]

        self.model = nn.Sequential(
            *[
                DiscriminatorBlock(
                    in_channels=channels[i],
                    out_channels=channels[i + 1],
                    kernel_size=k,
                    stride=s,
                    padding=p,
                    negative_slope=negative_slope,
                    dropout=dropout,
                )
                for i, (k, s, p) in enumerate(ksp_list[:-1])
            ],
            nn.Conv2d(channels[-2], channels[-1], *ksp_list[-1]),
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
        return self.model(x)


def determine_ksp(shape):
    """Determine kernel_size, stride, and padding for Conv2d and ConvTranspose2d

    Args:
        shape (tuple[int, int]): (H, W)

    Returns:
        tuple[int, int, int]: (kernel_size, stride, padding)
    """
    h, w = shape
    assert h == w, "Height and width must be equal"

    ksp_list = []

    def determine(x):
        if x % 2 == 0 and x > 4:
            ksp_list.append((4, 2, 1))
            determine(x // 2)
        elif x % 2 == 0 and x == 4:
            ksp_list.append((4, 1, 0))
        elif x % 2 == 1 and x > 3:
            ksp_list.append((3, 2, 0))
            determine(x // 2)
        elif x % 2 == 1 and x == 3:
            ksp_list.append((3, 1, 0))
        else:
            raise ValueError(f"Invalid height: {x}")

    determine(h)

    return ksp_list

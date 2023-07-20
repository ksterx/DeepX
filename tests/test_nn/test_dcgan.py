import torch

from deepx.nn import Discriminator, Generator


def test_dcgan():
    N, in_channels, H, W = 4, 3, 64, 64
    latent_dim = 100
    x = torch.randn(N, in_channels, H, W)
    noise = torch.randn(N, latent_dim, 1, 1)

    gen = Generator((in_channels, H, W), latent_dim, 32, 0.01, 0.1)
    out = gen(noise)
    assert out.shape == (N, in_channels, H, W)

    disc = Discriminator((in_channels, H, W), 32, 0.01, 0.1)
    out = disc(x)
    assert out.shape == (N, 1, 1, 1)

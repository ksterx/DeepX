import pytest
import torch

from deepx.utils.vision import denormalize


@pytest.mark.parametrize(
    "levels, dtype, N, C, H, W",
    [
        (256, torch.uint8, 1, 3, 32, 32),
        (256, torch.uint8, 4, 3, 32, 32),
        (2, torch.float, 1, 3, 32, 32),
        (2, torch.float, 4, 3, 32, 32),
    ],
)
def test_denormalize(levels, dtype, N, C, H, W):
    mean = (0.5, 0.5, 0.5)
    std = (0.5, 0.5, 0.5)
    if N == 1:
        img = torch.randn(C, H, W)
        denorm_img = denormalize(
            img, mean, std, is_batch=False, levels=levels, dtype=dtype
        )
    else:
        img = torch.randn(N, C, H, W)
        denorm_img = denormalize(
            img, mean, std, is_batch=True, levels=levels, dtype=dtype
        )
    assert denorm_img.shape == (N, C, H, W) if N > 1 else (C, H, W)
    assert denorm_img.min() >= 0
    assert denorm_img.max() <= 255

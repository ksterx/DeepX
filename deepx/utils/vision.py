import torch
from torch import Tensor


def inverse_transform(
    img: Tensor, mean: tuple, std: tuple, is_batch: bool, levels: int = 256
):
    """Inverse transform for images

    Args:
        img (Tensor): Image tensor. [batch_size, channels, height, width]
        mean (tuple): Mean of the dataset
        std (tuple): Standard deviation of the dataset
        is_batch (bool): If True, img is [batch_size, channels, height, width]
        levels (int, optional): Number of levels. Defaults to 256.

    Returns:
        Tensor: [batch_size, channels, height, width]
    """
    if is_batch:
        img = img.clone().detach()
        img = img * torch.tensor(std, device=img.device).view(1, -1, 1, 1)
        img = img + torch.tensor(mean, device=img.device).view(1, -1, 1, 1)
        img = img * levels
        img = img.clamp(0, levels - 1).to(torch.uint8)
    else:
        img = img.clone().detach()
        img = img * torch.tensor(std, device=img.device).view(-1, 1, 1)
        img = img + torch.tensor(mean, device=img.device).view(-1, 1, 1)
        img = img * levels
        img = img.clamp(0, levels - 1).to(torch.uint8)
    return img

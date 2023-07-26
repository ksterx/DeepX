import torch
from PIL import Image, ImageDraw
from torch import Tensor


def denormalize(
    img: Tensor,
    mean: tuple[int, int, int],
    std: tuple[int, int, int],
    is_batch: bool,
    levels: int = 256,
    dtype=torch.uint8,
) -> Tensor:
    """Inverse transform for images

    Args:
        img (Tensor): Image tensor. [batch_size, channels, height, width]
        mean (tuple): Mean of the dataset
        std (tuple): Standard deviation of the dataset
        is_batch (bool): If True, img is [batch_size, channels, height, width]
        levels (int, optional): Number of levels. Defaults to 256.
        dtype ([type], optional): Data type of the image. Defaults to torch.uint8.

    Returns:
        Tensor: [batch_size, channels, height, width]
    """
    if is_batch:
        img = img.clone().detach()
        img = img * torch.tensor(std, device=img.device).view(1, -1, 1, 1)
        img = img + torch.tensor(mean, device=img.device).view(1, -1, 1, 1)
        img = img * (levels - 1)
        img = img.clamp(0, levels - 1).to(dtype)
    else:
        img = img.clone().detach()
        img = img * torch.tensor(std, device=img.device).view(-1, 1, 1)
        img = img + torch.tensor(mean, device=img.device).view(-1, 1, 1)
        img = img * (levels - 1)
        img = img.clamp(0, levels - 1).to(dtype)
    return img


def make_gif_from_images(
    img_paths: list[str],
    save_path: str,
    metric: str | None = None,
    duration: int = 100,
    loop: int = 0,
) -> None:
    """Create gif from images

    Args:
        img_paths (list[str]): List of image paths
        save_path (str): Path to save the gif
        duration (int, optional): Duration of each frame in ms. Defaults to 100.
        loop (int, optional): Number of loops. Defaults to 0.
    """
    frames = []
    for i, img_path in enumerate(img_paths):
        img = Image.open(img_path)
        if metric is not None:
            draw = ImageDraw.Draw(img)
            draw.multiline_text(
                (0, 0),
                f"{metric}: {i+1:03d}",
                (255, 255, 255),
                stroke_width=5,
                stroke_fill=(0, 0, 0),
            )
        frames.append(img)

    frames[0].save(
        save_path,
        save_all=True,
        append_images=frames[1:],
        optimize=False,
        duration=duration,
        loop=loop,
    )

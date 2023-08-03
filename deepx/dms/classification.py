from torchvision import transforms

from .dm import DataModule


class ClassificationDM(DataModule):
    NAME: str
    NUM_CLASSES: int | None
    NUM_CHANNELS: int
    SIZE: tuple[int, int]

    def __init__(
        self,
        data_dir: str,
        batch_size: int,
        num_workers: int,
        **kwargs,
    ):
        super().__init__(
            data_dir=data_dir,
            batch_size=batch_size,
            num_workers=num_workers,
            **kwargs,
        )

    @classmethod
    def _transform(cls, size, mean, std, antialias=True):
        return transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize(
                    size,
                    antialias=antialias,
                    interpolation=transforms.InterpolationMode.BICUBIC,
                ),
                transforms.Normalize(mean, std),
            ]
        )

    @classmethod
    def _train_transform(cls, size, mean, std, antialias=True):
        return transforms.Compose(
            [
                transforms.RandomCrop(size, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Resize(
                    size,
                    antialias=antialias,
                    interpolation=transforms.InterpolationMode.BICUBIC,
                ),
                transforms.Normalize(mean, std),
            ]
        )

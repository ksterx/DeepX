from torchvision import transforms

from .dm import DataModuleX


class ClassificationDM(DataModuleX):
    def __init__(
        self,
        data_dir: str,
        batch_size: int,
        train_ratio: float,
        num_workers: int,
        download: bool = False,
        **kwargs,
    ):
        super().__init__(
            data_dir=data_dir,
            batch_size=batch_size,
            train_ratio=train_ratio,
            num_workers=num_workers,
            download=download,
        )

    def transform(self, size, antialias=True, mean=(0.1307,), std=(0.3081,)):
        return transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize(size, antialias=antialias),
                transforms.Normalize(mean, std),
            ]
        )

    def train_transform(self, size, antialias=True, mean=(0.1307,), std=(0.3081,)):
        return transforms.Compose(
            [
                transforms.RandomCrop(size, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Resize(size, antialias=antialias),
                transforms.Normalize(mean, std),
            ]
        )

from torchvision.datasets import FashionMNIST

from ..classification import ClassificationDM


class FashionMNISTDM(ClassificationDM):
    NAME = "fashionmnist"
    NUM_CLASSES = FashionMNIST.classes
    NUM_CHANNELS = 1
    SIZE = (28, 28)

    def __init__(
        self,
        data_dir: str,
        batch_size: int,
        train_ratio: float,
        num_workers: int,
        download: bool = False,
        mean: tuple[float, ...] = (0.5, 0.5, 0.5),
        std: tuple[float, ...] = (0.5, 0.5, 0.5),
        **kwargs,
    ):
        super().__init__(
            data_dir=data_dir,
            batch_size=batch_size,
            num_workers=num_workers,
            **kwargs,
        )

        self.train_ratio = train_ratio
        self.download = download
        self.mean = mean
        self.std = std

    def prepare_data(self):
        FashionMNIST(self.data_dir, train=True, download=self.download)
        FashionMNIST(self.data_dir, train=False, download=self.download)

    def setup(self, stage=None):
        if stage == "fit":
            data = FashionMNIST(
                self.data_dir,
                train=True,
                transform=self.train_transform(self.mean, self.std),
            )
            self.train_data, self.val_data = self._random_split(data, self.train_ratio)

            self.test_data = FashionMNIST(
                self.data_dir,
                train=False,
                transform=self.transform(self.mean, self.std),
            )

        if stage == "predict":
            self.predict_data = FashionMNIST(
                self.data_dir,
                train=False,
                transform=self.transform(self.mean, self.std),
            )

    @classmethod
    def transform(cls, mean, std):
        return cls._transform(cls.SIZE, mean, std)

    @classmethod
    def train_transform(cls, mean, std):
        return cls._train_transform(cls.SIZE, mean, std)

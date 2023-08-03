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

    def prepare_data(self):
        FashionMNIST(self.data_dir, train=True, download=self.download)
        FashionMNIST(self.data_dir, train=False, download=self.download)

    def setup(self, stage=None):
        if stage == "fit":
            data = FashionMNIST(
                self.data_dir, train=True, transform=self.train_transform()
            )
            self.train_data, self.val_data = self._random_split(data, self.train_ratio)

            self.test_data = FashionMNIST(
                self.data_dir, train=False, transform=self.transform()
            )

        if stage == "predict":
            self.predict_data = FashionMNIST(
                self.data_dir, train=False, transform=self.transform()
            )

    @classmethod
    def transform(cls):
        return cls._transform(cls.SIZE)

    @classmethod
    def train_transform(cls):
        return cls._train_transform(cls.SIZE)

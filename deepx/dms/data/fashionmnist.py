from torchvision.datasets import FashionMNIST

from ..classification import ClassificationDM


class FashionMNISTDM(ClassificationDM):
    NAME = "fashionmnist"
    NUM_CLASSES = FashionMNIST.classes
    NUM_CHANNELS = 1
    SIZE = (32, 32)

    def __init__(
        self,
        data_dir: str,
        batch_size: int = 32,
        train_ratio: float = 0.9,
        num_workers: int = 2,
        download: bool = False,
    ):
        super().__init__(
            data_dir=data_dir,
            batch_size=batch_size,
            train_ratio=train_ratio,
            num_workers=num_workers,
            download=download,
        )

    def prepare_data(self):
        FashionMNIST(self.data_dir, train=True, download=self.download)
        FashionMNIST(self.data_dir, train=False, download=self.download)

    def setup(self, stage=None):
        if stage == "fit":
            data = FashionMNIST(
                self.data_dir, train=True, transform=self.train_transform()
            )
            self.train_data, self.val_data = self._random_split(data)

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

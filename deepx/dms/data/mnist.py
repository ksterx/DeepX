from torchvision.datasets import MNIST

from ..classification import ClassificationDM


class MNISTDM(ClassificationDM):
    NAME = "mnist"
    NUM_CLASSES = 10
    NUM_CHANNELS = 1
    SIZE = (32, 32)

    def __init__(
        self,
        data_dir: str,
        batch_size: int = 32,
        train_ratio: float = 0.9,
        num_workers: int = 2,
        download: bool = False,
        **kwargs,
    ):
        super().__init__(
            data_dir=data_dir,
            batch_size=batch_size,
            train_ratio=train_ratio,
            num_workers=num_workers,
            download=download,
            **kwargs,
        )

    def prepare_data(self):
        MNIST(self.data_dir, train=True, download=self.download)
        MNIST(self.data_dir, train=False, download=self.download)

    def setup(self, stage=None):
        if stage == "fit":
            data = MNIST(self.data_dir, train=True, transform=self.transform())
            self.train_data, self.val_data = self._random_split(data)

            self.test_data = MNIST(
                self.data_dir, train=False, transform=self.transform()
            )

        if stage == "predict":
            self.predict_data = MNIST(
                self.data_dir, train=False, transform=self.transform()
            )

    @classmethod
    def transform(cls):
        return cls._transform(cls.SIZE)

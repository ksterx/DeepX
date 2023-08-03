from torchvision.datasets import MNIST

from ..classification import ClassificationDM


class MNISTDM(ClassificationDM):
    NAME = "mnist"
    NUM_CLASSES = 10
    NUM_CHANNELS = 1
    SIZE = (28, 28)

    def __init__(
        self,
        data_dir: str,
        batch_size: int,
        num_workers: int,
        train_ratio: float,
        download: bool = False,
        mean: tuple[float, ...] = (0.1307,),
        std: tuple[float, ...] = (0.3081,),
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
        MNIST(self.data_dir, train=True, download=self.download)
        MNIST(self.data_dir, train=False, download=self.download)

    def setup(self, stage=None):
        if stage == "fit":
            data = MNIST(
                self.data_dir, train=True, transform=self.transform(self.mean, self.std)
            )
            self.train_data, self.val_data = self._random_split(data, self.train_ratio)

            self.test_data = MNIST(
                self.data_dir,
                train=False,
                transform=self.transform(self.mean, self.std),
            )

        if stage == "predict":
            self.predict_data = MNIST(
                self.data_dir,
                train=False,
                transform=self.transform(self.mean, self.std),
            )

    @classmethod
    def transform(cls, mean, std):
        return cls._transform(cls.SIZE, mean, std)

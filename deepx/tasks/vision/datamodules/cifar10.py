from torch.utils.data import random_split
from torchvision.datasets import CIFAR10

from ..classification import ClassificationDM


class CIFAR10DM(ClassificationDM):
    NUM_CLASSES = 10
    NUM_CHANNELS = 3
    CLASSES = [
        "plane",
        "car",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    ]

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

        self._transform = self.transform((32, 32))
        self._train_transform = self.train_transform((32, 32))

    def prepare_data(self):
        CIFAR10(self.data_dir, train=True, download=self.download)
        CIFAR10(self.data_dir, train=False, download=self.download)

    def setup(self, stage=None):
        if stage == "fit":
            data = CIFAR10(self.data_dir, train=True, transform=self._train_transform)
            num_data = len(data)
            self.train_data, self.val_data = random_split(
                dataset=data,
                lengths=[num_data * self.train_ratio, num_data * (1 - self.train_ratio)],
            )

            self.test_data = CIFAR10(self.data_dir, train=False, transform=self._transform)

        if stage == "predict":
            self.predict_data = CIFAR10(self.data_dir, train=False, transform=self._transform)

    @property
    def name(self):
        return "cifar10"

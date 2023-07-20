from torchvision.datasets import CelebA

from ..classification import ClassificationDM


class CelebADM(ClassificationDM):
    NAME = "celeba"
    NUM_CLASSES = None
    NUM_CHANNELS = 3
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
        CelebA(self.data_dir, split="all", download=self.download)

    def setup(self, stage=None):
        if stage == "fit":
            self.train_data = CelebA(
                self.data_dir, split="train", transform=self.train_transform()
            )
            self.val_data = CelebA(
                self.data_dir, split="valid", transform=self.transform()
            )

            self.test_data = CelebA(
                self.data_dir, split="test", transform=self.transform()
            )
        if stage == "predict":
            self.predict_data = CelebA(
                self.data_dir, split="test", transform=self.transform()
            )

    @classmethod
    def transform(cls):
        return cls._transform(cls.SIZE)

    @classmethod
    def train_transform(cls):
        return cls._train_transform(cls.SIZE)
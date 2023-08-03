from torchvision.datasets import CelebA

from ..classification import ClassificationDM


class CelebADM(ClassificationDM):
    NAME = "celeba"
    NUM_CLASSES = None
    NUM_CHANNELS = 3
    SIZE = (128, 128)

    def __init__(
        self,
        data_dir: str,
        batch_size: int,
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

        self.download = download

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

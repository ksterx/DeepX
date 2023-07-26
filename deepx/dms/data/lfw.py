from torchvision.datasets import LFWPeople

from ..classification import ClassificationDM


class LFWPeopleDM(ClassificationDM):
    NAME = "lfw"
    NUM_CLASSES = None
    NUM_CHANNELS = 3
    SIZE = (128, 128)

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
        )

        self.train_ratio = train_ratio
        self.download = download

    def prepare_data(self):
        LFWPeople(self.data_dir, split="train", download=self.download)
        LFWPeople(self.data_dir, split="test", download=self.download)

    def setup(self, stage=None):
        if stage == "fit":
            data = LFWPeople(self.data_dir, split="train", transform=self.transform())
            self.train_data, self.val_data = self._random_split(data, self.train_ratio)

            self.test_data = LFWPeople(
                self.data_dir, split="test", transform=self.transform()
            )

        if stage == "predict":
            self.predict_data = LFWPeople(
                self.data_dir, split="test", transform=self.transform()
            )

    @classmethod
    def transform(cls):
        return cls._transform(cls.SIZE)

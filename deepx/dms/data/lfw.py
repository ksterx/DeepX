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
        LFWPeople(self.data_dir, split="train", download=self.download)
        LFWPeople(self.data_dir, split="test", download=self.download)

    def setup(self, stage=None):
        if stage == "fit":
            data = LFWPeople(
                self.data_dir,
                split="train",
                transform=self.transform(self.mean, self.std),
            )
            self.train_data, self.val_data = self._random_split(data, self.train_ratio)

            self.test_data = LFWPeople(
                self.data_dir,
                split="test",
                transform=self.transform(self.mean, self.std),
            )

        if stage == "predict":
            self.predict_data = LFWPeople(
                self.data_dir,
                split="test",
                transform=self.transform(self.mean, self.std),
            )

    @classmethod
    def transform(cls, mean, std):
        return cls._transform(cls.SIZE, mean, std)

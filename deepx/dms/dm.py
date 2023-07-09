import pathlib
from abc import ABC, abstractmethod

from lightning import LightningDataModule
from torch.utils.data import DataLoader, random_split


class DataModuleX(LightningDataModule, ABC):
    def __init__(
        self,
        data_dir: str | pathlib.Path,
        batch_size: int = 32,
        train_ratio: float = 0.9,
        num_workers: int = 2,
        download: bool = False,
    ):
        LightningDataModule.__init__(self)
        ABC.__init__(self)

        if not isinstance(data_dir, str):
            data_dir = str(data_dir)

        self.data_dir = data_dir
        self.batch_size = batch_size
        self.train_ratio = train_ratio
        self.num_workers = num_workers
        self.download = download

    @abstractmethod
    def setup(self, stage=None):
        self.train_data = None
        self.val_data = None
        self.test_data = None
        self.predict_data = None

    def train_dataloader(self):
        return DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_data,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_data,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def predict_dataloader(self):
        return DataLoader(
            self.predict_data,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def _random_split(self, data):
        num_data = len(data)
        len_train = int(num_data * self.train_ratio)
        len_val = num_data - len_train
        return random_split(dataset=data, lengths=[len_train, len_val])

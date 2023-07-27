import pathlib
from abc import ABC, abstractmethod

from lightning import LightningDataModule
from torch.utils.data import DataLoader, random_split

from ..utils.wrappers import watch_kwargs


class DataModule(LightningDataModule, ABC):
    @watch_kwargs
    def __init__(
        self,
        data_dir: str | pathlib.Path,
        batch_size: int = 32,
        num_workers: int = 2,
        **kwargs,
    ):
        """Base class for all data modules.

        Args:
            data_dir (str | pathlib.Path): Data directory
            batch_size (int, optional): Batch size. Defaults to 32.
            train_ratio (float, optional): Ratio of training data. Defaults to 0.9.
            num_workers (int, optional): Number of workers. Defaults to 2.
            download (bool, optional): Download data. Defaults to False.
        """

        LightningDataModule.__init__(self)
        ABC.__init__(self)

        if not isinstance(data_dir, str):
            data_dir = str(data_dir)

        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

    @abstractmethod
    def setup(self, stage=None):
        raise NotImplementedError

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

    def _random_split(self, data, train_ratio):
        num_data = len(data)
        len_train = int(num_data * train_ratio)
        len_val = num_data - len_train
        return random_split(dataset=data, lengths=[len_train, len_val])

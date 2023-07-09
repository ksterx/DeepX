import pathlib

from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from .dm import DataModuleX


class LangModelDM(DataModuleX):
    def __init__(
        self,
        tokenizer: str,
        max_length: int,
        data_dir: str | pathlib.Path,
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
        if isinstance(tokenizer, str):
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        else:
            self.tokenizer = tokenizer
        self.max_length = max_length

    def train_dataloader(self):
        return DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            collate_fn=self._collate_fn,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_data,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self._collate_fn,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_data,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self._collate_fn,
            pin_memory=True,
        )

    def predict_dataloader(self):
        return DataLoader(
            self.predict_data,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self._collate_fn,
            pin_memory=True,
        )

    def _collate_fn(self, batch):
        return tokenize(self.tokenizer, batch, max_length=self.max_length)["input_ids"]


def tokenize(tokenizer, text: str | list[str], max_length: int):
    return tokenizer(
        text,
        return_tensors="pt",
        padding="max_length",
        max_length=max_length + 1,
        truncation=True,
    )

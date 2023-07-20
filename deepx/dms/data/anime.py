import os
import zipfile

import requests
from torch.utils.data import Dataset

from ..classification import ClassificationDM


class AnimeDM(ClassificationDM):
    NAME = "anime"
    NUM_CLASSES = 10
    NUM_CHANNELS = 3
    SIZE = (128, 128)

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
        URL = "https://www.kaggle.com/datasets/shanmukh05/anime-names-and-image-generation/download?datasetVersionNumber=10"
        tgt_path = os.path.join(self.data_dir, "archive.zip")
        if self.download:
            if not os.path.exists(tgt_path):
                r = requests.get(URL)
                with open(tgt_path, "wb") as f:
                    f.write(r.content)
            if not os.path.exists(os.path.join(self.data_dir, "archive")):
                with zipfile.ZipFile(tgt_path) as f:
                    f.extractall(self.data_dir)
            else:
                print("Already downloaded.")

    def setup(self, stage=None):
        en_tokenizer = "bert-base-cased"
        ja_tokenizer = "cl-tohoku/bert-base-japanese"
        self.train_data = AnimeDataset(
            self.data_dir, en_tokenizer, ja_tokenizer, "train", self.max_length
        )
        self.val_data = AnimeDataset(
            self.data_dir, en_tokenizer, ja_tokenizer, "val", self.max_length
        )
        self.test_data = AnimeDataset(
            self.data_dir, en_tokenizer, ja_tokenizer, "test", self.max_length
        )


class AnimeDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        en_tokenizer: str,
        ja_tokenizer: str,
        mode: str,
        max_len: int,
    ):
        super().__init__()

        data_dir = os.path.join(data_dir, "kftt-data-1.0", "data", "orig")
        TRAIN_EN = os.path.join(data_dir, "kyoto-train.en")
        TRAIN_JA = os.path.join(data_dir, "kyoto-train.ja")
        VAL_EN = os.path.join(data_dir, "kyoto-dev.en")
        VAL_JA = os.path.join(data_dir, "kyoto-dev.ja")
        TEST_EN = os.path.join(data_dir, "kyoto-test.en")
        TEST_JA = os.path.join(data_dir, "kyoto-test.ja")

        match mode:
            case "train":
                en_path = TRAIN_EN
                ja_path = TRAIN_JA
            case "val":
                en_path = VAL_EN
                ja_path = VAL_JA
            case "test":
                en_path = TEST_EN
                ja_path = TEST_JA

        with open(en_path) as f:
            en_lines = f.readlines()
        with open(ja_path) as f:
            ja_lines = f.readlines()

        self.en_tokenizer = AutoTokenizer.from_pretrained(en_tokenizer)
        self.ja_tokenizer = AutoTokenizer.from_pretrained(ja_tokenizer)
        self.max_len = max_len

        self.en_lines = en_lines
        self.ja_lines = ja_lines

    def __len__(self):
        return len(self.en_lines)

    def __getitem__(self, idx):
        en_line = self.en_lines[idx]
        ja_line = self.ja_lines[idx]

        en_tokens = self.en_tokenizer(
            en_line,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        ja_tokens = self.ja_tokenizer(
            ja_line,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "input_ids": en_tokens["input_ids"].squeeze(),
            "attention_mask": en_tokens["attention_mask"].squeeze(),
            "labels": ja_tokens["input_ids"].squeeze(),
        }

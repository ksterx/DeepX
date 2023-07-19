import os
import tarfile

import requests
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from ..translation import TranslationDM


class KFTTDM(TranslationDM):
    NAME = "kftt"

    def prepare_data(self):
        URL = "http://www.phontron.com/kftt/download/kftt-data-1.0.tar.gz"
        tar_path = os.path.join(self.data_dir, "kftt-data-1.0.tar.gz")
        if not os.path.exists(tar_path):
            r = requests.get(URL)
            with open(tar_path, "wb") as f:
                f.write(r.content)
        if not os.path.exists(os.path.join(self.data_dir, "kftt-data-1.0")):
            with tarfile.open(tar_path) as f:
                f.extractall(self.data_dir)
        else:
            print("Already downloaded.")

    def setup(self, stage=None):
        en_tokenizer = "bert-base-cased"
        ja_tokenizer = "cl-tohoku/bert-base-japanese"
        self.train_data = KFTTDataset(
            self.data_dir, en_tokenizer, ja_tokenizer, "train", self.max_length
        )
        self.val_data = KFTTDataset(
            self.data_dir, en_tokenizer, ja_tokenizer, "val", self.max_length
        )
        self.test_data = KFTTDataset(
            self.data_dir, en_tokenizer, ja_tokenizer, "test", self.max_length
        )


class KFTTDataset(Dataset):
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

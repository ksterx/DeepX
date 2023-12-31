import glob
import os
import zipfile

import requests  # type: ignore
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from ..classification import ClassificationDM


class AnimeDM(ClassificationDM):
    NAME = "anime"
    NUM_CLASSES = None
    NUM_CHANNELS = 3
    SIZE = (64, 64)

    def __init__(
        self,
        data_dir: str,
        batch_size: int,
        num_workers: int,
        train_ratio: float,
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
        data = AnimeDataset(
            data_dir=self.data_dir, transform=self.transform(self.mean, self.std)
        )
        self.train_data, self.val_data = self._random_split(data, self.train_ratio)
        self.test_data = self.val_data

    @classmethod
    def transform(cls, mean, std):
        return transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize(
                    cls.SIZE, antialias=True, interpolation=Image.BICUBIC
                ),
                transforms.Normalize(mean=mean, std=std),
            ]
        )


class AnimeDataset(Dataset):
    def __init__(self, data_dir: str, transform):
        super().__init__()

        self.data_dir = os.path.join(data_dir, "anime", "dataset")
        self.img_paths = sorted(glob.glob(os.path.join(self.data_dir, "*.jpg")))
        self.transform = transform

        assert len(self.img_paths) > 0

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = Image.open(self.img_paths[idx]).convert("RGB")
        img = self.transform(img)
        return img, 0

from torchvision.datasets import CIFAR100

from ..classification import ClassificationDM


class CIFAR100DM(ClassificationDM):
    NAME = "cifar100"
    NUM_CLASSES = 100
    NUM_CHANNELS = 3
    SIZE = (32, 32)
    CLASSES = [
        "apple",
        "aquarium_fish",
        "baby",
        "bear",
        "beaver",
        "bed",
        "bee",
        "beetle",
        "bicycle",
        "bottle",
        "bowl",
        "boy",
        "bridge",
        "bus",
        "butterfly",
        "camel",
        "can",
        "castle",
        "caterpillar",
        "cattle",
        "chair",
        "chimpanzee",
        "clock",
        "cloud",
        "cockroach",
        "couch",
        "crab",
        "crocodile",
        "cup",
        "dinosaur",
        "dolphin",
        "elephant",
        "flatfish",
        "forest",
        "fox",
        "girl",
        "hamster",
        "house",
        "kangaroo",
        "keyboard",
        "lamp",
        "lawn_mower",
        "leopard",
        "lion",
        "lizard",
        "lobster",
        "man",
        "maple_tree",
        "motorcycle",
        "mountain",
        "mouse",
        "mushroom",
        "oak_tree",
        "orange",
        "orchid",
        "otter",
        "palm_tree",
        "pear",
        "pickup_truck",
        "pine_tree",
        "plain",
        "plate",
        "poppy",
        "porcupine",
        "possum",
        "rabbit",
        "raccoon",
        "ray",
        "road",
        "rocket",
        "rose",
        "sea",
        "seal",
        "shark",
        "shrew",
        "skunk",
        "skyscraper",
        "snail",
        "snake",
        "spider",
        "squirrel",
        "streetcar",
        "sunflower",
        "sweet_pepper",
        "table",
        "tank",
        "telephone",
        "television",
        "tiger",
        "tractor",
        "train",
        "trout",
        "tulip",
        "turtle",
        "wardrobe",
        "whale",
        "willow_tree",
        "wolf",
        "woman",
        "worm",
    ]

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
            **kwargs,
        )

        self.train_ratio = train_ratio
        self.download = download

    def prepare_data(self):
        CIFAR100(self.data_dir, train=True, download=self.download)
        CIFAR100(self.data_dir, train=False, download=self.download)

    def setup(self, stage=None):
        if stage == "fit":
            data = CIFAR100(self.data_dir, train=True, transform=self.train_transform())
            self.train_data, self.val_data = self._random_split(data, self.train_ratio)

            self.test_data = CIFAR100(
                self.data_dir, train=False, transform=self.transform()
            )

        if stage == "predict":
            self.predict_data = CIFAR100(
                self.data_dir, train=False, transform=self.transform()
            )

    @classmethod
    def transform(cls):
        return cls._transform(cls.SIZE)

    @classmethod
    def train_transform(cls):
        return cls._train_transform(cls.SIZE)

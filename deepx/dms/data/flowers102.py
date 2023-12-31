from torchvision.datasets import Flowers102

from ..classification import ClassificationDM


class Flowers102DM(ClassificationDM):
    NAME = "flowers102"
    NUM_CLASSES = 102
    NUM_CHANNELS = 3
    SIZE = (32, 32)
    CLASSES = [
        "pink primrose",
        "hard-leaved pocket orchid",
        "canterbury bells",
        "sweet pea",
        "english marigold",
        "tiger lily",
        "moon orchid",
        "bird of paradise",
        "monkshood",
        "globe thistle",
        "snapdragon",
        "colt's foot",
        "king protea",
        "spear thistle",
        "yellow iris",
        "globe-flower",
        "purple coneflower",
        "peruvian lily",
        "balloon flower",
        "giant white arum lily",
        "fire lily",
        "pincushion flower",
        "fritillary",
        "red ginger",
        "grape hyacinth",
        "corn poppy",
        "prince of wales feathers",
        "stemless gentian",
        "artichoke",
        "sweet william",
        "carnation",
        "garden phlox",
        "love in the mist",
        "mexican aster",
        "alpine sea holly",
        "ruby-lipped cattleya",
        "cape flower",
        "great masterwort",
        "siam tulip",
        "lenten rose",
        "barbeton daisy",
        "daffodil",
        "sword lily",
        "poinsettia",
        "bolero deep blue",
        "wallflower",
        "marigold",
        "buttercup",
        "oxeye daisy",
        "common dandelion",
        "petunia",
        "wild pansy",
        "primula",
        "sunflower",
        "pelargonium",
        "bishop of llandaff",
        "gaura",
        "geranium",
        "orange dahlia",
        "pink-yellow dahlia?",
        "cautleya spicata",
        "japanese anemone",
        "black-eyed susan",
        "silverbush",
        "californian poppy",
        "osteospermum",
        "spring crocus",
        "bearded iris",
        "windflower",
        "tree poppy",
        "gazania",
        "azalea",
        "water lily",
        "rose",
        "thorn apple",
        "morning glory",
        "passion flower",
        "lotus",
        "toad lily",
        "anthurium",
        "frangipani",
        "clematis",
        "hibiscus",
        "columbine",
        "desert-rose",
        "tree mallow",
        "magnolia",
        "cyclamen",
        "watercress",
        "canna lily",
        "hippeastrum",
        "bee balm",
        "ball moss",
        "foxglove",
        "bougainvillea",
        "camellia",
        "mallow",
        "mexican petunia",
        "bromelia",
        "blanket flower",
        "trumpet creeper",
        "blackberry lily",
    ]

    def __init__(
        self,
        data_dir: str,
        batch_size: int,
        num_workers: int,
        download: bool,
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

        self.download = download
        self.mean = mean
        self.std = std

    def prepare_data(self):
        Flowers102(self.data_dir, split="train", download=self.download)
        Flowers102(self.data_dir, split="val", download=self.download)
        Flowers102(self.data_dir, split="test", download=self.download)

    def setup(self, stage=None):
        if stage == "fit" or None:
            self.train_data = Flowers102(
                self.data_dir,
                split="train",
                transform=self.train_transform(self.mean, self.std),
            )
            self.val_data = Flowers102(
                self.data_dir,
                split="val",
                transform=self.transform(self.mean, self.std),
            )

        if stage == "test" or None:
            self.test_data = Flowers102(
                self.data_dir,
                split="test",
                transform=self.transform(self.mean, self.std),
            )

        if stage == "predict" or None:
            self.predict_data = Flowers102(
                self.data_dir,
                split="test",
                transform=self.transform(self.mean, self.std),
            )

    @classmethod
    def transform(cls, mean, std):
        return cls._transform(cls.SIZE, mean, std)

    @classmethod
    def train_transform(cls, mean, std):
        return cls._train_transform(cls.SIZE, mean, std)

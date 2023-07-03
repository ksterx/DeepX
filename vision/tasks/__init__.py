from torchvision import transforms
from torchvision.datasets import (
    CIFAR10,
    CIFAR100,
    MNIST,
    FashionMNIST,
    Flowers102,
    VOCSegmentation,
)

from vision.tasks.base import DataModule, Task
from vision.tasks.classification import ClassificationDataset, ClassificationTask
from vision.tasks.segmentation import SegmentationDataset, SegmentationTask

__all__ = [
    "DataModule",
    "Task",
    "ClassificationDataset",
    "ClassificationTask",
    "SegmentationDataset",
    "SegmentationTask",
]


def transform(size):
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize(size, antialias=True),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )


available_datasets = {
    "mnist": {
        "class": MNIST,
        "num_classes": len(MNIST.classes),
        "num_channels": 1,
        "size": (8, 8),
        "transform": transform((8, 8)),
    },
    "fashionmnist": {
        "class": FashionMNIST,
        "num_classes": len(FashionMNIST.classes),
        "num_channels": 1,
        "size": (8, 8),
        "transform": transform((8, 8)),
    },
    "cifar10": {
        "class": CIFAR10,
        "num_classes": 10,
        "num_channels": 3,
        "classes": [
            "plane",
            "car",
            "bird",
            "cat",
            "deer",
            "dog",
            "frog",
            "horse",
            "ship",
            "truck",
        ],
        "size": (32, 32),
        "transform": transform((32, 32)),
    },
    "cifar100": {
        "class": CIFAR100,
        "num_classes": 100,
        "num_channels": 3,
        "size": (32, 32),
        "classes": [
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
        ],
        "transform": transform((32, 32)),
    },
    "flowers102": {
        "class": Flowers102,
        "num_classes": 102,
        "num_channels": 3,
        "classes": [
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
        ],
        "transform": transform((224, 224)),
    },
    "voc": {
        "class": VOCSegmentation,
        "num_classes": 21,
        "num_channels": 3,
        "transform": transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize(
                    (224, 224), interpolation=transforms.InterpolationMode.NEAREST_EXACT
                ),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        ),
        "target_transform": transforms.Compose(
            [
                transforms.PILToTensor(),
                transforms.Resize(
                    (224, 224), interpolation=transforms.InterpolationMode.NEAREST_EXACT
                ),
            ]
        ),
    },
}

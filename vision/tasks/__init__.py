from torchvision.datasets import CIFAR10, CIFAR100, MNIST

available_datasets = {
    "mnist": {"class": MNIST, "num_classes": 10, "num_channels": 1},
    "cifar10": {
        "class": CIFAR10,
        "num_classes": 10,
        "num_channels": 3,
        "labels": [
            "airplane",
            "automobile",
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
    },
    "cifar100": {"class": CIFAR100, "num_classes": 100, "num_channels": 3},
}

from torchvision.datasets import VOCSegmentation

from ..segmentation import SegmentationDM


class VOCSegDM(SegmentationDM):
    NAME = "vocseg"
    NUM_CLASSES = 21
    NUM_CHANNELS = 3
    SIZE = (224, 224)
    CLASSES = [
        "background",
        "aeroplane",
        "bicycle",
        "bird",
        "boat",
        "bottle",
        "bus",
        "car",
        "cat",
        "chair",
        "cow",
        "diningtable",
        "dog",
        "horse",
        "motorbike",
        "person",
        "pottedplant",
        "sheep",
        "sofa",
        "train",
        "tvmonitor",
    ]

    def __init__(
        self,
        data_dir: str,
        batch_size: int,
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

        self.download = download

    def prepare_data(self):
        VOCSegmentation(self.data_dir, image_set="train", download=self.download)
        VOCSegmentation(self.data_dir, image_set="val", download=self.download)

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.train_data = VOCSegmentation(
                self.data_dir,
                image_set="train",
                transform=self.transform(),
                target_transform=self.target_transform(),
            )
            self.val_data = VOCSegmentation(
                self.data_dir,
                image_set="val",
                transform=self.transform(),
                target_transform=self.target_transform(),
            )

        if stage == "test" or stage is None:
            self.test_data = VOCSegmentation(
                self.data_dir,
                image_set="val",
                transform=self.transform(),
                target_transform=self.target_transform(),
            )

        if stage == "predict":
            self.predict_data = VOCSegmentation(
                self.data_dir,
                image_set="val",
                transform=self.transform(),
                target_transform=self.target_transform(),
            )

    @classmethod
    def transform(cls):
        return cls._transform(cls.SIZE)

    @classmethod
    def target_transform(cls):
        return cls._target_transform(cls.SIZE)

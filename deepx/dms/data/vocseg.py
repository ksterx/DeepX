from torchvision.datasets import VOCSegmentation

from ..segmentation import SegmentationDM


class VOCSegDM(SegmentationDM):
    NAME = "vocseg"
    NUM_CLASSES = 21
    NUM_CHANNELS = 3
    SIZE = (224, 224)

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
        )

        self.download = download

        self._transform = self.transform(self.SIZE)
        self._target_transform = self.target_transform(self.SIZE)

    def prepare_data(self):
        VOCSegmentation(self.data_dir, image_set="train", download=self.download)
        VOCSegmentation(self.data_dir, image_set="val", download=self.download)

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.train_data = VOCSegmentation(
                self.data_dir,
                image_set="train",
                transform=self._transform,
                target_transform=self._target_transform,
            )
            self.val_data = VOCSegmentation(
                self.data_dir,
                image_set="val",
                transform=self._transform,
                target_transform=self._target_transform,
            )

        if stage == "test" or stage is None:
            self.test_data = VOCSegmentation(
                self.data_dir,
                image_set="val",
                transform=self._transform,
                target_transform=self._target_transform,
            )

        if stage == "predict":
            self.predict_data = VOCSegmentation(
                self.data_dir,
                image_set="val",
                transform=self._transform,
                target_transform=self._target_transform,
            )

from torchvision import transforms

from .dm import DataModuleX


class SegmentationDM(DataModuleX):
    def transform(
        self,
        size,
        antialias=True,
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5),
        interpolation=transforms.InterpolationMode.NEAREST_EXACT,
    ):
        return transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize(size, antialias=antialias, interpolation=interpolation),
                transforms.Normalize(mean, std),
            ]
        )

    def target_transform(
        self, size, antialias=True, interpolation=transforms.InterpolationMode.NEAREST_EXACT
    ):
        return transforms.Compose(
            [
                transforms.PILToTensor(),
                transforms.Resize(size, antialias=antialias, interpolation=interpolation),
            ]
        )

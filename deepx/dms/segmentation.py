from torchvision import transforms

from .dm import DataModuleX


class SegmentationDM(DataModuleX):
    SIZE: tuple[int, int]

    @classmethod
    def _transform(
        cls,
        size,
        antialias=True,
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5),
        interpolation=transforms.InterpolationMode.NEAREST_EXACT,
    ) -> transforms.Compose:
        return transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize(
                    size, antialias=antialias, interpolation=interpolation
                ),
                transforms.Normalize(mean, std),
            ]
        )

    @classmethod
    def _target_transform(
        cls,
        size,
        antialias=True,
        interpolation=transforms.InterpolationMode.NEAREST_EXACT,
    ) -> transforms.Compose:
        return transforms.Compose(
            [
                transforms.PILToTensor(),
                transforms.Resize(
                    size, antialias=antialias, interpolation=interpolation
                ),
            ]
        )

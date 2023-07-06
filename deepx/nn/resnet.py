import torch
from torch import nn

__all__ = ["ResNet", "ResNet18", "ResNet34", "ResNet50", "ResNet101", "ResNet152"]


class ResidualBlock(nn.Module):
    CHANNELS_IN_LAYERS = [64, 128, 256, 512]
    EXPANSION = 1

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        is_first: bool = False,
        p_dropout: float = 0.5,
    ):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(hidden_channels, out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p_dropout)

        if is_first:
            self.rescale = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
        else:
            self.rescale = lambda x: x

    def forward(self, x):
        """
        x
        |-------+
        bn1     |
        |       |
        relu    |
        |       |
        conv1   |
        |       |
        bn2     |
        |       |
        relu    |
        |       |
        conv2   |
        |       |
        dropout |
        |-------+
        out

        """

        identity = x
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.dropout(out)
        identity = self.rescale(identity)
        out += identity
        return out


class BottleneckBlock(nn.Module):
    CHANNELS_IN_LAYERS = [64, 128, 256, 512]
    EXPANSION = 4

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        is_first: bool = False,
        p_dropout: float = 0.5,
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=1, padding=0)
        self.bn1 = nn.BatchNorm2d(hidden_channels)
        self.conv2 = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(hidden_channels)
        self.conv3 = nn.Conv2d(hidden_channels, out_channels, kernel_size=1, padding=0)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p_dropout)

        if is_first:
            self.rescale = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
        else:
            self.rescale = lambda x: x

    def forward(self, x):
        """
        x
        |-------+
        conv1   |
        |       |
        bn1     |
        |       |
        relu    |
        |       |
        conv2   |
        |       |
        bn2     |
        |       |
        relu    |
        |       |
        conv3   |
        |       |
        bn3     |
        |       |
        dropout |
        |-------+
        relu
        |
        out

        """

        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        out = self.dropout(out)
        identity = self.rescale(identity)
        out += identity
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        block: ResidualBlock | BottleneckBlock,
    ) -> None:
        super().__init__()
        self.input_channels = 64
        self.num_classes = num_classes
        self.block = block

        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layers = lambda x: x
        self.head = self._make_head()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layers(x)
        x = self.head(x)
        return x

    def _make_layers(
        self,
        blocks_in_layers: list[int],
        p_dropout: float = 0.5,
    ):
        layers = []
        for i, c in enumerate(self.block.CHANNELS_IN_LAYERS):
            for j in range(blocks_in_layers[i]):
                if j == 0:
                    layers.append(
                        self.block(
                            self.input_channels,
                            c,
                            c * self.block.EXPANSION,
                            is_first=True,
                            p_dropout=p_dropout,
                        )
                    )
                    self.input_channels = c * self.block.EXPANSION
                else:
                    layers.append(
                        self.block(
                            self.input_channels,
                            c,
                            c * self.block.EXPANSION,
                            is_first=False,
                            p_dropout=p_dropout,
                        )
                    )

        return nn.Sequential(*layers)

    def _make_head(self):
        return nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(self.block.CHANNELS_IN_LAYERS[-1] * self.block.EXPANSION, self.num_classes),
        )


class ResNet18(ResNet):
    def __init__(self, num_classes, in_channels: int = 3):
        super().__init__(in_channels, num_classes, ResidualBlock)
        self.layers = self._make_layers([2, 2, 2, 2], p_dropout=0.3)


class ResNet34(ResNet):
    def __init__(self, num_classes, in_channels: int = 3):
        super().__init__(in_channels, num_classes, ResidualBlock)
        self.layers = self._make_layers([3, 4, 6, 3], p_dropout=0.3)


class ResNet50(ResNet):
    def __init__(self, num_classes, in_channels: int = 3):
        super().__init__(in_channels, num_classes, BottleneckBlock)
        self.layers = self._make_layers([3, 4, 6, 3], p_dropout=0.3)


class ResNet101(ResNet):
    def __init__(self, num_classes, in_channels: int = 3):
        super().__init__(in_channels, num_classes, BottleneckBlock)
        self.layers = self._make_layers([3, 4, 23, 3], p_dropout=0.3)


class ResNet152(ResNet):
    def __init__(self, num_classes, in_channels: int = 3):
        super().__init__(in_channels, num_classes, BottleneckBlock)
        self.layers = self._make_layers([3, 8, 36, 3], p_dropout=0.3)

from typing import List, Optional

import lightning as L
import torch
from torch import nn


class BottleneckBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels,
        out_channels,
        is_first: bool = False,
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=1, padding=0)
        self.bn1 = nn.BatchNorm2d(hidden_channels)
        self.conv2 = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(hidden_channels)
        self.conv3 = nn.Conv2d(hidden_channels, out_channels, kernel_size=1, padding=0)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

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
        identity = self.rescale(identity)
        out += identity
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    CHANNELS_IN_LAYERS = [64, 128, 256, 512]
    EXPANSION = 4

    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.input_channels = 64
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(2048, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layers(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        return x

    def _make_layers(self, block, blocks_in_layers: List[int]):
        layers = []
        for i, c in enumerate(self.CHANNELS_IN_LAYERS):
            for j in range(blocks_in_layers[i]):
                if j == 0:
                    layers.append(block(self.input_channels, c, c * self.EXPANSION, True))
                    self.input_channels = c * self.EXPANSION
                else:
                    layers.append(block(self.input_channels, c, c * self.EXPANSION))

        return nn.Sequential(*layers)


class ResNet50(ResNet):
    def __init__(
        self, num_classes, in_channels: int = 3, block: Optional[nn.Module] = BottleneckBlock
    ):
        super().__init__(in_channels, num_classes)
        self.layers = self._make_layers(block, [3, 4, 6, 3])

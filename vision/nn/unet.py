import torch
from torch import nn


class UNet(nn.Module):
    def __init__(self, in_channels: int = 3, num_classes: int = 1):
        """
        enc1 ------------------------------- dec1
            |                                | up1
            enc2 ----------------------- dec2
                |                        | up2
                enc3 --------------- dec3
                    |                | up3
                    enc4 ------- dec4
                        |        | up4
                        |--enc5--|
        """
        super().__init__()

        HIDDEN_CHANNELS_LAYERS = [in_channels, 64, 128, 256, 512, 1024]
        self.encs = nn.ModuleList()  # [enc1, enc2, enc3, enc4, enc5]
        self.decs = nn.ModuleList()  # [dec1, dec2, dec3, dec4]
        self.ups = nn.ModuleList()  # [up1, up2, up3, up4]

        for i in range(len(HIDDEN_CHANNELS_LAYERS) - 1):
            self.encs.append(
                Conv3x3BNReLU(
                    [
                        HIDDEN_CHANNELS_LAYERS[i],
                        HIDDEN_CHANNELS_LAYERS[i + 1],
                        HIDDEN_CHANNELS_LAYERS[i + 1],
                    ]
                )
            )
            if i != 0:
                self.decs.append(
                    Conv3x3BNReLU(
                        [
                            HIDDEN_CHANNELS_LAYERS[i] * 2,
                            HIDDEN_CHANNELS_LAYERS[i],
                            HIDDEN_CHANNELS_LAYERS[i],
                        ]
                    )
                )
                self.ups.append(
                    self.upconv(HIDDEN_CHANNELS_LAYERS[i + 1], HIDDEN_CHANNELS_LAYERS[i])
                )

        self.final = nn.Conv2d(64, num_classes, kernel_size=1)

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

    def forward(self, x):
        identities = []

        # Encoder
        for enc in self.encs[:-1]:
            x = enc(x)
            identities.append(x)
            x = self.maxpool(x)
        x = self.encs[-1](x)

        # Decoder
        for identity, dec, up in zip(reversed(identities), reversed(self.decs), reversed(self.ups)):
            x = up(x)
            x = self.crop_cat(x, identity)
            x = dec(x)
        x = self.final(x)
        return x

    def crop_cat(self, x, identity):
        _, _, h, w = identity.shape
        _, _, h_, w_ = x.shape
        pad_h = (h_ - h) / 2
        pad_w = (w_ - w) / 2
        if pad_h != 0 and pad_w != 0:
            x = x[:, :, pad_h:-pad_h, pad_w:-pad_w]
        x = torch.cat([x, identity], dim=1)
        return x

    def upconv(self, in_channels, out_channels, bilinear=True):
        if bilinear:
            return nn.Sequential(
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
            )
        else:
            return nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
            )


class Conv3x3BNReLU(nn.Module):
    def __init__(
        self,
        channels_in_layers: list[int],
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = "same",
    ):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(len(channels_in_layers) - 1):
            self.layers.append(
                nn.Sequential(
                    nn.Conv2d(
                        channels_in_layers[i],
                        channels_in_layers[i + 1],
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=padding,
                    ),
                    nn.BatchNorm2d(channels_in_layers[i + 1]),
                    nn.ReLU(),
                )
            )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

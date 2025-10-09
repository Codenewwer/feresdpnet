import numpy as np
import torch
import torch.nn as nn


def get_eca_kernel_size(channel: int, gamma: int = 2, b: int = 1) -> int:
    """Dynamically determine 1-D conv kernel size for ECA."""
    k = int(abs((np.log2(channel) / gamma) + b))
    return k if k % 2 == 1 else k + 1  # ensure odd number


class ECALayer(nn.Module):
    """Efficient Channel Attention (ECA) block."""
    def __init__(self, channel: int):
        super().__init__()
        k_size = get_eca_kernel_size(channel)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(
            1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)


class LKCRB(nn.Module):
    """
    Long-Kernel Convolutional Residual Block (rename of ResidualBlock).
    Includes down-sampling option and integrated ECA.
    """
    def __init__(self, in_channels: int, out_channels: int, downsample: bool = False):
        super().__init__()
        stride = 2 if downsample else 1

        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=7,
            stride=stride, padding=3, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=7,
            stride=1, padding=3, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Shortcut branch
        self.shortcut = nn.Sequential()
        if downsample or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

        self.eca = ECALayer(out_channels)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return self.eca(out)


class ESDPNet(nn.Module):
    """
    ES-DPNet backbone + optional HOG branch.
    """
    def __init__(self, num_classes: int = 7, hog_input_dim: int | None = None):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)

        # Five LKCRB stages
        self.layer0 = self._make_layer(16, 32, downsample=True)
        self.layer1 = self._make_layer(32, 64, downsample=True)
        self.layer2 = self._make_layer(64, 128, downsample=True)
        self.layer3 = self._make_layer(128, 256, downsample=True)
        self.layer4 = self._make_layer(256, 512, downsample=True)

        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc_cnn = nn.Linear(512, num_classes)

        # HOG branch (optional)
        if hog_input_dim is not None:
            self.hog_head = nn.Sequential(
                nn.Linear(hog_input_dim, 512), nn.ReLU(inplace=True),
                nn.Linear(512, 256),          nn.ReLU(inplace=True),
                nn.Linear(256, 128),          nn.ReLU(inplace=True),
                nn.Linear(128, num_classes)
            )
        else:
            self.hog_head = None

    @staticmethod
    def _make_layer(in_ch: int, out_ch: int, downsample: bool):
        """Single-block layer builder."""
        return nn.Sequential(LKCRB(in_ch, out_ch, downsample=downsample))

    def forward(self, x, hog_features=None):
        # Stem
        x = self.relu(self.bn1(self.conv1(x)))

        # Stages
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # Global pooling & FC
        x = self.global_avg_pool(x).flatten(1)
        out_cnn = self.fc_cnn(x)

        # Optional HOG fusion
        if self.hog_head is not None and hog_features is not None:
            out_hog = self.hog_head(hog_features)
            return out_cnn + out_hog
        return out_cnn
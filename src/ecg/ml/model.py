"""ResNet1D model architecture module for ECG-ML-STREAM.

Copyright 2026 Mateusz Golebiewski
"""

import torch
import torch.nn.functional as F  # noqa: N812 - import as F is a common PyTorch convention
from torch import nn

from ecg.utils.constants import (
    NUM_CLASSES,
    NUM_LEADS,
)


class ResidualBlock1D(nn.Module):
    """One-dimenstional residual block for ResNet1D."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 7,
        stride: int = 1,
        dropout: float = 0.2,
    ) -> None:
        """Initialize ResidualBlock1D.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            kernel_size (int): Size of the convolutional kernel.
            stride (int): Stride for the convolution.
            dropout (float): Dropout rate.

        """
        super().__init__()

        padding = kernel_size // 2

        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )
        self.bn1 = nn.BatchNorm1d(out_channels)

        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=padding,
            bias=False,
        )
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.dropout = nn.Dropout(dropout)

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels),
            )
        else:
            self.shortcut = nn.Sequential()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute residual block output.

        Args:
            x (torch.Tensor): Input tensor of shape `(batch_size, in_channels, seq_length)`.

        Returns:
            torch.Tensor: Output tensor of shape `(batch_size, out_channels, seq_length // stride)`.

        """
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.dropout(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += self.shortcut(x)
        return F.relu(out)


class ResNet1D(nn.Module):
    """ResNet1D classifier for 12-lead ECG signals."""

    def __init__(
        self,
        input_channels: int = NUM_LEADS,
        num_classes: int = NUM_CLASSES,
        base_filters: int = 64,
        kernel_size: int = 7,
        num_blocks: list[int] | None = None,
        dropout: float = 0.2,
    ) -> None:
        """Initialize ResNet1D.

        Args:
            input_channels (int): Number of input channels (ECG leads).
            num_classes (int): Number of output classes.
            base_filters (int): Number of filters in the first block.
            kernel_size (int): Size of the convolutional kernel.
            num_blocks (list[int]): Number of residual blocks in each stage.
            dropout (float): Dropout rate.

        """
        super().__init__()

        if num_blocks is None:
            num_blocks = [2, 2, 2, 2]

        self.input_channels = input_channels
        self.num_classes = num_classes

        self.conv1 = nn.Conv1d(
            input_channels,
            base_filters,
            kernel_size=15,
            stride=2,
            padding=7,
            bias=False,
        )
        self.bn1 = nn.BatchNorm1d(base_filters)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(
            base_filters,
            base_filters,
            num_blocks[0],
            kernel_size,
            stride=1,
            dropout=dropout,
        )
        self.layer2 = self._make_layer(
            base_filters,
            base_filters * 2,
            num_blocks[1],
            kernel_size,
            stride=2,
            dropout=dropout,
        )
        self.layer3 = self._make_layer(
            base_filters * 2,
            base_filters * 4,
            num_blocks[2],
            kernel_size,
            stride=2,
            dropout=dropout,
        )
        self.layer4 = self._make_layer(
            base_filters * 4,
            base_filters * 8,
            num_blocks[3],
            kernel_size,
            stride=2,
            dropout=dropout,
        )

        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(base_filters * 8, num_classes)

        self._initialize_weights()

    @staticmethod
    def _make_layer(
        in_channels: int,
        out_channels: int,
        num_blocks: int,
        kernel_size: int,
        stride: int,
        dropout: float,
    ) -> nn.Sequential:
        """Build a residual group from multiple ResidualBlock1D blocks.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            num_blocks (int): Number of residual blocks in the stage.
            kernel_size (int): Size of the convolutional kernel.
            stride (int): Stride for the first block in the stage.
            dropout (float): Dropout rate.

        Returns:
            nn.Sequential: Sequential container of residual blocks.

        """
        layers = []
        layers.append(ResidualBlock1D(in_channels, out_channels, kernel_size, stride, dropout))

        layers.extend(
            ResidualBlock1D(out_channels, out_channels, kernel_size, 1, dropout)
            for _ in range(1, num_blocks)
        )

        return nn.Sequential(*layers)

    def _initialize_weights(self) -> None:
        """Apply Kaiming He initialization to convolutional and linear layers."""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute class logits for a batch of ECG windows.

        Args:
            x (torch.Tensor): Input tensor of shape `(batch_size, input_channels, seq_length)`.
                Example: `(32, 12, 250)` for a batch of 32 windows of 2.5 seconds at 100 Hz.

        Returns:
            torch.Tensor: Output tensor of shape `(batch_size, num_classes)` containing
            class logits.

        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Compute class probabilities for a batch of ECG windows.

        Args:
            x (torch.Tensor): Input tensor of shape `(batch_size, input_channels, seq_length)`.

        Returns:
            torch.Tensor: Output tensor of shape `(batch_size, num_classes)`
            containing class probabilities.

        """
        logits = self.forward(x)
        return F.softmax(logits, dim=1)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Return the predicted class index for each sample in the batch.

        Args:
            x (torch.Tensor): Input tensor of shape `(batch_size, input_channels, seq_length)`.

        Returns:
            torch.Tensor: Output tensor of shape `(batch_size,)` containing predicted class indices.

        """
        return torch.argmax(self.forward(x), dim=1)

from __future__ import annotations

import torch
from torch import nn


class ParkingSlotCNN(nn.Module):
    def __init__(self, num_classes: int = 2, dropout: float = 0.35) -> None:
        super().__init__()
        self.features = nn.Sequential(
            self._conv_block(3, 32),
            nn.MaxPool2d(kernel_size=2),
            self._conv_block(32, 64),
            nn.MaxPool2d(kernel_size=2),
            self._conv_block(64, 128),
            nn.MaxPool2d(kernel_size=2),
            self._conv_block(128, 192),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout),
            nn.Linear(192, 96),
            nn.ReLU(inplace=True),
            nn.Dropout(p=max(0.1, dropout * 0.6)),
            nn.Linear(96, num_classes),
        )

    @staticmethod
    def _conv_block(in_channels: int, out_channels: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)

class ParkingSlotCNNV2(nn.Module):
    def __init__(self, num_classes: int = 2, dropout: float = 0.3) -> None:
        super().__init__()
        self.features = nn.Sequential(
            self._conv_block(3, 48),
            nn.MaxPool2d(kernel_size=2),
            self._conv_block(48, 96),
            nn.MaxPool2d(kernel_size=2),
            self._conv_block(96, 160),
            nn.MaxPool2d(kernel_size=2),
            self._conv_block(160, 256),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=max(0.1, dropout * 0.5)),
            nn.Linear(128, num_classes),
        )

    @staticmethod
    def _conv_block(in_channels: int, out_channels: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)


def build_model(
    num_classes: int = 2,
    model_version: str = "v1",
    dropout: float = 0.35,
) -> nn.Module:
    if model_version == "v1":
        return ParkingSlotCNN(num_classes=num_classes, dropout=dropout)
    if model_version == "v2":
        return ParkingSlotCNNV2(num_classes=num_classes, dropout=dropout)
    raise ValueError(f"Unsupported CNN model version: {model_version}")

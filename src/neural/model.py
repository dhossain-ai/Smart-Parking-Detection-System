from __future__ import annotations

import torch
from torch import nn


class ParkingSlotCNN(nn.Module):
    def __init__(self, num_classes: int = 2) -> None:
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
            nn.Dropout(p=0.35),
            nn.Linear(192, 96),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
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


def build_model(num_classes: int = 2) -> ParkingSlotCNN:
    return ParkingSlotCNN(num_classes=num_classes)

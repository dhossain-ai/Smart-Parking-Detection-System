from __future__ import annotations

from typing import Any

import torch
from torch import nn


class _ParkingSlotBaseCNN(nn.Module):
    def __init__(
        self,
        channels: tuple[int, int, int, int],
        hidden_units: int,
        dropout: float,
        num_classes: int = 2,
    ) -> None:
        super().__init__()

        c1, c2, c3, c4 = channels

        self.features = nn.Sequential(
            self._conv_block(3, c1),
            nn.MaxPool2d(kernel_size=2),
            self._conv_block(c1, c2),
            nn.MaxPool2d(kernel_size=2),
            self._conv_block(c2, c3),
            nn.MaxPool2d(kernel_size=2),
            self._conv_block(c3, c4),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout),
            nn.Linear(c4, hidden_units),
            nn.ReLU(inplace=True),
            nn.Dropout(p=max(0.1, dropout * 0.5)),
            nn.Linear(hidden_units, num_classes),
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


class ParkingSlotCNN(_ParkingSlotBaseCNN):
    def __init__(self, num_classes: int = 2, dropout: float = 0.35) -> None:
        super().__init__(
            channels=(32, 64, 128, 192),
            hidden_units=96,
            dropout=dropout,
            num_classes=num_classes,
        )


class ParkingSlotCNNV2(_ParkingSlotBaseCNN):
    def __init__(self, num_classes: int = 2, dropout: float = 0.3) -> None:
        super().__init__(
            channels=(48, 96, 160, 256),
            hidden_units=128,
            dropout=dropout,
            num_classes=num_classes,
        )


def build_model(
    num_classes: int = 2,
    model_version: str = "v2",
    pretrained: bool = False,
    freeze_backbone: bool = False,
    dropout: float = 0.3,
) -> nn.Module:
    # This function keeps all neural model choices in one place.

    if model_version == "v1":
        return ParkingSlotCNN(num_classes=num_classes, dropout=dropout)

    if model_version == "v2":
        return ParkingSlotCNNV2(num_classes=num_classes, dropout=dropout)

    if model_version == "mobilenet_v3_small":
        return _build_mobilenet_v3_small(
            num_classes=num_classes,
            pretrained=pretrained,
            freeze_backbone=freeze_backbone,
            dropout=dropout,
        )

    raise ValueError(f"Unsupported model version: {model_version}")


def _build_mobilenet_v3_small(
    num_classes: int,
    pretrained: bool,
    freeze_backbone: bool,
    dropout: float,
) -> nn.Module:
    try:
        from torchvision import models
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError(
            "MobileNetV3-Small needs torchvision. Install requirements.txt first."
        ) from error

    model = _load_mobilenet_v3_small(models=models, pretrained=pretrained)
    _replace_classifier(model=model, num_classes=num_classes, dropout=dropout)

    if freeze_backbone:
        for parameter in model.features.parameters():
            parameter.requires_grad = False

        for parameter in model.classifier.parameters():
            parameter.requires_grad = True

    return model


def _load_mobilenet_v3_small(models: Any, pretrained: bool) -> nn.Module:
    if not pretrained:
        return models.mobilenet_v3_small(weights=None)

    try:
        weights = models.MobileNet_V3_Small_Weights.DEFAULT
        return models.mobilenet_v3_small(weights=weights)
    except AttributeError:
        return models.mobilenet_v3_small(pretrained=True)


def _replace_classifier(model: nn.Module, num_classes: int, dropout: float) -> None:
    classifier = getattr(model, "classifier", None)

    if not isinstance(classifier, nn.Sequential):
        raise ValueError("Unsupported MobileNetV3 classifier structure.")

    for layer in classifier:
        if isinstance(layer, nn.Dropout):
            layer.p = dropout

    output_layer_index = _find_last_linear_layer(classifier)
    old_output_layer = classifier[output_layer_index]

    if not isinstance(old_output_layer, nn.Linear):
        raise ValueError("MobileNetV3 output layer must be Linear.")

    classifier[output_layer_index] = nn.Linear(
        old_output_layer.in_features,
        num_classes,
    )


def _find_last_linear_layer(classifier: nn.Sequential) -> int:
    for index in range(len(classifier) - 1, -1, -1):
        if isinstance(classifier[index], nn.Linear):
            return index

    raise ValueError("MobileNetV3 classifier has no Linear layer.")
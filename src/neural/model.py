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

        # Feature extractor for the custom CNN.
        # It uses convolution, batch normalization, ReLU, and pooling.
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

        # Final classifier that predicts vacant or occupied.
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
        # One basic CNN block.
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Forward pass through feature extractor and classifier.
        x = self.features(x)
        return self.classifier(x)


class ParkingSlotCNN(_ParkingSlotBaseCNN):
    def __init__(self, num_classes: int = 2, dropout: float = 0.35) -> None:
        # Original custom CNN version.
        super().__init__(
            channels=(32, 64, 128, 192),
            hidden_units=96,
            dropout=dropout,
            num_classes=num_classes,
        )


class ParkingSlotCNNV2(_ParkingSlotBaseCNN):
    def __init__(self, num_classes: int = 2, dropout: float = 0.3) -> None:
        # Larger custom CNN version.
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
    # Model factory. This chooses which neural model to build.

    if model_version == "v1":
        return ParkingSlotCNN(num_classes=num_classes, dropout=dropout)

    if model_version == "v2":
        return ParkingSlotCNNV2(num_classes=num_classes, dropout=dropout)

    # Final project model uses this option.
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
    # Build MobileNetV3-Small transfer-learning model.
    try:
        from torchvision import models
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError(
            "MobileNetV3-Small needs torchvision. Install requirements.txt first."
        ) from error

    # Load MobileNetV3-Small, optionally with ImageNet pretrained weights.
    model = _load_mobilenet_v3_small(models=models, pretrained=pretrained)

    # Replace original ImageNet classifier with 2-class parking classifier.
    _replace_classifier(model=model, num_classes=num_classes, dropout=dropout)

    # Optional: freeze backbone and train only classifier.
    if freeze_backbone:
        for parameter in model.features.parameters():
            parameter.requires_grad = False

        for parameter in model.classifier.parameters():
            parameter.requires_grad = True

    return model


def _load_mobilenet_v3_small(models: Any, pretrained: bool) -> nn.Module:
    # Load MobileNetV3-Small from torchvision.
    if not pretrained:
        return models.mobilenet_v3_small(weights=None)

    # Support newer and older torchvision versions.
    try:
        weights = models.MobileNet_V3_Small_Weights.DEFAULT
        return models.mobilenet_v3_small(weights=weights)
    except AttributeError:
        return models.mobilenet_v3_small(pretrained=True)


def _replace_classifier(model: nn.Module, num_classes: int, dropout: float) -> None:
    # Replace MobileNetV3 classifier output layer for vacant/occupied prediction.
    classifier = getattr(model, "classifier", None)

    if not isinstance(classifier, nn.Sequential):
        raise ValueError("Unsupported MobileNetV3 classifier structure.")

    # Set dropout value in MobileNet classifier.
    for layer in classifier:
        if isinstance(layer, nn.Dropout):
            layer.p = dropout

    # Find the last Linear layer and replace it.
    output_layer_index = _find_last_linear_layer(classifier)
    old_output_layer = classifier[output_layer_index]

    if not isinstance(old_output_layer, nn.Linear):
        raise ValueError("MobileNetV3 output layer must be Linear.")

    classifier[output_layer_index] = nn.Linear(
        old_output_layer.in_features,
        num_classes,
    )


def _find_last_linear_layer(classifier: nn.Sequential) -> int:
    # Find final Linear layer inside MobileNetV3 classifier.
    for index in range(len(classifier) - 1, -1, -1):
        if isinstance(classifier[index], nn.Linear):
            return index

    raise ValueError("MobileNetV3 classifier has no Linear layer.")
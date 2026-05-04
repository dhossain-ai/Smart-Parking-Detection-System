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
    model_version: str = "v2",
    pretrained: bool = False,
    freeze_backbone: bool = False,
    dropout: float = 0.3,
) -> nn.Module:
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
    raise ValueError(f"Unsupported CNN model version: {model_version}")


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
            "MobileNetV3-Small requires torchvision. Install the project requirements "
            "before using --model-version mobilenet_v3_small."
        ) from error

    model = _load_torchvision_mobilenet_v3_small(models, pretrained=pretrained)
    _replace_mobilenet_classifier(model, num_classes=num_classes, dropout=dropout)

    if freeze_backbone:
        for parameter in model.parameters():
            parameter.requires_grad = False
        for parameter in model.classifier.parameters():
            parameter.requires_grad = True

    return model


def _load_torchvision_mobilenet_v3_small(models: object, pretrained: bool) -> nn.Module:
    weights = None
    if pretrained:
        try:
            weights_enum = getattr(models, "MobileNet_V3_Small_Weights")
            weights = weights_enum.DEFAULT
        except AttributeError:
            return models.mobilenet_v3_small(pretrained=True)

    try:
        return models.mobilenet_v3_small(weights=weights)
    except TypeError:
        return models.mobilenet_v3_small(pretrained=pretrained)


def _replace_mobilenet_classifier(model: nn.Module, num_classes: int, dropout: float) -> None:
    classifier = getattr(model, "classifier", None)
    if not isinstance(classifier, nn.Sequential):
        raise ValueError("Unsupported MobileNetV3 classifier layout.")

    for layer in classifier:
        if isinstance(layer, nn.Dropout):
            layer.p = dropout

    last_linear_index = None
    for index in range(len(classifier) - 1, -1, -1):
        if isinstance(classifier[index], nn.Linear):
            last_linear_index = index
            break

    if last_linear_index is None:
        raise ValueError("MobileNetV3 classifier does not contain a Linear output layer.")

    previous_layer = classifier[last_linear_index]
    if not isinstance(previous_layer, nn.Linear):
        raise ValueError("MobileNetV3 output layer is not Linear.")
    classifier[last_linear_index] = nn.Linear(previous_layer.in_features, num_classes)

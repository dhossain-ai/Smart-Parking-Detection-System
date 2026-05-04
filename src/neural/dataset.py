from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from src.classical.dataset import SlotRecord, load_manifest_records
from src.classical.features import crop_slot_image, resolve_image_path


LABEL_TO_TARGET = {"vacant": 0, "occupied": 1}
TARGET_TO_LABEL = {0: "vacant", 1: "occupied"}
NORMALIZE_STANDARD = "standard"
NORMALIZE_IMAGENET = "imagenet"
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
NORMALIZE_MODES = {NORMALIZE_STANDARD, NORMALIZE_IMAGENET}


class PKLotSlotDataset(Dataset):
    def __init__(
        self,
        manifest_path: Path,
        image_size: int = 64,
        augment: bool = False,
        max_per_class: int | None = None,
        seed: int = 42,
        allow_horizontal_flip: bool = True,
        normalize_mode: str = NORMALIZE_STANDARD,
    ) -> None:
        if normalize_mode not in NORMALIZE_MODES:
            raise ValueError(
                f"Unsupported normalize_mode: {normalize_mode}. "
                f"Expected one of: {sorted(NORMALIZE_MODES)}"
            )
        self.records = load_manifest_records(
            manifest_path,
            max_per_class=max_per_class,
            seed=seed,
        )
        self.image_size = image_size
        self.augment = augment
        self.allow_horizontal_flip = allow_horizontal_flip
        self.normalize_mode = normalize_mode
        self.rng = np.random.default_rng(seed)

        if not self.records:
            raise ValueError(f"No valid records loaded from manifest: {manifest_path}")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        record = self.records[index]
        crop = self._load_crop(record)
        if self.augment:
            crop = self._augment(crop)

        tensor = self._to_tensor(crop, normalize_mode=self.normalize_mode)
        target = torch.tensor(record.target, dtype=torch.long)
        return tensor, target

    def _load_crop(self, record: SlotRecord) -> np.ndarray:
        image_path = resolve_image_path(record.image_path)
        image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            return np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        crop = crop_slot_image(
            image_rgb,
            x1=record.x1,
            y1=record.y1,
            x2=record.x2,
            y2=record.y2,
        )
        if crop is None or crop.size == 0:
            return np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)

        return cv2.resize(
            crop,
            (self.image_size, self.image_size),
            interpolation=cv2.INTER_AREA,
        )

    def _augment(self, crop_rgb: np.ndarray) -> np.ndarray:
        crop = crop_rgb.astype(np.float32)

        contrast = float(self.rng.uniform(0.9, 1.1))
        brightness = float(self.rng.uniform(-12.0, 12.0))
        crop = crop * contrast + brightness

        if self.allow_horizontal_flip and self.rng.random() < 0.5:
            crop = np.ascontiguousarray(crop[:, ::-1, :])

        angle = float(self.rng.uniform(-5.0, 5.0))
        shift_x = float(self.rng.uniform(-0.04, 0.04) * self.image_size)
        shift_y = float(self.rng.uniform(-0.04, 0.04) * self.image_size)
        center = (self.image_size / 2.0, self.image_size / 2.0)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        matrix[0, 2] += shift_x
        matrix[1, 2] += shift_y
        crop = cv2.warpAffine(
            crop,
            matrix,
            (self.image_size, self.image_size),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )

        if self.rng.random() < 0.08:
            crop = cv2.GaussianBlur(crop, (3, 3), sigmaX=0.4)

        return np.clip(crop, 0, 255).astype(np.uint8)

    @staticmethod
    def _to_tensor(crop_rgb: np.ndarray, normalize_mode: str = NORMALIZE_STANDARD) -> torch.Tensor:
        if normalize_mode not in NORMALIZE_MODES:
            raise ValueError(
                f"Unsupported normalize_mode: {normalize_mode}. "
                f"Expected one of: {sorted(NORMALIZE_MODES)}"
            )
        array = crop_rgb.astype(np.float32) / 255.0
        if normalize_mode == NORMALIZE_IMAGENET:
            array = (array - IMAGENET_MEAN) / IMAGENET_STD
        else:
            array = (array - 0.5) / 0.5
        array = np.transpose(array, (2, 0, 1))
        return torch.from_numpy(np.ascontiguousarray(array)).float()


def count_targets(dataset: PKLotSlotDataset) -> dict[str, int]:
    counts = {"vacant": 0, "occupied": 0}
    for record in dataset.records:
        counts[record.label] = counts.get(record.label, 0) + 1
    return counts

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


class PKLotSlotDataset(Dataset):
    def __init__(
        self,
        manifest_path: Path,
        image_size: int = 64,
        augment: bool = False,
        max_per_class: int | None = None,
        seed: int = 42,
        allow_horizontal_flip: bool = True,
    ) -> None:
        self.records = load_manifest_records(
            manifest_path,
            max_per_class=max_per_class,
            seed=seed,
        )
        self.image_size = image_size
        self.augment = augment
        self.allow_horizontal_flip = allow_horizontal_flip
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

        tensor = self._to_tensor(crop)
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

        contrast = float(self.rng.uniform(0.85, 1.15))
        brightness = float(self.rng.uniform(-18.0, 18.0))
        crop = crop * contrast + brightness

        if self.allow_horizontal_flip and self.rng.random() < 0.5:
            crop = np.ascontiguousarray(crop[:, ::-1, :])

        angle = float(self.rng.uniform(-7.0, 7.0))
        shift_x = float(self.rng.uniform(-0.06, 0.06) * self.image_size)
        shift_y = float(self.rng.uniform(-0.06, 0.06) * self.image_size)
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

        return np.clip(crop, 0, 255).astype(np.uint8)

    @staticmethod
    def _to_tensor(crop_rgb: np.ndarray) -> torch.Tensor:
        array = crop_rgb.astype(np.float32) / 255.0
        array = (array - 0.5) / 0.5
        array = np.transpose(array, (2, 0, 1))
        return torch.from_numpy(np.ascontiguousarray(array)).float()


def count_targets(dataset: PKLotSlotDataset) -> dict[str, int]:
    counts = {"vacant": 0, "occupied": 0}
    for record in dataset.records:
        counts[record.label] = counts.get(record.label, 0) + 1
    return counts

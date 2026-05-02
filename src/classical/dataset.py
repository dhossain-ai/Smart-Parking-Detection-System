from __future__ import annotations

import csv
import random
from dataclasses import dataclass
from pathlib import Path

from src.utils.config import PROJECT_ROOT


LABEL_TO_TARGET = {"vacant": 0, "occupied": 1}
TARGET_TO_LABEL = {0: "vacant", 1: "occupied"}
REQUIRED_COLUMNS = {"image_path", "label", "x1", "y1", "x2", "y2"}


@dataclass(frozen=True)
class SlotRecord:
    image_path: str
    label: str
    target: int
    x1: float
    y1: float
    x2: float
    y2: float
    annotation_id: str = ""
    file_name: str = ""
    split: str = ""

    def as_feature_record(self) -> dict:
        return {
            "image_path": self.image_path,
            "x1": self.x1,
            "y1": self.y1,
            "x2": self.x2,
            "y2": self.y2,
        }


def resolve_project_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def _validate_columns(fieldnames: list[str] | None, manifest_path: Path) -> None:
    if fieldnames is None:
        raise ValueError(f"Manifest has no header: {manifest_path}")

    missing = REQUIRED_COLUMNS.difference(fieldnames)
    if missing:
        raise ValueError(
            f"Manifest is missing required columns {sorted(missing)}: {manifest_path}"
        )


def _record_from_row(row: dict[str, str]) -> SlotRecord | None:
    label = row.get("label", "").strip().lower()
    if label not in LABEL_TO_TARGET:
        return None

    try:
        x1 = float(row["x1"])
        y1 = float(row["y1"])
        x2 = float(row["x2"])
        y2 = float(row["y2"])
    except (KeyError, TypeError, ValueError):
        return None

    if x2 <= x1 or y2 <= y1:
        return None

    image_path = row.get("image_path", "").strip()
    if not image_path:
        return None

    return SlotRecord(
        image_path=image_path,
        label=label,
        target=LABEL_TO_TARGET[label],
        x1=x1,
        y1=y1,
        x2=x2,
        y2=y2,
        annotation_id=row.get("annotation_id", ""),
        file_name=row.get("file_name", ""),
        split=row.get("split", ""),
    )


def _sample_records_per_class(
    records: list[SlotRecord],
    max_per_class: int | None,
    seed: int,
) -> list[SlotRecord]:
    if max_per_class is None:
        return records

    rng = random.Random(seed)
    sampled: list[SlotRecord] = []
    for label in sorted(LABEL_TO_TARGET):
        label_records = [record for record in records if record.label == label]
        if len(label_records) > max_per_class:
            label_records = rng.sample(label_records, max_per_class)
        sampled.extend(label_records)

    rng.shuffle(sampled)
    return sampled


def load_manifest_records(
    manifest_path: Path,
    max_per_class: int | None = None,
    seed: int = 42,
) -> list[SlotRecord]:
    resolved_path = resolve_project_path(manifest_path)
    if not resolved_path.is_file():
        raise FileNotFoundError(f"Missing manifest CSV: {resolved_path}")

    records: list[SlotRecord] = []
    with resolved_path.open("r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        _validate_columns(reader.fieldnames, resolved_path)
        for row in reader:
            record = _record_from_row(row)
            if record is not None:
                records.append(record)

    return _sample_records_per_class(records, max_per_class=max_per_class, seed=seed)


def targets_from_records(records: list[SlotRecord]) -> list[int]:
    return [record.target for record in records]

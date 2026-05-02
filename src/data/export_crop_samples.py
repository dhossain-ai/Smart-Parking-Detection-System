from __future__ import annotations

import argparse
import csv
import math
import random
from collections import Counter
from pathlib import Path

from PIL import Image

from src.data.coco_utils import readable_relative_path
from src.utils.config import PROJECT_ROOT, get_pklot_dir


EXPECTED_LABELS = {"occupied", "vacant"}
DEFAULT_SAMPLE_OUTPUT_DIR = Path("data/processed/crop_samples")
DEFAULT_FULL_OUTPUT_DIR = Path("data/processed/crops")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export limited parking-slot crop samples for visual QA."
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=Path("data/processed/metadata/slot_annotations.csv"),
        help="Path to slot annotation metadata CSV.",
    )
    parser.add_argument(
        "--samples-per-class",
        type=int,
        default=20,
        help="Number of crops per split/class in sample mode.",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=64,
        help="Output crop size in pixels.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to crop samples, or full crops with --export-all.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sample selection.",
    )
    parser.add_argument(
        "--export-all",
        action="store_true",
        help="Export every crop to data/processed/crops unless --output-dir is set.",
    )
    return parser.parse_args()


def _resolve_project_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def _resolve_image_path(image_path_value: str) -> Path:
    image_path = Path(image_path_value)
    if image_path.is_absolute():
        return image_path

    dataset_path = get_pklot_dir() / image_path
    if dataset_path.is_file():
        return dataset_path

    return PROJECT_ROOT / image_path


def _safe_int_bounds(row: dict[str, str], image: Image.Image) -> tuple[int, int, int, int] | None:
    try:
        x1 = math.floor(float(row["x1"]))
        y1 = math.floor(float(row["y1"]))
        x2 = math.ceil(float(row["x2"]))
        y2 = math.ceil(float(row["y2"]))
    except (KeyError, TypeError, ValueError):
        return None

    width, height = image.size
    x1 = max(0, min(width, x1))
    y1 = max(0, min(height, y1))
    x2 = max(0, min(width, x2))
    y2 = max(0, min(height, y2))

    if x2 <= x1 or y2 <= y1:
        return None

    return x1, y1, x2, y2


def _crop_filename(row: dict[str, str], sequence: int) -> str:
    annotation_id = row.get("annotation_id") or str(sequence)
    image_id = row.get("image_id") or "image"
    return f"{row['split']}_{row['label']}_img{image_id}_ann{annotation_id}.jpg"


class CropWriter:
    def __init__(self, output_dir: Path, size: int) -> None:
        self.output_dir = output_dir
        self.size = size
        self._cached_path: Path | None = None
        self._cached_image: Image.Image | None = None

    def _open_image(self, image_path: Path) -> Image.Image:
        if self._cached_path == image_path and self._cached_image is not None:
            return self._cached_image

        if self._cached_image is not None:
            self._cached_image.close()

        self._cached_path = image_path
        self._cached_image = Image.open(image_path).convert("RGB")
        return self._cached_image

    def close(self) -> None:
        if self._cached_image is not None:
            self._cached_image.close()
        self._cached_path = None
        self._cached_image = None

    def save_crop(self, row: dict[str, str], sequence: int) -> bool:
        label = row.get("label", "")
        split = row.get("split", "")
        if label not in EXPECTED_LABELS or not split:
            return False

        image_path = _resolve_image_path(row.get("image_path", ""))
        if not image_path.is_file():
            return False

        image = self._open_image(image_path)
        bounds = _safe_int_bounds(row, image)
        if bounds is None:
            return False

        destination_dir = self.output_dir / split / label
        destination_dir.mkdir(parents=True, exist_ok=True)
        destination_path = destination_dir / _crop_filename(row, sequence)

        crop = image.crop(bounds).resize((self.size, self.size), Image.Resampling.BILINEAR)
        crop.save(destination_path, quality=95)
        return True


def _sample_rows(
    metadata_path: Path,
    samples_per_class: int,
    seed: int,
) -> list[dict[str, str]]:
    rng = random.Random(seed)
    reservoirs: dict[tuple[str, str], list[dict[str, str]]] = {}
    seen: Counter[tuple[str, str]] = Counter()

    with metadata_path.open("r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            split = row.get("split", "")
            label = row.get("label", "")
            if label not in EXPECTED_LABELS or not split:
                continue

            key = (split, label)
            seen[key] += 1
            bucket = reservoirs.setdefault(key, [])

            if len(bucket) < samples_per_class:
                bucket.append(row)
                continue

            replacement_index = rng.randint(0, seen[key] - 1)
            if replacement_index < samples_per_class:
                bucket[replacement_index] = row

    selected_rows: list[dict[str, str]] = []
    for key in sorted(reservoirs):
        selected_rows.extend(reservoirs[key])
    return selected_rows


def export_crops(
    metadata: Path,
    samples_per_class: int,
    size: int,
    output_dir: Path | None,
    seed: int,
    export_all: bool,
) -> int:
    metadata_path = _resolve_project_path(metadata)
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Missing metadata CSV: {metadata_path}")

    default_output = DEFAULT_FULL_OUTPUT_DIR if export_all else DEFAULT_SAMPLE_OUTPUT_DIR
    output_path = _resolve_project_path(output_dir or default_output)
    output_path.mkdir(parents=True, exist_ok=True)

    writer = CropWriter(output_path, size=size)
    saved_counts: Counter[tuple[str, str]] = Counter()
    skipped = 0
    sequence = 0

    try:
        if export_all:
            with metadata_path.open("r", newline="", encoding="utf-8") as file:
                reader = csv.DictReader(file)
                for row in reader:
                    sequence += 1
                    if writer.save_crop(row, sequence):
                        saved_counts[(row["split"], row["label"])] += 1
                    else:
                        skipped += 1
        else:
            for row in _sample_rows(metadata_path, samples_per_class, seed):
                sequence += 1
                if writer.save_crop(row, sequence):
                    saved_counts[(row["split"], row["label"])] += 1
                else:
                    skipped += 1
    finally:
        writer.close()

    mode = "full export" if export_all else "sample export"
    print(f"Crop {mode} complete")
    print(f"Output directory: {readable_relative_path(output_path, PROJECT_ROOT)}")
    for split, label in sorted(saved_counts):
        print(f"{split}/{label}: {saved_counts[(split, label)]}")
    print(f"Skipped crops: {skipped}")

    return 0


def main() -> int:
    args = _parse_args()
    return export_crops(
        metadata=args.metadata,
        samples_per_class=args.samples_per_class,
        size=args.size,
        output_dir=args.output_dir,
        seed=args.seed,
        export_all=args.export_all,
    )


if __name__ == "__main__":
    raise SystemExit(main())

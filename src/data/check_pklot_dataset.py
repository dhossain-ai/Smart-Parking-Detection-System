from __future__ import annotations

import os
from pathlib import Path

from src.utils.config import DEFAULT_PKLOT_DIR, get_pklot_dir


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
XML_EXTENSION = ".xml"
COCO_ANNOTATION_NAME = "_annotations.coco.json"
SPLIT_NAMES = ("train", "valid", "test")
SAMPLE_LIMIT = 5


def _scan_dataset(dataset_dir: Path) -> tuple[int, int, int, list[Path], list[Path]]:
    image_count = 0
    xml_count = 0
    coco_count = 0
    image_samples: list[Path] = []
    coco_samples: list[Path] = []

    for root, _, files in os.walk(dataset_dir):
        root_path = Path(root)

        for filename in files:
            path = root_path / filename
            suffix = path.suffix.lower()

            if suffix in IMAGE_EXTENSIONS:
                image_count += 1
                if len(image_samples) < SAMPLE_LIMIT:
                    image_samples.append(path)
            elif suffix == XML_EXTENSION:
                xml_count += 1

            if filename == COCO_ANNOTATION_NAME:
                coco_count += 1
                if len(coco_samples) < SAMPLE_LIMIT:
                    coco_samples.append(path)

    return image_count, xml_count, coco_count, image_samples, coco_samples


def _detect_split_folders(dataset_dir: Path) -> list[str]:
    return [split for split in SPLIT_NAMES if (dataset_dir / split).is_dir()]


def _detect_dataset_format(xml_count: int, coco_count: int) -> str:
    if xml_count and coco_count:
        return "mixed"
    if xml_count:
        return "xml"
    if coco_count:
        return "coco"
    return "unknown"


def _format_sample_paths(paths: list[Path], dataset_dir: Path) -> list[str]:
    samples: list[str] = []

    for path in paths[:SAMPLE_LIMIT]:
        try:
            samples.append(str(path.relative_to(dataset_dir)))
        except ValueError:
            samples.append(str(path))

    return samples


def _print_samples(title: str, samples: list[str]) -> None:
    print(title)
    if not samples:
        print("- None found")
        return

    for sample in samples:
        print(f"- {sample}")


def _print_missing_instructions(dataset_dir: Path) -> None:
    print("Dataset status: missing")
    print("Download the PKLot Dataset manually from Kaggle, then either:")
    print(f"- unzip it into {DEFAULT_PKLOT_DIR}")
    print("- or set PKLOT_DATA_DIR=/absolute/path/to/PKLot")
    print("Do not commit the full dataset or Kaggle credentials.")
    print(f"Resolved dataset path: {dataset_dir}")


def main() -> int:
    dataset_dir = get_pklot_dir()

    print("PKLot dataset check")
    print(f"Dataset path: {dataset_dir}")
    print(f"Exists: {'Yes' if dataset_dir.exists() else 'No'}")

    if not dataset_dir.exists():
        _print_missing_instructions(dataset_dir)
        return 1

    image_count, xml_count, coco_count, image_samples, coco_samples = _scan_dataset(dataset_dir)
    split_folders = _detect_split_folders(dataset_dir)
    dataset_format = _detect_dataset_format(xml_count, coco_count)

    print(f"Images found: {image_count}")
    print(f"XML files found: {xml_count}")
    print(f"COCO annotation JSON files found: {coco_count}")
    print("Splits detected: " + (", ".join(split_folders) if split_folders else "None"))
    print(f"Dataset format: {dataset_format}")
    _print_samples("Sample image paths:", _format_sample_paths(image_samples, dataset_dir))
    _print_samples("Sample COCO JSON paths:", _format_sample_paths(coco_samples, dataset_dir))

    if not image_count or not (xml_count or coco_count):
        print("Dataset status: incomplete")
        print("Expected image files and XML or COCO annotation files.")
        print("No annotation parsing or crop generation was performed.")
        return 2

    print("Dataset status: OK")
    print("No annotation parsing or crop generation was performed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

# os is used to walk through folders and files.
import os

# Path is used to handle file and folder paths safely.
from pathlib import Path

# Import default dataset path and function that finds the real PKLot path.
from src.utils.config import DEFAULT_PKLOT_DIR, get_pklot_dir


# Image file types that this script will count.
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

# XML annotation file extension.
XML_EXTENSION = ".xml"

# COCO annotation file name used by Roboflow export.
COCO_ANNOTATION_NAME = "_annotations.coco.json"

# Expected dataset split folder names.
SPLIT_NAMES = ("train", "valid", "test")

# Maximum number of sample paths to print.
SAMPLE_LIMIT = 5


def _scan_dataset(dataset_dir: Path) -> tuple[int, int, int, list[Path], list[Path]]:
    """
    Scan the dataset folder.

    This function counts:
    - image files
    - XML annotation files
    - COCO annotation JSON files

    It also saves a few sample image paths and COCO JSON paths.
    """

    # Counters for images, XML files, and COCO JSON files.
    image_count = 0
    xml_count = 0
    coco_count = 0

    # Store a few example image paths.
    image_samples: list[Path] = []

    # Store a few example COCO annotation JSON paths.
    coco_samples: list[Path] = []

    # Walk through all folders and files inside the dataset directory.
    for root, _, files in os.walk(dataset_dir):
        root_path = Path(root)

        # Check every file in the current folder.
        for filename in files:
            path = root_path / filename
            suffix = path.suffix.lower()

            # If the file is an image, count it.
            if suffix in IMAGE_EXTENSIONS:
                image_count += 1

                # Save only a few image examples.
                if len(image_samples) < SAMPLE_LIMIT:
                    image_samples.append(path)

            # If the file is XML annotation, count it.
            elif suffix == XML_EXTENSION:
                xml_count += 1

            # If the file is a COCO annotation JSON, count it.
            if filename == COCO_ANNOTATION_NAME:
                coco_count += 1

                # Save only a few COCO JSON examples.
                if len(coco_samples) < SAMPLE_LIMIT:
                    coco_samples.append(path)

    # Return all counts and sample paths.
    return image_count, xml_count, coco_count, image_samples, coco_samples


def _detect_split_folders(dataset_dir: Path) -> list[str]:
    """
    Check which dataset split folders exist.

    Expected folders:
    - train
    - valid
    - test
    """

    # Return only the split names that exist as folders.
    return [split for split in SPLIT_NAMES if (dataset_dir / split).is_dir()]


def _detect_dataset_format(xml_count: int, coco_count: int) -> str:
    """
    Detect dataset annotation format.

    Possible results:
    - mixed: both XML and COCO annotations exist
    - xml: only XML annotations exist
    - coco: only COCO annotations exist
    - unknown: no annotation files found
    """

    # Dataset has both XML and COCO annotations.
    if xml_count and coco_count:
        return "mixed"

    # Dataset has only XML annotations.
    if xml_count:
        return "xml"

    # Dataset has only COCO annotations.
    if coco_count:
        return "coco"

    # No known annotation format was found.
    return "unknown"


def _format_sample_paths(paths: list[Path], dataset_dir: Path) -> list[str]:
    """
    Convert sample file paths to readable strings.

    If possible, show paths relative to the dataset folder.
    This makes the output shorter and easier to read.
    """

    samples: list[str] = []

    # Format only a limited number of sample paths.
    for path in paths[:SAMPLE_LIMIT]:
        try:
            # Try to make the path relative to dataset_dir.
            samples.append(str(path.relative_to(dataset_dir)))

        except ValueError:
            # If relative path fails, use the full path.
            samples.append(str(path))

    return samples


def _print_samples(title: str, samples: list[str]) -> None:
    """
    Print sample file paths.

    If there are no samples, print 'None found'.
    """

    print(title)

    # If no samples exist, show a simple message.
    if not samples:
        print("- None found")
        return

    # Print each sample path.
    for sample in samples:
        print(f"- {sample}")


def _print_missing_instructions(dataset_dir: Path) -> None:
    """
    Print instructions when the dataset folder is missing.

    This tells the user where to place the PKLot dataset.
    """

    print("Dataset status: missing")
    print("Download the PKLot Dataset manually from Kaggle, then either:")
    print(f"- unzip it into {DEFAULT_PKLOT_DIR}")
    print("- or set PKLOT_DATA_DIR=/absolute/path/to/PKLot")
    print("Do not commit the full dataset or Kaggle credentials.")
    print(f"Resolved dataset path: {dataset_dir}")


def main() -> int:
    """
    Main function for checking the PKLot dataset.

    This function:
    1. Finds the dataset path
    2. Checks if the folder exists
    3. Counts images and annotations
    4. Detects split folders
    5. Prints dataset status
    """

    # Get the PKLot dataset path from config or environment variable.
    dataset_dir = get_pklot_dir()

    # Print basic dataset path information.
    print("PKLot dataset check")
    print(f"Dataset path: {dataset_dir}")
    print(f"Exists: {'Yes' if dataset_dir.exists() else 'No'}")

    # If the dataset folder does not exist, print instructions and stop.
    if not dataset_dir.exists():
        _print_missing_instructions(dataset_dir)
        return 1

    # Scan the dataset folder and count files.
    image_count, xml_count, coco_count, image_samples, coco_samples = _scan_dataset(dataset_dir)

    # Detect train/valid/test folders.
    split_folders = _detect_split_folders(dataset_dir)

    # Detect annotation format: XML, COCO, mixed, or unknown.
    dataset_format = _detect_dataset_format(xml_count, coco_count)

    # Print dataset summary.
    print(f"Images found: {image_count}")
    print(f"XML files found: {xml_count}")
    print(f"COCO annotation JSON files found: {coco_count}")
    print("Splits detected: " + (", ".join(split_folders) if split_folders else "None"))
    print(f"Dataset format: {dataset_format}")

    # Print a few example image paths.
    _print_samples("Sample image paths:", _format_sample_paths(image_samples, dataset_dir))

    # Print a few example COCO JSON paths.
    _print_samples("Sample COCO JSON paths:", _format_sample_paths(coco_samples, dataset_dir))

    # Dataset is incomplete if there are no images or no annotations.
    if not image_count or not (xml_count or coco_count):
        print("Dataset status: incomplete")
        print("Expected image files and XML or COCO annotation files.")
        print("No annotation parsing or crop generation was performed.")
        return 2

    # Dataset exists and has images plus annotations.
    print("Dataset status: OK")

    # This script only checks files. It does not create crops or metadata.
    print("No annotation parsing or crop generation was performed.")
    return 0


# Run main() only when this file is executed directly.
# Example:
# python -m src.data.check_pklot_dataset
if __name__ == "__main__":
    raise SystemExit(main())
from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

from src.data.coco_utils import (
    clip_coco_bbox,
    index_categories_by_id,
    index_images_by_id,
    load_coco_json,
    normalize_category_name,
    readable_relative_path,
)
from src.utils.config import PROJECT_ROOT, get_pklot_dir


# Dataset split folders expected in the PKLot COCO export.
DEFAULT_SPLITS = ("train", "valid", "test")

# COCO annotation filename inside each split folder.
ANNOTATION_FILENAME = "_annotations.coco.json"

# Columns written to the main slot metadata CSV.
SLOT_FIELDS = [
    "split",
    "image_id",
    "annotation_id",
    "file_name",
    "image_path",
    "image_width",
    "image_height",
    "category_id",
    "category_name",
    "label",
    "x",
    "y",
    "width",
    "height",
    "x1",
    "y1",
    "x2",
    "y2",
    "area",
]

# Columns for label count summary by split.
SPLIT_SUMMARY_FIELDS = ["split", "label", "count"]

# Columns for original COCO category summary.
CATEGORY_SUMMARY_FIELDS = [
    "split",
    "category_id",
    "category_name",
    "normalized_label",
    "count",
]


def _parse_args() -> argparse.Namespace:
    # Read command-line options for preprocessing.
    parser = argparse.ArgumentParser(
        description="Prepare PKLot parking-slot metadata from Roboflow COCO annotations."
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=list(DEFAULT_SPLITS),
        help="Dataset splits to process.",
    )
    parser.add_argument(
        "--limit-images",
        type=int,
        default=None,
        help="Limit images per split for smoke tests.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/metadata"),
        help="Directory for metadata CSV outputs.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print skipped annotation details.",
    )
    return parser.parse_args()


def _format_number(value: float) -> str:
    # Format float values cleanly before writing to CSV.
    return f"{value:.6f}".rstrip("0").rstrip(".")


def _image_path_for_row(dataset_dir: Path, split: str, file_name: str) -> str:
    # Build image path for one metadata row.
    image_path = dataset_dir / split / file_name
    try:
        return readable_relative_path(image_path, dataset_dir)
    except ValueError:
        return f"{split}/{file_name}"


def _selected_image_ids(
    images_by_id: dict[int, dict],
    limit_images: int | None,
) -> set[int]:
    # Select all images, or only a small number for smoke testing.
    ordered_ids = list(images_by_id.keys())
    if limit_images is not None:
        ordered_ids = ordered_ids[: max(0, limit_images)]
    return set(ordered_ids)


def _write_split_summary(path: Path, split_label_counts: Counter[tuple[str, str]]) -> None:
    # Save how many occupied/vacant slots exist in each split.
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=SPLIT_SUMMARY_FIELDS)
        writer.writeheader()
        for split, label in sorted(split_label_counts):
            writer.writerow(
                {
                    "split": split,
                    "label": label,
                    "count": split_label_counts[(split, label)],
                }
            )


def _write_category_summary(
    path: Path,
    category_counts: Counter[tuple[str, int, str, str]],
) -> None:
    # Save original COCO category names and their normalized labels.
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=CATEGORY_SUMMARY_FIELDS)
        writer.writeheader()
        for split, category_id, category_name, normalized_label in sorted(category_counts):
            writer.writerow(
                {
                    "split": split,
                    "category_id": category_id,
                    "category_name": category_name,
                    "normalized_label": normalized_label,
                    "count": category_counts[
                        (split, category_id, category_name, normalized_label)
                    ],
                }
            )


def _print_split_report(
    split: str,
    images_count: int,
    annotations_count: int,
    valid_slots: int,
    skipped_invalid_boxes: int,
    skipped_unknown_labels: int,
    label_counts: Counter[str],
) -> None:
    # Print a short preprocessing summary for one split.
    print(f"Split: {split}")
    print(f"Images: {images_count}")
    print(f"Annotations: {annotations_count}")
    print(f"Valid slots: {valid_slots}")
    print(f"Skipped invalid boxes: {skipped_invalid_boxes}")
    if skipped_unknown_labels:
        print(f"Skipped unknown labels: {skipped_unknown_labels}")
    print(f"Occupied: {label_counts.get('occupied', 0)}")
    print(f"Vacant: {label_counts.get('vacant', 0)}")


def prepare_metadata(
    splits: list[str],
    limit_images: int | None,
    output_dir: Path,
    verbose: bool = False,
) -> int:
    # Main preprocessing workflow.
    # It reads COCO annotations and writes slot_annotations.csv.
    dataset_dir = get_pklot_dir()
    output_dir = output_dir if output_dir.is_absolute() else PROJECT_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    slot_annotations_path = output_dir / "slot_annotations.csv"
    split_summary_path = output_dir / "split_summary.csv"
    category_summary_path = output_dir / "category_summary.csv"

    # Counters used for final reports.
    split_label_counts: Counter[tuple[str, str]] = Counter()
    category_counts: Counter[tuple[str, int, str, str]] = Counter()
    category_names_detected: set[str] = set()
    total_valid_slots = 0
    total_invalid_boxes = 0
    total_unknown_labels = 0
    total_images = 0

    # Open the main output CSV and write one row per valid parking slot.
    with slot_annotations_path.open("w", newline="", encoding="utf-8") as slot_file:
        slot_writer = csv.DictWriter(slot_file, fieldnames=SLOT_FIELDS)
        slot_writer.writeheader()

        # Process train, valid, and test splits.
        for split in splits:
            annotation_path = dataset_dir / split / ANNOTATION_FILENAME
            if not annotation_path.is_file():
                raise FileNotFoundError(f"Missing COCO annotation file: {annotation_path}")

            # Load COCO data and build lookup dictionaries.
            coco_data = load_coco_json(annotation_path)
            images_by_id = index_images_by_id(coco_data)
            categories_by_id = index_categories_by_id(coco_data)
            selected_image_ids = _selected_image_ids(images_by_id, limit_images)

            annotations_count = 0
            valid_slots = 0
            skipped_invalid_boxes = 0
            skipped_unknown_labels = 0
            split_label_counter: Counter[str] = Counter()

            # Read every annotation box from the COCO file.
            for annotation in coco_data.get("annotations", []):
                image_id = int(annotation.get("image_id", -1))

                # Skip annotations from images not selected for this run.
                if image_id not in selected_image_ids:
                    continue

                annotations_count += 1
                image = images_by_id.get(image_id)

                # Skip annotation if its image is missing.
                if image is None:
                    skipped_invalid_boxes += 1
                    if verbose:
                        print(f"Skipping annotation with missing image: {annotation}")
                    continue

                # Read category and convert it to occupied/vacant.
                category_id = int(annotation.get("category_id", -1))
                category = categories_by_id.get(category_id, {})
                category_name = str(category.get("name", ""))
                label = normalize_category_name(category_name)
                category_names_detected.add(category_name)

                # Skip categories that are not occupied or vacant.
                if label is None:
                    skipped_unknown_labels += 1
                    if verbose:
                        print(
                            "Skipping annotation with unknown category "
                            f"{category_id}: {category_name}"
                        )
                    continue

                # Read image size and clip bbox inside image boundaries.
                image_width = int(image.get("width", 0))
                image_height = int(image.get("height", 0))
                clipped_bbox = clip_coco_bbox(
                    annotation.get("bbox", []),
                    image_width=image_width,
                    image_height=image_height,
                )

                # Skip invalid bounding boxes.
                if clipped_bbox is None:
                    skipped_invalid_boxes += 1
                    if verbose:
                        print(f"Skipping invalid bbox: {annotation}")
                    continue

                # Build one clean metadata row for this parking slot.
                file_name = str(image.get("file_name", ""))
                row = {
                    "split": split,
                    "image_id": image_id,
                    "annotation_id": annotation.get("id", ""),
                    "file_name": file_name,
                    "image_path": _image_path_for_row(dataset_dir, split, file_name),
                    "image_width": image_width,
                    "image_height": image_height,
                    "category_id": category_id,
                    "category_name": category_name,
                    "label": label,
                    **{
                        key: _format_number(value)
                        for key, value in clipped_bbox.items()
                    },
                }
                slot_writer.writerow(row)

                # Update counters after writing a valid slot.
                valid_slots += 1
                total_valid_slots += 1
                split_label_counter[label] += 1
                split_label_counts[(split, label)] += 1
                category_counts[(split, category_id, category_name, label)] += 1

            # Update total counters and print split summary.
            images_count = len(selected_image_ids)
            total_images += images_count
            total_invalid_boxes += skipped_invalid_boxes
            total_unknown_labels += skipped_unknown_labels

            _print_split_report(
                split=split,
                images_count=images_count,
                annotations_count=annotations_count,
                valid_slots=valid_slots,
                skipped_invalid_boxes=skipped_invalid_boxes,
                skipped_unknown_labels=skipped_unknown_labels,
                label_counts=split_label_counter,
            )

    # Save summary CSV files after all splits are processed.
    _write_split_summary(split_summary_path, split_label_counts)
    _write_category_summary(category_summary_path, category_counts)

    # Print output file locations and overall summary.
    print("Metadata outputs:")
    print(f"- {readable_relative_path(slot_annotations_path, PROJECT_ROOT)}")
    print(f"- {readable_relative_path(split_summary_path, PROJECT_ROOT)}")
    print(f"- {readable_relative_path(category_summary_path, PROJECT_ROOT)}")
    print("Overall:")
    print(f"Images processed: {total_images}")
    print(f"Valid slots: {total_valid_slots}")
    print(f"Skipped invalid boxes: {total_invalid_boxes}")
    if total_unknown_labels:
        print(f"Skipped unknown labels: {total_unknown_labels}")
    print(
        "Category names detected: "
        + (", ".join(sorted(category_names_detected)) if category_names_detected else "None")
    )

    return 0


def main() -> int:
    # Parse command-line arguments and start preprocessing.
    args = _parse_args()
    return prepare_metadata(
        splits=args.splits,
        limit_images=args.limit_images,
        output_dir=args.output_dir,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    raise SystemExit(main())
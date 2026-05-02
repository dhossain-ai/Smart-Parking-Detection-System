from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.data.coco_utils import readable_relative_path
from src.utils.config import METRICS_DIR, PROJECT_ROOT, get_pklot_dir


EXPECTED_SPLITS = {"train", "valid", "test"}
EXPECTED_LABELS = {"occupied", "vacant"}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate PKLot slot metadata splits.")
    parser.add_argument(
        "--metadata",
        type=Path,
        default=Path("data/processed/metadata/slot_annotations.csv"),
        help="Slot annotation metadata CSV.",
    )
    parser.add_argument(
        "--check-images",
        action="store_true",
        help="Check whether referenced image files exist.",
    )
    parser.add_argument(
        "--max-image-checks",
        type=int,
        default=500,
        help="Maximum image paths to check when --check-images is enabled.",
    )
    return parser.parse_args()


def _resolve_project_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def _resolve_image_path(image_path_value: str) -> Path:
    image_path = Path(str(image_path_value))
    if image_path.is_absolute():
        return image_path

    dataset_path = get_pklot_dir() / image_path
    if dataset_path.exists():
        return dataset_path

    return PROJECT_ROOT / image_path


def _load_metadata(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Missing metadata CSV: {path}")

    df = pd.read_csv(path)
    required = {
        "split",
        "image_path",
        "annotation_id",
        "label",
        "width",
        "height",
    }
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Metadata is missing required columns: {sorted(missing)}")

    df["width"] = pd.to_numeric(df["width"], errors="coerce")
    df["height"] = pd.to_numeric(df["height"], errors="coerce")
    return df


def _check_image_paths(
    df: pd.DataFrame,
    enabled: bool,
    max_checks: int,
) -> tuple[list[str], int]:
    if not enabled:
        return [], 0

    unique_paths = sorted(str(path) for path in df["image_path"].dropna().unique())
    checked_paths = unique_paths[: max(0, max_checks)]
    missing_paths = [
        path for path in checked_paths if not _resolve_image_path(path).is_file()
    ]
    return missing_paths, len(checked_paths)


def _write_report(
    report_path: Path,
    df: pd.DataFrame,
    errors: list[str],
    warnings: list[str],
    image_checks: int,
    missing_image_paths: list[str],
) -> None:
    split_counts = df.groupby("split").size().to_dict()
    label_counts = df.groupby("label").size().to_dict()
    split_label_counts = df.groupby(["split", "label"]).size().to_dict()

    lines = [
        "# Split Validation Report",
        "",
        f"- Status: {'Passed' if not errors else 'Failed'}",
        f"- Total slot annotations: {len(df)}",
        f"- Image path checks performed: {image_checks}",
        "",
        "## Split Counts",
        "",
    ]

    for split in sorted(split_counts):
        lines.append(f"- {split}: {split_counts[split]}")

    lines.extend(["", "## Label Counts", ""])
    for label in sorted(label_counts):
        lines.append(f"- {label}: {label_counts[label]}")

    lines.extend(["", "## Split Label Counts", ""])
    for split, label in sorted(split_label_counts):
        lines.append(f"- {split} / {label}: {split_label_counts[(split, label)]}")

    lines.extend(["", "## Errors", ""])
    if errors:
        lines.extend(f"- {error}" for error in errors)
    else:
        lines.append("- None")

    lines.extend(["", "## Warnings", ""])
    if warnings:
        lines.extend(f"- {warning}" for warning in warnings)
    else:
        lines.append("- None")

    if missing_image_paths:
        lines.extend(["", "## Missing Checked Image Paths", ""])
        lines.extend(f"- {path}" for path in missing_image_paths[:50])

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def validate_splits(metadata: Path, check_images: bool, max_image_checks: int) -> int:
    metadata_path = _resolve_project_path(metadata)
    output_dir = METRICS_DIR / "dataset_eda"
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "split_validation_report.md"

    df = _load_metadata(metadata_path)
    errors: list[str] = []
    warnings: list[str] = []

    present_splits = set(df["split"].dropna().unique())
    missing_splits = sorted(EXPECTED_SPLITS.difference(present_splits))
    unexpected_splits = sorted(present_splits.difference(EXPECTED_SPLITS))
    if missing_splits:
        errors.append(f"Missing expected splits: {', '.join(missing_splits)}")
    if unexpected_splits:
        errors.append(f"Unexpected splits found: {', '.join(unexpected_splits)}")

    present_labels = set(df["label"].dropna().unique())
    unexpected_labels = sorted(present_labels.difference(EXPECTED_LABELS))
    if unexpected_labels:
        errors.append(f"Unexpected labels found: {', '.join(unexpected_labels)}")

    for split in sorted(EXPECTED_SPLITS.intersection(present_splits)):
        split_labels = set(df.loc[df["split"] == split, "label"].dropna().unique())
        missing_labels = sorted(EXPECTED_LABELS.difference(split_labels))
        if missing_labels:
            errors.append(
                f"Split {split} is missing labels: {', '.join(missing_labels)}"
            )

    duplicate_count = int(
        df.duplicated(subset=["split", "annotation_id"], keep=False).sum()
    )
    if duplicate_count:
        errors.append(
            f"Duplicate annotation IDs within the same split: {duplicate_count} rows"
        )

    invalid_box_count = int(((df["width"] <= 0) | (df["height"] <= 0)).sum())
    invalid_box_count += int((df["width"].isna() | df["height"].isna()).sum())
    if invalid_box_count:
        errors.append(f"Invalid non-positive or missing boxes: {invalid_box_count}")

    missing_image_paths, image_checks = _check_image_paths(
        df=df,
        enabled=check_images,
        max_checks=max_image_checks,
    )
    if missing_image_paths:
        errors.append(
            f"Missing checked image paths: {len(missing_image_paths)} of {image_checks}"
        )
    if not check_images:
        warnings.append("Image existence checks were not run.")

    _write_report(
        report_path=report_path,
        df=df,
        errors=errors,
        warnings=warnings,
        image_checks=image_checks,
        missing_image_paths=missing_image_paths,
    )

    print("Split validation")
    print(f"Metadata rows: {len(df)}")
    print(f"Splits present: {', '.join(sorted(present_splits))}")
    print(f"Labels present: {', '.join(sorted(present_labels))}")
    print(f"Image paths checked: {image_checks}")
    print(f"Report: {readable_relative_path(report_path, PROJECT_ROOT)}")
    print(f"Status: {'FAILED' if errors else 'PASSED'}")

    if errors:
        for error in errors:
            print(f"- {error}")
        return 1

    return 0


def main() -> int:
    args = _parse_args()
    return validate_splits(
        metadata=args.metadata,
        check_images=args.check_images,
        max_image_checks=args.max_image_checks,
    )


if __name__ == "__main__":
    raise SystemExit(main())

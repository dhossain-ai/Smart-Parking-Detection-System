from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.data.coco_utils import readable_relative_path
from src.utils.config import PROJECT_ROOT


EXPECTED_SPLITS = ("train", "valid", "test")
EXPECTED_LABELS = ("occupied", "vacant")
MANIFEST_COLUMNS = [
    "split",
    "image_path",
    "file_name",
    "annotation_id",
    "label",
    "x1",
    "y1",
    "x2",
    "y2",
    "width",
    "height",
]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create PKLot training manifest CSVs from slot metadata."
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=Path("data/processed/metadata/slot_annotations.csv"),
        help="Slot annotation metadata CSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/splits"),
        help="Directory for generated split manifest CSVs.",
    )
    parser.add_argument(
        "--samples-per-class",
        type=int,
        default=2000,
        help="Samples per label for balanced small manifests.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for balanced small manifests.",
    )
    return parser.parse_args()


def _resolve_project_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def _load_manifest_columns(metadata_path: Path) -> pd.DataFrame:
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Missing metadata CSV: {metadata_path}")

    df = pd.read_csv(metadata_path, usecols=MANIFEST_COLUMNS)
    missing = set(MANIFEST_COLUMNS).difference(df.columns)
    if missing:
        raise ValueError(f"Metadata is missing manifest columns: {sorted(missing)}")

    return df


def _balanced_subset(
    split_df: pd.DataFrame,
    samples_per_class: int,
    seed: int,
) -> pd.DataFrame:
    groups: list[pd.DataFrame] = []

    for label in EXPECTED_LABELS:
        label_df = split_df.loc[split_df["label"] == label]
        if label_df.empty:
            continue

        sample_size = min(samples_per_class, len(label_df))
        groups.append(label_df.sample(n=sample_size, random_state=seed))

    if not groups:
        return split_df.iloc[0:0].copy()

    return (
        pd.concat(groups, ignore_index=True)
        .sample(frac=1.0, random_state=seed)
        .reset_index(drop=True)
    )


def create_training_manifests(
    metadata: Path,
    output_dir: Path,
    samples_per_class: int,
    seed: int,
) -> int:
    metadata_path = _resolve_project_path(metadata)
    output_path = _resolve_project_path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    df = _load_manifest_columns(metadata_path)

    print("Training manifest generation")
    print(f"Metadata rows: {len(df)}")
    print(f"Output directory: {readable_relative_path(output_path, PROJECT_ROOT)}")

    for split in EXPECTED_SPLITS:
        split_df = df.loc[df["split"] == split].copy()
        manifest_path = output_path / f"{split}_slots.csv"
        split_df.to_csv(manifest_path, index=False)

        balanced_df = _balanced_subset(split_df, samples_per_class, seed)
        balanced_path = output_path / f"{split}_slots_balanced_small.csv"
        balanced_df.to_csv(balanced_path, index=False)

        label_counts = split_df["label"].value_counts().to_dict()
        balanced_counts = balanced_df["label"].value_counts().to_dict()
        print(
            f"{split}: full={len(split_df)} "
            f"occupied={label_counts.get('occupied', 0)} "
            f"vacant={label_counts.get('vacant', 0)}"
        )
        print(
            f"{split}: balanced_small={len(balanced_df)} "
            f"occupied={balanced_counts.get('occupied', 0)} "
            f"vacant={balanced_counts.get('vacant', 0)}"
        )

    return 0


def main() -> int:
    args = _parse_args()
    return create_training_manifests(
        metadata=args.metadata,
        output_dir=args.output_dir,
        samples_per_class=args.samples_per_class,
        seed=args.seed,
    )


if __name__ == "__main__":
    raise SystemExit(main())

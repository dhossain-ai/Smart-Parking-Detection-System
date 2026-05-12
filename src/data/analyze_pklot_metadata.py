"""This script performs dataset EDA. It loads the generated slot metadata,
 checks required columns, summarizes class balance, train/validation/test splits,
   image counts, and bounding-box sizes, then saves CSV summaries, plots, and 
   a markdown report."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

# Use non-GUI backend so plots can be saved without opening a window.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from src.data.coco_utils import readable_relative_path
from src.utils.config import FIGURES_DIR, METRICS_DIR, PROJECT_ROOT


# Columns that must exist in slot_annotations.csv.
REQUIRED_COLUMNS = {
    "split",
    "image_id",
    "annotation_id",
    "file_name",
    "image_path",
    "image_width",
    "image_height",
    "label",
    "width",
    "height",
    "area",
}


def _parse_args() -> argparse.Namespace:
    # Read command-line options for metadata input and output folders.
    parser = argparse.ArgumentParser(
        description="Analyze PKLot slot metadata and create dataset EDA outputs."
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
        default=Path("results/metrics/dataset_eda"),
        help="Directory for EDA CSV summaries and report.",
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=Path("results/figures"),
        help="Directory for EDA figures.",
    )
    return parser.parse_args()


def _resolve_project_path(path: Path) -> Path:
    # Convert relative paths into full project paths.
    return path if path.is_absolute() else PROJECT_ROOT / path


def _load_metadata(path: Path) -> pd.DataFrame:
    # Load slot metadata and check that required columns exist.
    if not path.is_file():
        raise FileNotFoundError(f"Missing metadata CSV: {path}")

    df = pd.read_csv(path)
    missing = REQUIRED_COLUMNS.difference(df.columns)
    if missing:
        raise ValueError(f"Metadata is missing required columns: {sorted(missing)}")

    # Convert size-related columns to numbers.
    numeric_columns = [
        "image_width",
        "image_height",
        "width",
        "height",
        "area",
    ]
    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    # Add aspect ratio for bounding box analysis.
    df["aspect_ratio"] = df["width"] / df["height"]
    return df


def _write_class_distribution(df: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    # Count how many vacant and occupied slot annotations exist.
    summary = (
        df.groupby("label")
        .size()
        .reset_index(name="count")
        .sort_values("label")
        .reset_index(drop=True)
    )
    summary["percent"] = (summary["count"] / len(df) * 100).round(4)
    summary.to_csv(output_dir / "class_distribution.csv", index=False)
    return summary


def _write_split_distribution(df: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    # Count slots and unique images in each split.
    split_counts = df.groupby("split").size().rename("slot_count")
    image_counts = df.groupby("split")["image_path"].nunique().rename("image_count")
    summary = (
        pd.concat([split_counts, image_counts], axis=1)
        .reset_index()
        .sort_values("split")
        .reset_index(drop=True)
    )
    summary["slot_percent"] = (summary["slot_count"] / len(df) * 100).round(4)
    summary.to_csv(output_dir / "split_distribution.csv", index=False)
    return summary


def _write_split_label_distribution(df: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    # Count occupied/vacant slots separately for train, valid, and test.
    summary = (
        df.groupby(["split", "label"])
        .size()
        .reset_index(name="count")
        .sort_values(["split", "label"])
        .reset_index(drop=True)
    )
    split_totals = summary.groupby("split")["count"].transform("sum")
    summary["split_percent"] = (summary["count"] / split_totals * 100).round(4)
    summary.to_csv(output_dir / "split_label_distribution.csv", index=False)
    return summary


def _stats_for_group(df: pd.DataFrame, group_name: str, split: str, label: str) -> list[dict]:
    # Calculate box size statistics for one group of data.
    rows: list[dict] = []
    for metric in ["width", "height", "area", "aspect_ratio"]:
        values = df[metric].dropna()
        rows.append(
            {
                "group": group_name,
                "split": split,
                "label": label,
                "metric": metric,
                "count": int(values.count()),
                "mean": round(float(values.mean()), 4),
                "std": round(float(values.std(ddof=0)), 4),
                "min": round(float(values.min()), 4),
                "p25": round(float(values.quantile(0.25)), 4),
                "median": round(float(values.median()), 4),
                "p75": round(float(values.quantile(0.75)), 4),
                "max": round(float(values.max()), 4),
            }
        )
    return rows


def _write_bbox_summary(df: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    # Create summary statistics for bounding box width, height, area, and aspect ratio.
    rows = _stats_for_group(df, "all", "all", "all")

    for split, split_df in df.groupby("split", sort=True):
        rows.extend(_stats_for_group(split_df, "split", str(split), "all"))

    for (split, label), group_df in df.groupby(["split", "label"], sort=True):
        rows.extend(_stats_for_group(group_df, "split_label", str(split), str(label)))

    summary = pd.DataFrame(rows)
    summary.to_csv(output_dir / "bbox_summary.csv", index=False)
    return summary


def _write_image_summary(df: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    # First summarize each image, then summarize images by split.
    image_rows = (
        df.groupby(["split", "image_path", "file_name"], as_index=False)
        .agg(
            slot_count=("annotation_id", "count"),
            occupied_count=("label", lambda labels: int((labels == "occupied").sum())),
            vacant_count=("label", lambda labels: int((labels == "vacant").sum())),
            image_width=("image_width", "first"),
            image_height=("image_height", "first"),
        )
    )
    summary = (
        image_rows.groupby("split", as_index=False)
        .agg(
            image_count=("image_path", "count"),
            total_slots=("slot_count", "sum"),
            min_slots_per_image=("slot_count", "min"),
            mean_slots_per_image=("slot_count", "mean"),
            median_slots_per_image=("slot_count", "median"),
            max_slots_per_image=("slot_count", "max"),
            occupied_slots=("occupied_count", "sum"),
            vacant_slots=("vacant_count", "sum"),
            image_width_min=("image_width", "min"),
            image_width_max=("image_width", "max"),
            image_height_min=("image_height", "min"),
            image_height_max=("image_height", "max"),
        )
        .sort_values("split")
    )
    summary["mean_slots_per_image"] = summary["mean_slots_per_image"].round(4)
    summary["median_slots_per_image"] = summary["median_slots_per_image"].round(4)
    summary.to_csv(output_dir / "image_summary.csv", index=False)
    return summary


def _plot_class_distribution(class_distribution: pd.DataFrame, figures_dir: Path) -> None:
    # Save bar chart for occupied vs vacant counts.
    plt.figure(figsize=(6, 4))
    colors = ["#2f855a" if label == "vacant" else "#c53030" for label in class_distribution["label"]]
    plt.bar(class_distribution["label"], class_distribution["count"], color=colors)
    plt.title("PKLot Class Distribution")
    plt.xlabel("Label")
    plt.ylabel("Slot annotations")
    plt.tight_layout()
    plt.savefig(figures_dir / "class_distribution.png", dpi=150)
    plt.close()


def _plot_split_label_distribution(split_label_distribution: pd.DataFrame, figures_dir: Path) -> None:
    # Save bar chart showing labels inside each dataset split.
    pivot = split_label_distribution.pivot(index="split", columns="label", values="count").fillna(0)
    pivot = pivot.reindex(columns=["occupied", "vacant"], fill_value=0)
    ax = pivot.plot(kind="bar", figsize=(7, 4), color=["#c53030", "#2f855a"])
    ax.set_title("PKLot Split and Label Distribution")
    ax.set_xlabel("Split")
    ax.set_ylabel("Slot annotations")
    ax.legend(title="Label")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(figures_dir / "split_label_distribution.png", dpi=150)
    plt.close()


def _plot_bbox_width_height_hist(df: pd.DataFrame, figures_dir: Path) -> None:
    # Save histogram for bounding box width and height.
    plt.figure(figsize=(7, 4))
    plt.hist(df["width"].dropna(), bins=40, alpha=0.65, label="width", color="#3182ce")
    plt.hist(df["height"].dropna(), bins=40, alpha=0.65, label="height", color="#dd6b20")
    plt.title("Bounding Box Width and Height")
    plt.xlabel("Pixels")
    plt.ylabel("Slot annotations")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures_dir / "bbox_width_height_hist.png", dpi=150)
    plt.close()


def _plot_bbox_aspect_ratio_hist(df: pd.DataFrame, figures_dir: Path) -> None:
    # Save histogram for bounding box shape ratio.
    plt.figure(figsize=(7, 4))
    plt.hist(df["aspect_ratio"].dropna(), bins=40, color="#805ad5")
    plt.title("Bounding Box Aspect Ratio")
    plt.xlabel("width / height")
    plt.ylabel("Slot annotations")
    plt.tight_layout()
    plt.savefig(figures_dir / "bbox_aspect_ratio_hist.png", dpi=150)
    plt.close()


def _dataframe_to_markdown(df: pd.DataFrame) -> str:
    # Convert a pandas table into markdown text for the report.
    columns = [str(column) for column in df.columns]
    rows = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]

    for _, row in df.iterrows():
        rows.append("| " + " | ".join(str(row[column]) for column in df.columns) + " |")

    return "\n".join(rows)


def _write_report(
    output_dir: Path,
    figures_dir: Path,
    class_distribution: pd.DataFrame,
    split_distribution: pd.DataFrame,
    split_label_distribution: pd.DataFrame,
    bbox_summary: pd.DataFrame,
    image_summary: pd.DataFrame,
) -> None:
    # Build one markdown report using all EDA summary tables.
    total_annotations = int(class_distribution["count"].sum())
    occupied = int(
        class_distribution.loc[class_distribution["label"] == "occupied", "count"].sum()
    )
    vacant = int(class_distribution.loc[class_distribution["label"] == "vacant", "count"].sum())
    total_images = int(split_distribution["image_count"].sum())

    report = [
        "# Dataset EDA Report",
        "",
        "This report summarizes generated PKLot slot metadata. It does not contain model training or model evaluation metrics.",
        "",
        "## Overview",
        "",
        f"- Total slot annotations: {total_annotations}",
        f"- Total images referenced: {total_images}",
        f"- Occupied slots: {occupied}",
        f"- Vacant slots: {vacant}",
        "- Positive class: occupied",
        "",
        "## Class Distribution",
        "",
        _dataframe_to_markdown(class_distribution),
        "",
        "## Split Distribution",
        "",
        _dataframe_to_markdown(split_distribution),
        "",
        "## Split Label Distribution",
        "",
        _dataframe_to_markdown(split_label_distribution),
        "",
        "## Image Summary",
        "",
        _dataframe_to_markdown(image_summary),
        "",
        "## Bounding Box Summary",
        "",
        _dataframe_to_markdown(bbox_summary.loc[bbox_summary["group"] == "all"]),
        "",
        "## Figures",
        "",
        f"- {readable_relative_path(figures_dir / 'class_distribution.png', PROJECT_ROOT)}",
        f"- {readable_relative_path(figures_dir / 'split_label_distribution.png', PROJECT_ROOT)}",
        f"- {readable_relative_path(figures_dir / 'bbox_width_height_hist.png', PROJECT_ROOT)}",
        f"- {readable_relative_path(figures_dir / 'bbox_aspect_ratio_hist.png', PROJECT_ROOT)}",
        "",
    ]
    (output_dir / "dataset_eda_report.md").write_text("\n".join(report), encoding="utf-8")


def analyze_metadata(metadata: Path, output_dir: Path, figures_dir: Path) -> int:
    # Main workflow: load metadata, create summaries, save plots, and write report.
    metadata_path = _resolve_project_path(metadata)
    output_path = _resolve_project_path(output_dir)
    figures_path = _resolve_project_path(figures_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    figures_path.mkdir(parents=True, exist_ok=True)

    df = _load_metadata(metadata_path)

    class_distribution = _write_class_distribution(df, output_path)
    split_distribution = _write_split_distribution(df, output_path)
    split_label_distribution = _write_split_label_distribution(df, output_path)
    bbox_summary = _write_bbox_summary(df, output_path)
    image_summary = _write_image_summary(df, output_path)

    _plot_class_distribution(class_distribution, figures_path)
    _plot_split_label_distribution(split_label_distribution, figures_path)
    _plot_bbox_width_height_hist(df, figures_path)
    _plot_bbox_aspect_ratio_hist(df, figures_path)

    _write_report(
        output_dir=output_path,
        figures_dir=figures_path,
        class_distribution=class_distribution,
        split_distribution=split_distribution,
        split_label_distribution=split_label_distribution,
        bbox_summary=bbox_summary,
        image_summary=image_summary,
    )

    print("Dataset EDA complete")
    print(f"Metadata rows: {len(df)}")
    print(f"Images referenced: {df['image_path'].nunique()}")
    print(f"Output directory: {readable_relative_path(output_path, PROJECT_ROOT)}")
    print(f"Figures directory: {readable_relative_path(figures_path, PROJECT_ROOT)}")
    return 0


def main() -> int:
    # Parse arguments and run dataset metadata analysis.
    args = _parse_args()
    return analyze_metadata(
        metadata=args.metadata,
        output_dir=args.output_dir,
        figures_dir=args.figures_dir,
    )


if __name__ == "__main__":
    raise SystemExit(main())
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.data.coco_utils import readable_relative_path
from src.data.weather_labels import (
    UNKNOWN_WEATHER,
    create_image_weather_template,
    load_manual_weather_csv,
    merge_weather_labels_into_metadata,
)
from src.utils.config import METRICS_DIR, PROJECT_ROOT, SPLITS_DIR


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare honest image-level weather labels for PKLot metadata."
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=Path("data/processed/metadata/slot_annotations.csv"),
        help="Slot annotation metadata CSV.",
    )
    parser.add_argument(
        "--manual-weather-csv",
        type=Path,
        default=None,
        help="Optional manually curated weather labels CSV.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/splits/weather_labels_template.csv"),
        help="Output image-level weather template CSV.",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=Path("results/metrics/dataset_eda/weather_summary.csv"),
        help="Output weather summary CSV.",
    )
    return parser.parse_args()


def _resolve_project_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def _write_weather_summary(
    metadata_with_weather: pd.DataFrame,
    weather_template: pd.DataFrame,
    output_path: Path,
) -> pd.DataFrame:
    image_counts = (
        weather_template.groupby(["split", "weather", "weather_source"])
        .size()
        .rename("image_count")
    )
    slot_counts = (
        metadata_with_weather.groupby(["split", "weather", "weather_source"])
        .size()
        .rename("slot_count")
    )
    summary = (
        pd.concat([image_counts, slot_counts], axis=1)
        .fillna(0)
        .reset_index()
        .sort_values(["split", "weather", "weather_source"])
    )
    summary["image_count"] = summary["image_count"].astype(int)
    summary["slot_count"] = summary["slot_count"].astype(int)

    overall = pd.DataFrame(
        [
            {
                "split": "all",
                "weather": weather,
                "weather_source": source,
                "image_count": int(group["image_path"].nunique()),
                "slot_count": int(len(group)),
            }
            for (weather, source), group in metadata_with_weather.groupby(
                ["weather", "weather_source"], sort=True
            )
        ]
    )

    summary = pd.concat([overall, summary], ignore_index=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_path, index=False)
    return summary


def prepare_weather_labels(
    metadata: Path,
    manual_weather_csv: Path | None,
    output: Path,
    summary_output: Path,
) -> int:
    metadata_path = _resolve_project_path(metadata)
    output_path = _resolve_project_path(output)
    summary_path = _resolve_project_path(summary_output)

    if not metadata_path.is_file():
        raise FileNotFoundError(f"Missing metadata CSV: {metadata_path}")

    slot_metadata = pd.read_csv(
        metadata_path,
        usecols=["split", "file_name", "image_path"],
    )
    full_metadata = pd.read_csv(metadata_path)

    manual_weather = None
    if manual_weather_csv is not None:
        manual_path = _resolve_project_path(manual_weather_csv)
        manual_weather = load_manual_weather_csv(manual_path)

    weather_template = create_image_weather_template(slot_metadata, manual_weather)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    weather_template.to_csv(output_path, index=False)

    metadata_with_weather = merge_weather_labels_into_metadata(
        full_metadata,
        weather_template,
    )
    summary = _write_weather_summary(metadata_with_weather, weather_template, summary_path)

    known_images = int((weather_template["weather"] != UNKNOWN_WEATHER).sum())
    total_images = len(weather_template)

    print("Weather label preparation complete")
    print(f"Template: {readable_relative_path(output_path, PROJECT_ROOT)}")
    print(f"Summary: {readable_relative_path(summary_path, PROJECT_ROOT)}")
    print(f"Images with known weather: {known_images}/{total_images}")
    if known_images == 0:
        print(
            "All weather labels are unknown because no reliable weather path metadata "
            "or manual weather CSV was provided."
        )
    else:
        print(summary.to_string(index=False))

    return 0


def main() -> int:
    args = _parse_args()
    return prepare_weather_labels(
        metadata=args.metadata,
        manual_weather_csv=args.manual_weather_csv,
        output=args.output,
        summary_output=args.summary_output,
    )


if __name__ == "__main__":
    raise SystemExit(main())

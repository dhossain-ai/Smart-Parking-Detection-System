from __future__ import annotations

import re
from pathlib import Path

import pandas as pd


KNOWN_WEATHER_LABELS = {"sunny", "rainy", "cloudy", "overcast"}
UNKNOWN_WEATHER = "unknown"
WEATHER_SOURCE_UNKNOWN = "unknown"
WEATHER_SOURCE_PATH = "path"
WEATHER_SOURCE_MANUAL = "manual_csv"


def normalize_weather_label(value: str | None) -> str:
    if value is None:
        return UNKNOWN_WEATHER

    label = str(value).strip().lower()
    if label in KNOWN_WEATHER_LABELS:
        return label
    return UNKNOWN_WEATHER


def infer_weather_from_path(path_value: str | Path) -> tuple[str, str]:
    """Infer weather only when a reliable weather token appears in path folders."""
    path = Path(str(path_value))
    parts = path.parts
    if path.suffix:
        parts = parts[:-1]

    detected: set[str] = set()
    for part in parts:
        tokens = [
            token
            for token in re.split(r"[^a-zA-Z]+", part.lower())
            if token
        ]
        detected.update(token for token in tokens if token in KNOWN_WEATHER_LABELS)

    if len(detected) == 1:
        return detected.pop(), WEATHER_SOURCE_PATH

    return UNKNOWN_WEATHER, WEATHER_SOURCE_UNKNOWN


def load_manual_weather_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "weather" not in df.columns:
        raise ValueError(f"Manual weather CSV must contain a weather column: {path}")
    if not {"image_path", "file_name"}.intersection(df.columns):
        raise ValueError(
            "Manual weather CSV must contain at least one key column: image_path or file_name"
        )

    if "split" not in df.columns:
        df["split"] = ""
    if "image_path" not in df.columns:
        df["image_path"] = ""
    if "file_name" not in df.columns:
        df["file_name"] = ""
    if "weather_source" not in df.columns:
        df["weather_source"] = WEATHER_SOURCE_MANUAL

    df["weather"] = df["weather"].map(normalize_weather_label)
    df.loc[df["weather_source"].isna(), "weather_source"] = WEATHER_SOURCE_MANUAL
    return df[["split", "file_name", "image_path", "weather", "weather_source"]]


def create_image_weather_template(
    slot_metadata: pd.DataFrame,
    manual_weather: pd.DataFrame | None = None,
) -> pd.DataFrame:
    required = {"split", "file_name", "image_path"}
    missing = required.difference(slot_metadata.columns)
    if missing:
        raise ValueError(f"Slot metadata is missing required columns: {sorted(missing)}")

    template = (
        slot_metadata[["split", "file_name", "image_path"]]
        .drop_duplicates()
        .sort_values(["split", "image_path", "file_name"])
        .reset_index(drop=True)
    )

    inferred = template["image_path"].map(infer_weather_from_path)
    template["weather"] = [weather for weather, _ in inferred]
    template["weather_source"] = [source for _, source in inferred]

    if manual_weather is not None and not manual_weather.empty:
        manual_by_image_path = {
            (str(row.split), str(row.image_path)): (row.weather, row.weather_source)
            for row in manual_weather.itertuples(index=False)
            if str(row.image_path)
        }
        manual_by_file_name = {
            (str(row.split), str(row.file_name)): (row.weather, row.weather_source)
            for row in manual_weather.itertuples(index=False)
            if str(row.file_name)
        }

        for index, row in template.iterrows():
            split = str(row["split"])
            image_path_key = (split, str(row["image_path"]))
            file_name_key = (split, str(row["file_name"]))

            weather_info = manual_by_image_path.get(image_path_key)
            if weather_info is None:
                weather_info = manual_by_file_name.get(file_name_key)

            if weather_info is not None:
                template.at[index, "weather"] = weather_info[0]
                template.at[index, "weather_source"] = weather_info[1]

    return template


def merge_weather_labels_into_metadata(
    slot_metadata: pd.DataFrame,
    weather_labels: pd.DataFrame,
) -> pd.DataFrame:
    required = {"split", "image_path", "weather", "weather_source"}
    missing = required.difference(weather_labels.columns)
    if missing:
        raise ValueError(f"Weather labels are missing required columns: {sorted(missing)}")

    return slot_metadata.merge(
        weather_labels[["split", "image_path", "weather", "weather_source"]],
        on=["split", "image_path"],
        how="left",
    ).assign(
        weather=lambda df: df["weather"].fillna(UNKNOWN_WEATHER),
        weather_source=lambda df: df["weather_source"].fillna(WEATHER_SOURCE_UNKNOWN),
    )

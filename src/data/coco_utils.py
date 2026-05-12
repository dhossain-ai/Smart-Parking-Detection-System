from __future__ import annotations

import json
from pathlib import Path
from typing import Any


# Final labels used by this parking project.
PROJECT_LABELS = {"occupied", "vacant"}


def load_coco_json(path: Path) -> dict[str, Any]:
    """Load a COCO annotation JSON file."""
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    # COCO annotation file should be a JSON object/dictionary.
    if not isinstance(data, dict):
        raise ValueError(f"COCO file did not contain a JSON object: {path}")

    return data


def index_images_by_id(coco_data: dict[str, Any]) -> dict[int, dict[str, Any]]:
    # Create a lookup table: image_id -> image information.
    images_by_id: dict[int, dict[str, Any]] = {}

    for image in coco_data.get("images", []):
        image_id = image.get("id")
        if image_id is None:
            continue
        images_by_id[int(image_id)] = image

    return images_by_id


def index_categories_by_id(coco_data: dict[str, Any]) -> dict[int, dict[str, Any]]:
    # Create a lookup table: category_id -> category information.
    categories_by_id: dict[int, dict[str, Any]] = {}

    for category in coco_data.get("categories", []):
        category_id = category.get("id")
        if category_id is None:
            continue
        categories_by_id[int(category_id)] = category

    return categories_by_id


def normalize_category_name(category_name: str) -> str | None:
    # Convert different dataset category names into project labels.
    normalized = (
        str(category_name)
        .strip()
        .lower()
        .replace("_", "-")
        .replace(" ", "-")
    )

    if not normalized:
        return None

    # Direct known names for occupied slots.
    if normalized in {"occupied", "space-occupied", "slot-occupied"}:
        return "occupied"

    # Direct known names for vacant slots.
    if normalized in {"vacant", "empty", "space-empty", "slot-empty", "space-vacant"}:
        return "vacant"

    # Fallback check for category names with extra words.
    tokens = [token for token in normalized.split("-") if token]
    if "occupied" in tokens:
        return "occupied"
    if "vacant" in tokens or "empty" in tokens:
        return "vacant"

    # Unknown category name.
    return None


def clip_coco_bbox(
    bbox: list[float] | tuple[float, ...],
    image_width: int,
    image_height: int,
) -> dict[str, float] | None:
    # Validate COCO bbox format: [x, y, width, height].
    if len(bbox) != 4:
        return None

    try:
        x, y, width, height = (float(value) for value in bbox)
    except (TypeError, ValueError):
        return None

    # Reject invalid boxes or invalid image sizes.
    if width <= 0 or height <= 0 or image_width <= 0 or image_height <= 0:
        return None

    # Clip the box so it stays inside the image boundary.
    x1 = max(0.0, x)
    y1 = max(0.0, y)
    x2 = min(float(image_width), x + width)
    y2 = min(float(image_height), y + height)

    clipped_width = x2 - x1
    clipped_height = y2 - y1

    # Reject boxes that become empty after clipping.
    if clipped_width <= 0 or clipped_height <= 0:
        return None

    # Return both COCO-style box values and corner coordinates.
    return {
        "x": x1,
        "y": y1,
        "width": clipped_width,
        "height": clipped_height,
        "x1": x1,
        "y1": y1,
        "x2": x2,
        "y2": y2,
        "area": clipped_width * clipped_height,
    }


def readable_relative_path(path: Path, base_dir: Path) -> str:
    # Make paths shorter for printed output and reports.
    try:
        readable = path.relative_to(base_dir)
    except ValueError:
        readable = path

    return readable.as_posix()
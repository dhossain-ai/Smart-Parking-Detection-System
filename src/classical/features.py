from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from skimage.feature import hog, local_binary_pattern

from src.utils.config import PROJECT_ROOT, get_pklot_dir


@dataclass(frozen=True)
class FeatureConfig:
    image_size: int = 64
    lbp_radius: int = 2
    lbp_points: int = 16
    hsv_bins: tuple[int, int, int] = (8, 8, 8)
    hog_orientations: int = 9
    hog_pixels_per_cell: tuple[int, int] = (8, 8)
    hog_cells_per_block: tuple[int, int] = (2, 2)


def resolve_image_path(image_path_value: str | Path) -> Path:
    image_path = Path(str(image_path_value))
    if image_path.is_absolute():
        return image_path

    dataset_path = get_pklot_dir() / image_path
    if dataset_path.is_file():
        return dataset_path

    return PROJECT_ROOT / image_path


def load_image_rgb(image_path: str | Path) -> np.ndarray | None:
    resolved_path = resolve_image_path(image_path)
    image_bgr = cv2.imread(str(resolved_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        return None
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def crop_slot_image(
    image: np.ndarray,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
) -> np.ndarray | None:
    height, width = image.shape[:2]

    left = max(0, min(width, int(np.floor(x1))))
    top = max(0, min(height, int(np.floor(y1))))
    right = max(0, min(width, int(np.ceil(x2))))
    bottom = max(0, min(height, int(np.ceil(y2))))

    if right <= left or bottom <= top:
        return None

    return image[top:bottom, left:right]


def resize_crop(crop: np.ndarray, image_size: int = 64) -> np.ndarray:
    return cv2.resize(crop, (image_size, image_size), interpolation=cv2.INTER_AREA)


def extract_lbp_features(
    crop_rgb: np.ndarray,
    radius: int = 2,
    points: int = 16,
) -> np.ndarray:
    gray = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2GRAY)
    lbp = local_binary_pattern(gray, P=points, R=radius, method="uniform")
    bins = points + 2
    hist, _ = np.histogram(lbp.ravel(), bins=bins, range=(0, bins), density=False)
    hist = hist.astype(np.float32)
    total = hist.sum()
    if total > 0:
        hist /= total
    return hist


def extract_hsv_histogram(
    crop_rgb: np.ndarray,
    bins: tuple[int, int, int] = (8, 8, 8),
) -> np.ndarray:
    crop_hsv = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2HSV)
    hist = cv2.calcHist(
        [crop_hsv],
        [0, 1, 2],
        None,
        list(bins),
        [0, 180, 0, 256, 0, 256],
    )
    hist = hist.astype(np.float32).ravel()
    total = hist.sum()
    if total > 0:
        hist /= total
    return hist


def extract_hog_features(
    crop_rgb: np.ndarray,
    orientations: int = 9,
    pixels_per_cell: tuple[int, int] = (8, 8),
    cells_per_block: tuple[int, int] = (2, 2),
) -> np.ndarray:
    gray = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    features = hog(
        gray,
        orientations=orientations,
        pixels_per_cell=pixels_per_cell,
        cells_per_block=cells_per_block,
        block_norm="L2-Hys",
        transform_sqrt=True,
        feature_vector=True,
    )
    return features.astype(np.float32)


def extract_combined_features(
    crop_rgb: np.ndarray,
    config: FeatureConfig | None = None,
    feature_set: str = "lbp_hsv_hog",
) -> np.ndarray:
    config = config or FeatureConfig()
    resized = resize_crop(crop_rgb, config.image_size)
    feature_parts: list[np.ndarray] = []
    requested = set(feature_set.split("_"))

    if "lbp" in requested:
        feature_parts.append(
            extract_lbp_features(
                resized,
                radius=config.lbp_radius,
                points=config.lbp_points,
            )
        )
    if "hsv" in requested:
        feature_parts.append(extract_hsv_histogram(resized, bins=config.hsv_bins))
    if "hog" in requested:
        feature_parts.append(
            extract_hog_features(
                resized,
                orientations=config.hog_orientations,
                pixels_per_cell=config.hog_pixels_per_cell,
                cells_per_block=config.hog_cells_per_block,
            )
        )

    if not feature_parts:
        raise ValueError(f"Unsupported feature set: {feature_set}")

    return np.concatenate(feature_parts).astype(np.float32)


def extract_features_from_record(
    record: dict,
    config: FeatureConfig | None = None,
    feature_set: str = "lbp_hsv_hog",
) -> np.ndarray | None:
    image = load_image_rgb(record["image_path"])
    if image is None:
        return None

    try:
        crop = crop_slot_image(
            image=image,
            x1=float(record["x1"]),
            y1=float(record["y1"]),
            x2=float(record["x2"]),
            y2=float(record["y2"]),
        )
    except (KeyError, TypeError, ValueError):
        return None

    if crop is None or crop.size == 0:
        return None

    features = extract_combined_features(crop, config=config, feature_set=feature_set)
    if not np.isfinite(features).all():
        return None

    return features

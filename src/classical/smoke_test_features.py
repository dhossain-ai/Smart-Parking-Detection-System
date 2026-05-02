from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np

try:
    from tqdm import tqdm
except ModuleNotFoundError:
    def tqdm(iterable, **_: object):
        return iterable

from src.classical.dataset import load_manifest_records
from src.classical.features import FeatureConfig, extract_features_from_record


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Smoke test classical feature extraction on a small manifest sample."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/splits/train_slots_balanced_small.csv"),
        help="Manifest CSV to sample from.",
    )
    parser.add_argument(
        "--samples-per-class",
        type=int,
        default=5,
        help="Records per class to test.",
    )
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--feature-set", default="lbp_hsv_hog")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    records = load_manifest_records(
        args.manifest,
        max_per_class=args.samples_per_class,
        seed=args.seed,
    )
    if not records:
        raise RuntimeError("No records loaded for feature smoke test.")

    config = FeatureConfig(image_size=args.image_size)
    features: list[np.ndarray] = []
    skipped = 0

    for record in tqdm(records, desc="Smoke testing features", unit="slot"):
        vector = extract_features_from_record(
            record.as_feature_record(),
            config=config,
            feature_set=args.feature_set,
        )
        if vector is None:
            skipped += 1
            continue
        features.append(vector)

    if not features:
        raise RuntimeError("Feature smoke test failed: no feature vectors extracted.")

    matrix = np.vstack(features)
    labels = Counter(record.label for record in records)
    invalid_values = int((~np.isfinite(matrix)).sum())

    print("Classical feature smoke test")
    print(f"Records loaded: {len(records)}")
    print(f"Features extracted: {len(features)}")
    print(f"Skipped records: {skipped}")
    print(f"Label counts: {dict(labels)}")
    print(f"Feature matrix shape: {matrix.shape}")
    print(f"NaN/inf values: {invalid_values}")

    if invalid_values:
        raise RuntimeError("Feature matrix contains NaN or infinite values.")
    if set(labels).difference({"occupied", "vacant"}):
        raise RuntimeError(f"Unexpected labels found: {sorted(labels)}")

    print("Status: PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

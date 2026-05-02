from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.neural.dataset import PKLotSlotDataset, count_targets
from src.neural.model import build_model


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke test the CNN data/model path.")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/splits/train_slots_balanced_small.csv"),
        help="Manifest CSV to sample from.",
    )
    parser.add_argument("--samples-per-class", type=int, default=4)
    parser.add_argument("--model-version", choices=["v1", "v2"], default="v2")
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    dataset = PKLotSlotDataset(
        args.manifest,
        image_size=args.image_size,
        augment=True,
        max_per_class=args.samples_per_class,
        seed=args.seed,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    inputs, labels = next(iter(loader))
    model = build_model(
        num_classes=2,
        model_version=args.model_version,
        dropout=args.dropout,
    )
    logits = model(inputs)

    print("CNN smoke test")
    print(f"Dataset records: {len(dataset)}")
    print(f"Model version: {args.model_version}")
    print(f"Label counts: {count_targets(dataset)}")
    print(f"Input batch shape: {tuple(inputs.shape)}")
    print(f"Logits shape: {tuple(logits.shape)}")
    print(f"Labels: {labels.tolist()}")

    if inputs.shape[1:] != (3, args.image_size, args.image_size):
        raise RuntimeError(f"Unexpected input shape: {tuple(inputs.shape)}")
    if logits.shape != (inputs.shape[0], 2):
        raise RuntimeError(f"Unexpected logits shape: {tuple(logits.shape)}")
    if not torch.isfinite(inputs).all():
        raise RuntimeError("Input batch contains NaN or infinite values.")
    if not torch.isfinite(logits).all():
        raise RuntimeError("Logits contain NaN or infinite values.")
    if not set(labels.tolist()).issubset({0, 1}):
        raise RuntimeError(f"Unexpected label targets: {labels.tolist()}")

    print("Status: PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

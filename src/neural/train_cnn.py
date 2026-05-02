from __future__ import annotations

import argparse
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

try:
    from tqdm import tqdm
except ModuleNotFoundError:
    def tqdm(iterable, **_: object):
        return iterable

from src.data.coco_utils import readable_relative_path
from src.neural.dataset import PKLotSlotDataset, count_targets
from src.neural.evaluate_cnn import (
    build_threshold_sweep,
    compute_metrics,
    predict_loader_scores,
    save_evaluation_outputs,
    select_threshold_from_sweep,
    threshold_meets_requirements,
)
from src.neural.model import build_model
from src.utils.config import FIGURES_DIR, PROJECT_ROOT


DEFAULT_TRAIN_MANIFEST = Path("data/splits/train_slots_balanced_small.csv")
DEFAULT_VALID_MANIFEST = Path("data/splits/valid_slots_balanced_small.csv")
DEFAULT_TEST_MANIFEST = Path("data/splits/test_slots_balanced_small.csv")
FULL_TRAIN_MANIFEST = Path("data/splits/train_slots.csv")
FULL_VALID_MANIFEST = Path("data/splits/valid_slots.csv")
FULL_TEST_MANIFEST = Path("data/splits/test_slots.csv")
DEFAULT_OUTPUT_MODEL = Path("models/cnn/best_cnn_model_v2.pth")
DEFAULT_OUTPUT_DIR = Path("results/metrics/cnn_tuned")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a custom CNN for PKLot slots.")
    parser.add_argument("--train-manifest", type=Path, default=None)
    parser.add_argument("--valid-manifest", type=Path, default=None)
    parser.add_argument("--test-manifest", type=Path, default=None)
    parser.add_argument("--model-version", choices=["v1", "v2"], default="v2")
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--samples-per-class", type=int, default=None)
    parser.add_argument("--max-train-per-class", type=int, default=None)
    parser.add_argument("--max-valid-per-class", type=int, default=None)
    parser.add_argument("--max-test-per-class", type=int, default=None)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--no-augment", action="store_true")
    parser.add_argument("--eval-threshold", default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-model", type=Path, default=DEFAULT_OUTPUT_MODEL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def _resolve_project_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def _default_manifest(split: str, samples_per_class: int | None) -> Path:
    if samples_per_class is None:
        return {
            "train": DEFAULT_TRAIN_MANIFEST,
            "valid": DEFAULT_VALID_MANIFEST,
            "test": DEFAULT_TEST_MANIFEST,
        }[split]

    return {
        "train": FULL_TRAIN_MANIFEST,
        "valid": FULL_VALID_MANIFEST,
        "test": FULL_TEST_MANIFEST,
    }[split]


def _per_class_limit(
    specific_limit: int | None,
    shared_limit: int | None,
) -> int | None:
    return specific_limit if specific_limit is not None else shared_limit


def _parse_eval_threshold(value: str) -> float | None:
    if value.strip().lower() == "auto":
        return None

    threshold = float(value)
    if not 0.0 < threshold < 1.0:
        raise ValueError("--eval-threshold must be 'auto' or a number between 0 and 1")
    return threshold


def set_random_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def make_loader(
    dataset: PKLotSlotDataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    seed: int,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        generator=generator,
    )


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
) -> float:
    model.train()
    total_loss = 0.0
    total_records = 0

    for inputs, targets in tqdm(loader, desc=f"Epoch {epoch} train", unit="batch"):
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(inputs)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()

        batch_size = int(targets.size(0))
        total_loss += float(loss.item()) * batch_size
        total_records += batch_size

    return total_loss / max(1, total_records)


@torch.no_grad()
def evaluate_loss_and_predictions(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, np.ndarray, np.ndarray, float]:
    model.eval()
    total_loss = 0.0
    total_records = 0
    all_targets: list[np.ndarray] = []
    all_predictions: list[np.ndarray] = []
    start = time.perf_counter()

    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets_device = targets.to(device)
        logits = model(inputs)
        loss = criterion(logits, targets_device)
        predictions = torch.argmax(logits, dim=1).cpu().numpy()

        batch_size = int(targets.size(0))
        total_loss += float(loss.item()) * batch_size
        total_records += batch_size
        all_targets.append(targets.numpy())
        all_predictions.append(predictions)

    elapsed = time.perf_counter() - start
    return (
        total_loss / max(1, total_records),
        np.concatenate(all_targets),
        np.concatenate(all_predictions),
        elapsed,
    )


def train_cnn(args: argparse.Namespace) -> int:
    set_random_seeds(args.seed)
    train_manifest = args.train_manifest or _default_manifest("train", args.samples_per_class)
    valid_manifest = args.valid_manifest or _default_manifest("valid", args.samples_per_class)
    test_manifest = args.test_manifest or _default_manifest("test", args.samples_per_class)
    max_train_per_class = _per_class_limit(args.max_train_per_class, args.samples_per_class)
    max_valid_per_class = _per_class_limit(args.max_valid_per_class, args.samples_per_class)
    max_test_per_class = _per_class_limit(args.max_test_per_class, args.samples_per_class)

    output_model_path = _resolve_project_path(args.output_model)
    output_dir = _resolve_project_path(args.output_dir)
    output_model_path.parent.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dataset = PKLotSlotDataset(
        train_manifest,
        image_size=args.image_size,
        augment=not args.no_augment,
        max_per_class=max_train_per_class,
        seed=args.seed,
    )
    valid_dataset = PKLotSlotDataset(
        valid_manifest,
        image_size=args.image_size,
        augment=False,
        max_per_class=max_valid_per_class,
        seed=args.seed,
    )
    test_dataset = PKLotSlotDataset(
        test_manifest,
        image_size=args.image_size,
        augment=False,
        max_per_class=max_test_per_class,
        seed=args.seed,
    )

    train_loader = make_loader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        seed=args.seed,
    )
    valid_loader = make_loader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        seed=args.seed,
    )
    test_loader = make_loader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        seed=args.seed,
    )

    model = build_model(
        num_classes=2,
        model_version=args.model_version,
        dropout=args.dropout,
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=1,
    )

    print("CNN training configuration")
    print(f"Device: {device}")
    print(f"Model version: {args.model_version}")
    print(f"Train manifest: {train_manifest}")
    print(f"Valid manifest: {valid_manifest}")
    print(f"Test manifest: {test_manifest}")
    print(f"Train records: {len(train_dataset)} {count_targets(train_dataset)}")
    print(f"Valid records: {len(valid_dataset)} {count_targets(valid_dataset)}")
    print(f"Test records: {len(test_dataset)} {count_targets(test_dataset)}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")

    history: list[dict[str, Any]] = []
    best_valid_f1 = -1.0
    best_epoch = 0
    epochs_without_improvement = 0

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch)
        valid_loss, y_valid, pred_valid, valid_seconds = evaluate_loss_and_predictions(
            model,
            valid_loader,
            criterion,
            device,
        )
        valid_metrics = compute_metrics(y_valid, pred_valid, "valid", valid_seconds)
        scheduler.step(valid_metrics["f1_score"])

        epoch_row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "valid_loss": valid_loss,
            "valid_accuracy": valid_metrics["accuracy"],
            "valid_precision": valid_metrics["precision"],
            "valid_recall": valid_metrics["recall"],
            "valid_f1": valid_metrics["f1_score"],
            "lr": optimizer.param_groups[0]["lr"],
        }
        history.append(epoch_row)
        print(
            f"Epoch {epoch}: "
            f"train_loss={train_loss:.4f} "
            f"valid_loss={valid_loss:.4f} "
            f"valid_f1={valid_metrics['f1_score']:.4f}"
        )

        if valid_metrics["f1_score"] > best_valid_f1:
            best_valid_f1 = valid_metrics["f1_score"]
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "architecture": "ParkingSlotCNN" if args.model_version == "v1" else "ParkingSlotCNNV2",
                    "model_version": args.model_version,
                    "dropout": args.dropout,
                    "image_size": args.image_size,
                    "epoch": epoch,
                    "best_valid_f1": best_valid_f1,
                    "history": history,
                    "label_to_target": {"vacant": 0, "occupied": 1},
                },
                output_model_path,
            )
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= args.patience:
            print(f"Early stopping after epoch {epoch}")
            break

    checkpoint = torch.load(output_model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    valid_loss, y_valid, pred_valid, valid_seconds = evaluate_loss_and_predictions(
        model,
        valid_loader,
        criterion,
        device,
    )
    y_valid_threshold, _, valid_scores, _ = predict_loader_scores(
        model,
        valid_loader,
        device,
        threshold=0.5,
    )
    valid_sweep = build_threshold_sweep(y_valid_threshold, valid_scores, split="valid")
    fixed_threshold = _parse_eval_threshold(args.eval_threshold)
    if fixed_threshold is None:
        decision_threshold, threshold_selection = select_threshold_from_sweep(valid_sweep)
    else:
        decision_threshold = fixed_threshold
        fixed_row = valid_sweep.loc[
            valid_sweep["threshold"].sub(decision_threshold).abs().idxmin()
        ]
        threshold_selection = {
            "selection_reason": "fixed_cli_threshold",
            "requirements_met": bool(threshold_meets_requirements(fixed_row)),
            "selected_validation_accuracy": float(fixed_row["accuracy"]),
            "selected_validation_precision": float(fixed_row["precision"]),
            "selected_validation_recall": float(fixed_row["recall"]),
            "selected_validation_f1": float(fixed_row["f1_score"]),
            "selected_validation_false_occupancy_rate": float(fixed_row["false_occupancy_rate"]),
        }
    threshold_valid_f1 = float(threshold_selection["selected_validation_f1"])
    y_valid, pred_valid, _, valid_seconds = predict_loader_scores(
        model,
        valid_loader,
        device,
        threshold=decision_threshold,
    )
    y_test, pred_test, test_scores, test_seconds = predict_loader_scores(
        model,
        test_loader,
        device,
        threshold=decision_threshold,
    )
    test_sweep = build_threshold_sweep(y_test, test_scores, split="test")
    threshold_sweep = pd.concat([valid_sweep, test_sweep], ignore_index=True)
    valid_metrics = compute_metrics(y_valid, pred_valid, "valid", valid_seconds)
    test_metrics = compute_metrics(y_test, pred_test, "test", test_seconds)

    for metrics in [valid_metrics, test_metrics]:
        metrics["architecture"] = "ParkingSlotCNN" if args.model_version == "v1" else "ParkingSlotCNNV2"
        metrics["model_version"] = args.model_version
        metrics["image_size"] = args.image_size
        metrics["batch_size"] = args.batch_size
        metrics["epochs_completed"] = len(history)
        metrics["best_epoch"] = best_epoch
        metrics["best_valid_f1"] = best_valid_f1
        metrics["threshold_valid_f1"] = threshold_valid_f1
        metrics["decision_threshold"] = decision_threshold
        metrics["threshold_selection_reason"] = threshold_selection["selection_reason"]
        metrics["requirements_met"] = threshold_selection["requirements_met"]
        metrics["train_records_used"] = len(train_dataset)
        metrics["valid_records_used"] = len(valid_dataset)
        metrics["test_records_used"] = len(test_dataset)
    valid_metrics["loss"] = float(valid_loss)

    checkpoint["decision_threshold"] = decision_threshold
    checkpoint["threshold_selection"] = threshold_selection
    checkpoint["threshold_valid_f1"] = threshold_valid_f1
    checkpoint["history"] = history
    torch.save(checkpoint, output_model_path)

    pd.DataFrame(history).to_csv(output_dir / "training_history.csv", index=False)
    figure_prefix = "cnn_tuned" if output_dir.name == "cnn_tuned" else "cnn"
    save_evaluation_outputs(
        output_dir=output_dir,
        figures_dir=FIGURES_DIR,
        valid_metrics=valid_metrics,
        test_metrics=test_metrics,
        valid_predictions=(y_valid, pred_valid),
        test_predictions=(y_test, pred_test),
        history=history,
        extra={
            "model": readable_relative_path(output_model_path, PROJECT_ROOT),
            "device": str(device),
            "model_version": args.model_version,
            "decision_threshold": decision_threshold,
            "threshold_selection": threshold_selection,
            "threshold_valid_f1": threshold_valid_f1,
            "threshold_note": "Threshold is applied to occupied probability.",
            "confusion_matrix_layout": "Rows are actual labels, columns are predicted labels, order is vacant then occupied.",
        },
        threshold_sweep=threshold_sweep,
        figure_prefix=figure_prefix,
    )

    print("CNN training complete")
    print(f"Model: {readable_relative_path(output_model_path, PROJECT_ROOT)}")
    print(f"Metrics: {readable_relative_path(output_dir / 'cnn_metrics.csv', PROJECT_ROOT)}")
    print(f"Epochs completed: {len(history)}")
    print(f"Best validation F1: {best_valid_f1:.4f} at epoch {best_epoch}")
    print(
        f"Decision threshold: {decision_threshold:.2f} "
        f"(validation F1={threshold_valid_f1:.4f})"
    )
    print(f"Threshold selection: {threshold_selection['selection_reason']}")
    print(f"Requirement met: {'Yes' if threshold_selection['requirements_met'] else 'No'}")
    print(
        "Test metrics: "
        f"accuracy={test_metrics['accuracy']:.4f} "
        f"precision={test_metrics['precision']:.4f} "
        f"recall={test_metrics['recall']:.4f} "
        f"f1={test_metrics['f1_score']:.4f} "
        f"false_occupancy_rate={test_metrics['false_occupancy_rate']:.4f}"
    )
    return 0


def main() -> int:
    return train_cnn(_parse_args())


if __name__ == "__main__":
    raise SystemExit(main())

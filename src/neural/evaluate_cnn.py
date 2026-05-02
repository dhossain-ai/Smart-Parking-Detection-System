from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader

from src.data.coco_utils import readable_relative_path
from src.neural.dataset import PKLotSlotDataset
from src.neural.model import build_model
from src.utils.config import FIGURES_DIR, PROJECT_ROOT


LABELS = [0, 1]
LABEL_NAMES = ["vacant", "occupied"]
REQUIREMENT_THRESHOLDS = {
    "accuracy": 0.98,
    "precision": 0.97,
    "recall": 0.97,
    "f1_score": 0.97,
}


def false_occupancy_rate(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Actual vacant predicted as occupied divided by total actual vacant."""
    vacant_mask = y_true == 0
    total_vacant = int(vacant_mask.sum())
    if total_vacant == 0:
        return 0.0
    false_occupied = int(((y_pred == 1) & vacant_mask).sum())
    return false_occupied / total_vacant


@torch.no_grad()
def predict_loader(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, float]:
    model.eval()
    all_targets: list[np.ndarray] = []
    all_predictions: list[np.ndarray] = []
    start = time.perf_counter()

    for inputs, targets in loader:
        inputs = inputs.to(device)
        logits = model(inputs)
        predictions = torch.argmax(logits, dim=1).cpu().numpy()
        all_predictions.append(predictions)
        all_targets.append(targets.numpy())

    elapsed = time.perf_counter() - start
    return np.concatenate(all_targets), np.concatenate(all_predictions), elapsed


@torch.no_grad()
def predict_loader_scores(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    threshold: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    model.eval()
    all_targets: list[np.ndarray] = []
    all_scores: list[np.ndarray] = []
    start = time.perf_counter()

    for inputs, targets in loader:
        inputs = inputs.to(device)
        logits = model(inputs)
        scores = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
        all_scores.append(scores)
        all_targets.append(targets.numpy())

    elapsed = time.perf_counter() - start
    y_true = np.concatenate(all_targets)
    occupied_scores = np.concatenate(all_scores)
    predictions = (occupied_scores >= threshold).astype(np.int64)
    return y_true, predictions, occupied_scores, elapsed


def build_threshold_sweep(
    y_true: np.ndarray,
    occupied_scores: np.ndarray,
    split: str,
) -> pd.DataFrame:
    rows = []
    for threshold in np.round(np.arange(0.10, 0.901, 0.01), 2):
        predictions = (occupied_scores >= threshold).astype(np.int64)
        metrics = compute_metrics(
            y_true,
            predictions,
            split=split,
            inference_seconds=0.0,
        )
        metrics["threshold"] = float(threshold)
        rows.append(metrics)
    return pd.DataFrame(rows)


def threshold_meets_requirements(row: pd.Series) -> bool:
    return all(float(row[metric]) >= required for metric, required in REQUIREMENT_THRESHOLDS.items())


def select_threshold_from_sweep(sweep: pd.DataFrame) -> tuple[float, dict[str, Any]]:
    candidates = sweep[sweep.apply(threshold_meets_requirements, axis=1)]
    if not candidates.empty:
        selected = candidates.sort_values(
            ["f1_score", "accuracy", "precision", "recall"],
            ascending=False,
        ).iloc[0]
        reason = "meets_all_requirements"
    else:
        recall_safe = sweep[sweep["recall"] >= REQUIREMENT_THRESHOLDS["recall"]]
        if not recall_safe.empty:
            selected = recall_safe.sort_values(
                ["f1_score", "accuracy", "precision"],
                ascending=False,
            ).iloc[0]
            reason = "best_f1_with_recall_at_least_0.97"
        else:
            selected = sweep.sort_values(["f1_score", "accuracy"], ascending=False).iloc[0]
            reason = "best_f1_overall"

    result = {
        "selection_reason": reason,
        "requirements_met": bool(threshold_meets_requirements(selected)),
        "selected_validation_accuracy": float(selected["accuracy"]),
        "selected_validation_precision": float(selected["precision"]),
        "selected_validation_recall": float(selected["recall"]),
        "selected_validation_f1": float(selected["f1_score"]),
        "selected_validation_false_occupancy_rate": float(selected["false_occupancy_rate"]),
    }
    return float(selected["threshold"]), result


def find_best_threshold(
    y_true: np.ndarray,
    occupied_scores: np.ndarray,
    metric: str = "f1",
) -> tuple[float, float]:
    sweep = build_threshold_sweep(y_true, occupied_scores, split="valid")
    if metric == "accuracy":
        selected = sweep.sort_values("accuracy", ascending=False).iloc[0]
        return float(selected["threshold"]), float(selected["accuracy"])

    threshold, selection = select_threshold_from_sweep(sweep)
    return threshold, float(selection["selected_validation_f1"])


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    split: str,
    inference_seconds: float,
) -> dict[str, Any]:
    return {
        "split": split,
        "records": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "false_occupancy_rate": float(false_occupancy_rate(y_true, y_pred)),
        "inference_seconds": float(inference_seconds),
        "inference_speed_per_slot_sec": float(inference_seconds / max(1, len(y_true))),
    }


def save_confusion_matrix_csv(
    valid_matrix: np.ndarray,
    test_matrix: np.ndarray,
    output_path: Path,
) -> None:
    rows = []
    for split_name, matrix in [("valid", valid_matrix), ("test", test_matrix)]:
        for actual_index, actual_label in enumerate(LABEL_NAMES):
            for predicted_index, predicted_label in enumerate(LABEL_NAMES):
                rows.append(
                    {
                        "split": split_name,
                        "actual": actual_label,
                        "predicted": predicted_label,
                        "count": int(matrix[actual_index, predicted_index]),
                    }
                )
    pd.DataFrame(rows).to_csv(output_path, index=False)


def save_confusion_matrix_figure(matrix: np.ndarray, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    image = ax.imshow(matrix, cmap="Blues")
    ax.set_title("CNN Test Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_xticks(range(len(LABEL_NAMES)))
    ax.set_yticks(range(len(LABEL_NAMES)))
    ax.set_xticklabels(LABEL_NAMES)
    ax.set_yticklabels(LABEL_NAMES)

    max_value = matrix.max() if matrix.size else 0
    threshold = max_value / 2 if max_value else 0
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            color = "white" if matrix[row, col] > threshold else "black"
            ax.text(col, row, str(matrix[row, col]), ha="center", va="center", color=color)

    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def save_training_curves(history: list[dict[str, Any]], output_path: Path) -> None:
    if not history:
        return

    frame = pd.DataFrame(history)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(frame["epoch"], frame["train_loss"], label="train")
    axes[0].plot(frame["epoch"], frame["valid_loss"], label="valid")
    axes[0].set_title("CNN Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    axes[1].plot(frame["epoch"], frame["valid_f1"], label="valid F1", color="#2f855a")
    axes[1].plot(frame["epoch"], frame["valid_accuracy"], label="valid accuracy", color="#3182ce")
    axes[1].set_title("CNN Validation Metrics")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Score")
    axes[1].set_ylim(0, 1)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def save_threshold_sweep_figure(
    validation_sweep: pd.DataFrame,
    selected_threshold: float,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for metric, color in [
        ("accuracy", "#3182ce"),
        ("precision", "#805ad5"),
        ("recall", "#dd6b20"),
        ("f1_score", "#2f855a"),
    ]:
        ax.plot(validation_sweep["threshold"], validation_sweep[metric], label=metric, color=color)

    ax.axvline(selected_threshold, color="#2d3748", linestyle="--", linewidth=1.2, label="selected")
    ax.set_title("CNN Validation Threshold Sweep")
    ax.set_xlabel("Occupied probability threshold")
    ax.set_ylabel("Score")
    ax.set_ylim(0.85, 1.01)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def save_evaluation_outputs(
    output_dir: Path,
    figures_dir: Path,
    valid_metrics: dict[str, Any],
    test_metrics: dict[str, Any],
    valid_predictions: tuple[np.ndarray, np.ndarray],
    test_predictions: tuple[np.ndarray, np.ndarray],
    history: list[dict[str, Any]],
    extra: dict[str, Any] | None = None,
    threshold_sweep: pd.DataFrame | None = None,
    figure_prefix: str = "cnn",
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    y_valid, pred_valid = valid_predictions
    y_test, pred_test = test_predictions
    valid_matrix = confusion_matrix(y_valid, pred_valid, labels=LABELS)
    test_matrix = confusion_matrix(y_test, pred_test, labels=LABELS)

    pd.DataFrame([valid_metrics, test_metrics]).to_csv(
        output_dir / "cnn_metrics.csv",
        index=False,
    )
    (output_dir / "cnn_metrics.json").write_text(
        json.dumps(
            {
                "valid": valid_metrics,
                "test": test_metrics,
                "confusion_matrix_labels": LABEL_NAMES,
                "extra": extra or {},
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    report_text = "\n".join(
        [
            "CNN PKLot Slot Occupancy Classification Report",
            "",
            "Positive class: occupied",
            "",
            "Validation",
            classification_report(
                y_valid,
                pred_valid,
                labels=LABELS,
                target_names=LABEL_NAMES,
                zero_division=0,
            ),
            "",
            "Test",
            classification_report(
                y_test,
                pred_test,
                labels=LABELS,
                target_names=LABEL_NAMES,
                zero_division=0,
            ),
            "",
        ]
    )
    (output_dir / "classification_report.txt").write_text(report_text, encoding="utf-8")
    save_confusion_matrix_csv(valid_matrix, test_matrix, output_dir / "confusion_matrix.csv")
    save_confusion_matrix_figure(test_matrix, figures_dir / f"{figure_prefix}_confusion_matrix.png")
    save_training_curves(history, figures_dir / f"{figure_prefix}_training_curves.png")

    if threshold_sweep is not None:
        threshold_sweep.to_csv(output_dir / "threshold_sweep.csv", index=False)
        selected_threshold = float((extra or {}).get("decision_threshold", 0.5))
        validation_sweep = threshold_sweep.loc[threshold_sweep["split"] == "valid"]
        save_threshold_sweep_figure(
            validation_sweep,
            selected_threshold=selected_threshold,
            output_path=figures_dir / f"{figure_prefix}_threshold_sweep.png",
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a saved CNN checkpoint.")
    parser.add_argument("--checkpoint", type=Path, default=Path("models/cnn/best_cnn_model.pth"))
    parser.add_argument("--valid-manifest", type=Path, default=Path("data/splits/valid_slots_balanced_small.csv"))
    parser.add_argument("--test-manifest", type=Path, default=Path("data/splits/test_slots_balanced_small.csv"))
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=Path("results/metrics/cnn"))
    parser.add_argument("--figure-prefix", default="cnn")
    return parser.parse_args()


def _resolve_project_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def main() -> int:
    args = _parse_args()
    checkpoint_path = _resolve_project_path(args.checkpoint)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Missing CNN checkpoint: {checkpoint_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_version = checkpoint.get("model_version", "v1")
    dropout = float(checkpoint.get("dropout", 0.35 if model_version == "v1" else 0.3))
    model = build_model(num_classes=2, model_version=model_version, dropout=dropout).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    threshold = float(checkpoint.get("decision_threshold", 0.5))

    valid_loader = DataLoader(
        PKLotSlotDataset(args.valid_manifest, image_size=args.image_size, augment=False),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    test_loader = DataLoader(
        PKLotSlotDataset(args.test_manifest, image_size=args.image_size, augment=False),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    y_valid, pred_valid, valid_scores, valid_seconds = predict_loader_scores(
        model,
        valid_loader,
        device,
        threshold=threshold,
    )
    y_test, pred_test, test_scores, test_seconds = predict_loader_scores(
        model,
        test_loader,
        device,
        threshold=threshold,
    )
    valid_metrics = compute_metrics(y_valid, pred_valid, "valid", valid_seconds)
    test_metrics = compute_metrics(y_test, pred_test, "test", test_seconds)
    valid_sweep = build_threshold_sweep(y_valid, valid_scores, split="valid")
    test_sweep = build_threshold_sweep(y_test, test_scores, split="test")
    threshold_sweep = pd.concat([valid_sweep, test_sweep], ignore_index=True)
    output_dir = _resolve_project_path(args.output_dir)
    save_evaluation_outputs(
        output_dir=output_dir,
        figures_dir=FIGURES_DIR,
        valid_metrics=valid_metrics,
        test_metrics=test_metrics,
        valid_predictions=(y_valid, pred_valid),
        test_predictions=(y_test, pred_test),
        history=checkpoint.get("history", []),
        extra={
            "checkpoint": readable_relative_path(checkpoint_path, PROJECT_ROOT),
            "decision_threshold": threshold,
            "model_version": model_version,
            "threshold_note": "Threshold is applied to occupied probability.",
        },
        threshold_sweep=threshold_sweep,
        figure_prefix=args.figure_prefix,
    )
    print(f"CNN evaluation complete: {readable_relative_path(output_dir, PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

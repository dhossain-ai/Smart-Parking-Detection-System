from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC

try:
    from tqdm import tqdm
except ModuleNotFoundError:
    def tqdm(iterable, **_: object):
        return iterable

from src.classical.dataset import SlotRecord, load_manifest_records, targets_from_records
from src.classical.features import FeatureConfig, extract_features_from_record
from src.data.coco_utils import readable_relative_path
from src.utils.config import FIGURES_DIR, PROJECT_ROOT


DEFAULT_TRAIN_MANIFEST = Path("data/splits/train_slots_balanced_small.csv")
DEFAULT_VALID_MANIFEST = Path("data/splits/valid_slots_balanced_small.csv")
DEFAULT_TEST_MANIFEST = Path("data/splits/test_slots_balanced_small.csv")
DEFAULT_OUTPUT_MODEL = Path("models/classical/classical_lbp_hsv_hog_svm.joblib")
DEFAULT_OUTPUT_DIR = Path("results/metrics/classical")
LABELS = [0, 1]
LABEL_NAMES = ["vacant", "occupied"]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and evaluate a classical PKLot slot occupancy model."
    )
    parser.add_argument("--train-manifest", type=Path, default=DEFAULT_TRAIN_MANIFEST)
    parser.add_argument("--valid-manifest", type=Path, default=DEFAULT_VALID_MANIFEST)
    parser.add_argument("--test-manifest", type=Path, default=DEFAULT_TEST_MANIFEST)
    parser.add_argument(
        "--classifier",
        choices=["linear_svm", "rbf_svm", "random_forest"],
        default="linear_svm",
    )
    parser.add_argument("--feature-set", default="lbp_hsv_hog")
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--max-train-per-class", type=int, default=None)
    parser.add_argument("--max-valid-per-class", type=int, default=None)
    parser.add_argument("--max-test-per-class", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-model", type=Path, default=DEFAULT_OUTPUT_MODEL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def _resolve_project_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def _extract_feature_matrix(
    records: list[SlotRecord],
    split_name: str,
    config: FeatureConfig,
    feature_set: str,
) -> tuple[np.ndarray, np.ndarray, list[SlotRecord], int, float]:
    features: list[np.ndarray] = []
    targets: list[int] = []
    kept_records: list[SlotRecord] = []
    skipped = 0

    start = time.perf_counter()
    for record in tqdm(records, desc=f"Extracting {split_name} features", unit="slot"):
        vector = extract_features_from_record(
            record.as_feature_record(),
            config=config,
            feature_set=feature_set,
        )
        if vector is None:
            skipped += 1
            continue

        features.append(vector)
        targets.append(record.target)
        kept_records.append(record)

    elapsed = time.perf_counter() - start
    if not features:
        raise RuntimeError(f"No valid features extracted for {split_name}")

    return (
        np.vstack(features).astype(np.float32),
        np.asarray(targets, dtype=np.int64),
        kept_records,
        skipped,
        elapsed,
    )


def _build_classifier(classifier_name: str, seed: int) -> Pipeline | RandomForestClassifier:
    if classifier_name == "linear_svm":
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    LinearSVC(
                        C=1.0,
                        max_iter=10000,
                        class_weight=None,
                        random_state=seed,
                    ),
                ),
            ]
        )
    if classifier_name == "rbf_svm":
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    SVC(
                        C=3.0,
                        kernel="rbf",
                        gamma="scale",
                        class_weight=None,
                        random_state=seed,
                    ),
                ),
            ]
        )
    if classifier_name == "random_forest":
        return RandomForestClassifier(
            n_estimators=250,
            max_depth=None,
            min_samples_leaf=1,
            n_jobs=-1,
            random_state=seed,
            class_weight=None,
        )

    raise ValueError(f"Unsupported classifier: {classifier_name}")


def _false_occupancy_rate(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    vacant_mask = y_true == 0
    total_vacant = int(vacant_mask.sum())
    if total_vacant == 0:
        return 0.0
    false_occupied = int(((y_pred == 1) & vacant_mask).sum())
    return false_occupied / total_vacant


def _evaluate_split(
    model: Any,
    x: np.ndarray,
    y: np.ndarray,
    split_name: str,
    feature_seconds: float,
) -> tuple[dict[str, Any], np.ndarray, str]:
    start = time.perf_counter()
    predictions = model.predict(x)
    inference_seconds = time.perf_counter() - start

    metrics = {
        "split": split_name,
        "records": int(len(y)),
        "accuracy": float(accuracy_score(y, predictions)),
        "precision": float(precision_score(y, predictions, pos_label=1, zero_division=0)),
        "recall": float(recall_score(y, predictions, pos_label=1, zero_division=0)),
        "f1_score": float(f1_score(y, predictions, pos_label=1, zero_division=0)),
        "false_occupancy_rate": float(_false_occupancy_rate(y, predictions)),
        "inference_seconds": float(inference_seconds),
        "inference_speed_per_slot_sec": float(inference_seconds / max(1, len(y))),
        "feature_extraction_seconds": float(feature_seconds),
        "feature_extraction_speed_per_slot_sec": float(feature_seconds / max(1, len(y))),
    }
    matrix = confusion_matrix(y, predictions, labels=LABELS)
    report = classification_report(
        y,
        predictions,
        labels=LABELS,
        target_names=LABEL_NAMES,
        zero_division=0,
    )
    return metrics, matrix, report


def _save_confusion_matrix_csv(
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


def _save_confusion_matrix_figure(matrix: np.ndarray, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    image = ax.imshow(matrix, cmap="Blues")
    ax.set_title("Classical Model Test Confusion Matrix")
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


def train_classical(args: argparse.Namespace) -> int:
    output_model_path = _resolve_project_path(args.output_model)
    output_dir = _resolve_project_path(args.output_dir)
    figures_dir = FIGURES_DIR
    output_model_path.parent.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    feature_config = FeatureConfig(image_size=args.image_size)

    train_records = load_manifest_records(
        args.train_manifest,
        max_per_class=args.max_train_per_class,
        seed=args.seed,
    )
    valid_records = load_manifest_records(
        args.valid_manifest,
        max_per_class=args.max_valid_per_class,
        seed=args.seed,
    )
    test_records = load_manifest_records(
        args.test_manifest,
        max_per_class=args.max_test_per_class,
        seed=args.seed,
    )

    print("Classical training configuration")
    print(f"Feature set: {args.feature_set}")
    print(f"Classifier: {args.classifier}")
    print(f"Train records requested: {len(train_records)}")
    print(f"Valid records requested: {len(valid_records)}")
    print(f"Test records requested: {len(test_records)}")

    x_train, y_train, train_kept, train_skipped, train_feature_seconds = _extract_feature_matrix(
        train_records,
        "train",
        feature_config,
        args.feature_set,
    )
    x_valid, y_valid, valid_kept, valid_skipped, valid_feature_seconds = _extract_feature_matrix(
        valid_records,
        "valid",
        feature_config,
        args.feature_set,
    )
    x_test, y_test, test_kept, test_skipped, test_feature_seconds = _extract_feature_matrix(
        test_records,
        "test",
        feature_config,
        args.feature_set,
    )

    model = _build_classifier(args.classifier, seed=args.seed)
    train_start = time.perf_counter()
    model.fit(x_train, y_train)
    train_seconds = time.perf_counter() - train_start

    valid_metrics, valid_matrix, valid_report = _evaluate_split(
        model,
        x_valid,
        y_valid,
        "valid",
        valid_feature_seconds,
    )
    test_metrics, test_matrix, test_report = _evaluate_split(
        model,
        x_test,
        y_test,
        "test",
        test_feature_seconds,
    )

    for metrics in [valid_metrics, test_metrics]:
        metrics["classifier"] = args.classifier
        metrics["feature_set"] = args.feature_set
        metrics["image_size"] = args.image_size
        metrics["train_records_used"] = int(len(y_train))
        metrics["train_seconds"] = float(train_seconds)
        metrics["skipped_train_records"] = int(train_skipped)
        metrics["skipped_valid_records"] = int(valid_skipped)
        metrics["skipped_test_records"] = int(test_skipped)

    metrics_df = pd.DataFrame([valid_metrics, test_metrics])
    metrics_df.to_csv(output_dir / "classical_metrics.csv", index=False)
    (output_dir / "classical_metrics.json").write_text(
        json.dumps(
            {
                "valid": valid_metrics,
                "test": test_metrics,
                "confusion_matrix_labels": LABEL_NAMES,
                "feature_vector_length": int(x_train.shape[1]),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    report_text = "\n".join(
        [
            "Classical PKLot Slot Occupancy Classification Report",
            "",
            "Positive class: occupied",
            "",
            "Validation",
            valid_report,
            "",
            "Test",
            test_report,
            "",
        ]
    )
    (output_dir / "classification_report.txt").write_text(report_text, encoding="utf-8")
    _save_confusion_matrix_csv(
        valid_matrix,
        test_matrix,
        output_dir / "confusion_matrix.csv",
    )
    _save_confusion_matrix_figure(
        test_matrix,
        figures_dir / "classical_confusion_matrix.png",
    )

    joblib.dump(
        {
            "model": model,
            "feature_config": feature_config,
            "feature_set": args.feature_set,
            "label_to_target": {"vacant": 0, "occupied": 1},
            "target_to_label": {0: "vacant", 1: "occupied"},
            "classifier": args.classifier,
        },
        output_model_path,
    )

    print("Classical training complete")
    print(f"Model: {readable_relative_path(output_model_path, PROJECT_ROOT)}")
    print(f"Metrics: {readable_relative_path(output_dir / 'classical_metrics.csv', PROJECT_ROOT)}")
    print(f"Feature vector length: {x_train.shape[1]}")
    print(f"Train records used: {len(train_kept)} skipped: {train_skipped}")
    print(f"Valid records used: {len(valid_kept)} skipped: {valid_skipped}")
    print(f"Test records used: {len(test_kept)} skipped: {test_skipped}")
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
    return train_classical(_parse_args())


if __name__ == "__main__":
    raise SystemExit(main())

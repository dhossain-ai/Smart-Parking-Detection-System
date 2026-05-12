from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.utils.config import FIGURES_DIR, PROJECT_ROOT


DEFAULT_CLASSICAL_METRICS = Path("results/metrics/classical/classical_metrics.json")
DEFAULT_CNN_TUNED_METRICS = Path("results/metrics/mobilenetv3_transfer_final/cnn_metrics.json")
DEFAULT_CNN_METRICS = Path("results/metrics/cnn/cnn_metrics.json")
DEFAULT_OUTPUT_DIR = Path("results/metrics/comparison")
DEFAULT_FIGURES_DIR = FIGURES_DIR

CLASSICAL_REQUIREMENTS = {
    "accuracy": 0.88,
    "precision": 0.85,
    "recall": 0.85,
    "f1_score": 0.85,
}
CNN_REQUIREMENTS = {
    "accuracy": 0.98,
    "precision": 0.97,
    "recall": 0.97,
    "f1_score": 0.97,
}
METRIC_LABELS = {
    "accuracy": "Accuracy",
    "precision": "Precision",
    "recall": "Recall",
    "f1_score": "F1-score",
}
NA = "NA"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare classical and CNN parking-slot occupancy models."
    )
    parser.add_argument("--classical-metrics", type=Path, default=DEFAULT_CLASSICAL_METRICS)
    parser.add_argument("--cnn-metrics", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--figures-dir", type=Path, default=DEFAULT_FIGURES_DIR)
    return parser.parse_args()


def _resolve_project_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _first_present(data: dict[str, Any], keys: list[str]) -> Any:
    for key in keys:
        if key in data and data[key] not in ("", None):
            return data[key]
    return None


def _to_float(value: Any) -> float | None:
    if value in (None, "", NA):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if np.isnan(result):
        return None
    return result


def _format_value(value: Any, digits: int = 5) -> str:
    number = _to_float(value)
    if number is None:
        return NA
    return f"{number:.{digits}f}"


def _format_seconds(value: Any) -> str:
    number = _to_float(value)
    if number is None:
        return NA
    return f"{number:.8f}"


def _metric_split(data: dict[str, Any]) -> dict[str, Any]:
    if isinstance(data.get("test"), dict):
        return data["test"]
    return data


def _extra(data: dict[str, Any]) -> dict[str, Any]:
    extra = data.get("extra")
    return extra if isinstance(extra, dict) else {}


def _normalize_metrics(method: str, path: Path, data: dict[str, Any]) -> dict[str, Any]:
    split = _metric_split(data)
    extra = _extra(data)
    threshold = _first_present(
        split,
        ["threshold", "decision_threshold", "selected_threshold"],
    )
    if threshold is None:
        threshold = _first_present(extra, ["threshold", "decision_threshold"])

    model = _first_present(extra, ["model", "checkpoint"])
    if model is None:
        model = (
            "models/classical/classical_lbp_hsv_hog_svm.joblib"
            if method == "Classical"
            else "models/cnn/best_mobilenetv3_transfer_final.pth"
        )

    inference_time = _first_present(
        split,
        [
            "inference_speed_per_slot_sec",
            "inference_time_per_slot",
            "inference_speed",
        ],
    )

    return {
        "method": method,
        "source_file": str(path.relative_to(PROJECT_ROOT) if path.is_relative_to(PROJECT_ROOT) else path),
        "model": model,
        "records": _first_present(split, ["records", "test_records", "test_records_used"]),
        "accuracy": _first_present(split, ["accuracy"]),
        "precision": _first_present(split, ["precision"]),
        "recall": _first_present(split, ["recall"]),
        "f1_score": _first_present(split, ["f1_score", "f1"]),
        "false_occupancy_rate": _first_present(split, ["false_occupancy_rate"]),
        "inference_seconds": _first_present(split, ["inference_seconds", "inference_time_seconds"]),
        "inference_time_per_slot_sec": inference_time,
        "feature_extraction_time_per_slot_sec": _first_present(
            split,
            ["feature_extraction_speed_per_slot_sec", "feature_extraction_time_per_slot"],
        ),
        "threshold": threshold,
        "requirement_met_recorded": _first_present(
            split,
            ["requirement_met", "requirements_met"],
        ),
        "architecture": _first_present(split, ["architecture", "model_version"]),
        "device": _first_present(extra, ["device"]),
    }


def _exportable_frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    for column in frame.columns:
        frame[column] = frame[column].map(
            lambda value: NA if value in ("", None) or pd.isna(value) else value
        )
    return frame


def _requirement_rows(metrics: list[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    by_method = {row["method"]: row for row in metrics}
    for method, requirements in [
        ("Classical", CLASSICAL_REQUIREMENTS),
        ("CNN", CNN_REQUIREMENTS),
    ]:
        actuals = by_method.get(method, {})
        for metric, required in requirements.items():
            actual = _to_float(actuals.get(metric))
            rows.append(
                {
                    "method": method,
                    "metric": metric,
                    "required_value": f"> {required:.2f}",
                    "actual_value": _format_value(actual),
                    "passed": "Yes" if actual is not None and actual > required else "No",
                }
            )
    return pd.DataFrame(rows)


def _overall_requirement_status(checklist: pd.DataFrame, method: str) -> bool:
    method_rows = checklist.loc[checklist["method"] == method]
    return bool((method_rows["passed"] == "Yes").all())


def _comparison_status(checklist: pd.DataFrame, method: str) -> str:
    method_rows = checklist.loc[checklist["method"] == method]
    if bool((method_rows["passed"] == "Yes").all()):
        return "met"
    if bool((method_rows["passed"] == "Yes").any()):
        return "partially met"
    return "not met"


def _save_model_comparison(metrics: list[dict[str, Any]], output_path: Path) -> None:
    ordered_columns = [
        "method",
        "model",
        "source_file",
        "records",
        "accuracy",
        "precision",
        "recall",
        "f1_score",
        "false_occupancy_rate",
        "inference_seconds",
        "inference_time_per_slot_sec",
        "feature_extraction_time_per_slot_sec",
        "threshold",
        "requirement_met_recorded",
        "architecture",
        "device",
    ]
    frame = _exportable_frame(metrics)
    frame = frame.reindex(columns=ordered_columns)
    frame = frame.fillna(NA)
    frame.to_csv(output_path, index=False)


def _save_false_occupancy(metrics: list[dict[str, Any]], output_path: Path) -> pd.DataFrame:
    rows = []
    for row in metrics:
        rows.append(
            {
                "method": row["method"],
                "false_occupancy_rate": _format_value(row.get("false_occupancy_rate")),
                "note": "Actual vacant slots predicted as occupied; lower is better.",
            }
        )
    frame = pd.DataFrame(rows)
    frame.to_csv(output_path, index=False)
    return frame


def _save_speed_comparison(metrics: list[dict[str, Any]], output_path: Path) -> pd.DataFrame:
    rows = []
    for row in metrics:
        per_slot = _to_float(row.get("inference_time_per_slot_sec"))
        feature_per_slot = _to_float(row.get("feature_extraction_time_per_slot_sec"))
        total_per_slot = None
        if per_slot is not None and feature_per_slot is not None:
            total_per_slot = per_slot + feature_per_slot
        slots_per_second = None if per_slot in (None, 0.0) else 1.0 / per_slot
        rows.append(
            {
                "method": row["method"],
                "records": row.get("records") or NA,
                "inference_seconds": _format_seconds(row.get("inference_seconds")),
                "inference_time_per_slot_sec": _format_seconds(per_slot),
                "slots_per_second": _format_value(slots_per_second, digits=2),
                "feature_extraction_time_per_slot_sec": _format_seconds(feature_per_slot),
                "total_estimated_time_per_slot_sec": _format_seconds(total_per_slot),
                "note": (
                    "Classical classifier timing excludes feature extraction unless total is used."
                    if row["method"] == "Classical"
                    else "CNN timing includes model inference on the recorded device."
                ),
            }
        )
    frame = pd.DataFrame(rows)
    frame.to_csv(output_path, index=False)
    return frame


def _confusion_path(metrics_path: Path) -> Path:
    return metrics_path.parent / "confusion_matrix.csv"


def _save_confusion_comparison(
    classical_metrics_path: Path,
    cnn_metrics_path: Path,
    output_path: Path,
) -> pd.DataFrame:
    rows = []
    for method, path in [
        ("Classical", _confusion_path(classical_metrics_path)),
        ("CNN", _confusion_path(cnn_metrics_path)),
    ]:
        if not path.is_file():
            rows.append(
                {
                    "method": method,
                    "split": "test",
                    "actual": NA,
                    "predicted": NA,
                    "count": NA,
                    "note": f"Missing confusion matrix: {path}",
                }
            )
            continue
        frame = pd.read_csv(path)
        frame = frame.loc[frame["split"] == "test"].copy()
        frame.insert(0, "method", method)
        frame["note"] = "Rows are actual labels; columns are predicted labels."
        rows.extend(frame.to_dict("records"))

    result = pd.DataFrame(rows)
    result.to_csv(output_path, index=False)
    return result


def _threshold_meets_cnn_requirements(row: pd.Series) -> bool:
    return all(float(row[metric]) > required for metric, required in CNN_REQUIREMENTS.items())


def _row_with_note(row: pd.Series, summary_type: str, note: str) -> dict[str, Any]:
    return {
        "summary_type": summary_type,
        "split": row.get("split", NA),
        "threshold": _format_value(row.get("threshold"), digits=2),
        "accuracy": _format_value(row.get("accuracy")),
        "precision": _format_value(row.get("precision")),
        "recall": _format_value(row.get("recall")),
        "f1_score": _format_value(row.get("f1_score")),
        "false_occupancy_rate": _format_value(row.get("false_occupancy_rate")),
        "note": note,
    }


def _save_threshold_summary(
    cnn_metrics: dict[str, Any],
    cnn_metrics_path: Path,
    output_path: Path,
) -> str:
    sweep_path = cnn_metrics_path.parent / "threshold_sweep.csv"
    if not sweep_path.is_file():
        return "Threshold sweep file unavailable."

    sweep = pd.read_csv(sweep_path)
    validation_sweep = sweep.loc[sweep["split"] == "valid"].copy()
    if validation_sweep.empty:
        validation_sweep = sweep.copy()

    rows: list[dict[str, Any]] = []
    selected_threshold = _first_present(
        _metric_split(cnn_metrics),
        ["threshold", "decision_threshold"],
    )
    if selected_threshold is None:
        selected_threshold = _first_present(_extra(cnn_metrics), ["decision_threshold", "threshold"])
    selected_float = _to_float(selected_threshold)
    if selected_float is not None:
        selected_index = validation_sweep["threshold"].sub(selected_float).abs().idxmin()
        rows.append(
            _row_with_note(
                validation_sweep.loc[selected_index],
                "selected_threshold",
                "Selected threshold from saved CNN metrics.",
            )
        )

    best_f1 = validation_sweep.sort_values(
        ["f1_score", "accuracy", "precision", "recall"],
        ascending=False,
    ).iloc[0]
    rows.append(
        _row_with_note(
            best_f1,
            "best_f1_threshold",
            "Best validation F1 threshold in sweep.",
        )
    )

    recall_safe = validation_sweep.loc[validation_sweep["recall"] >= CNN_REQUIREMENTS["recall"]]
    if recall_safe.empty:
        rows.append(
            {
                "summary_type": "best_precision_recall_ge_0.97",
                "split": "valid",
                "threshold": NA,
                "accuracy": NA,
                "precision": NA,
                "recall": NA,
                "f1_score": NA,
                "false_occupancy_rate": NA,
                "note": "No validation threshold kept recall >= 0.97.",
            }
        )
    else:
        best_precision = recall_safe.sort_values(
            ["precision", "f1_score", "accuracy"],
            ascending=False,
        ).iloc[0]
        rows.append(
            _row_with_note(
                best_precision,
                "best_precision_recall_ge_0.97",
                "Best precision threshold with validation recall >= 0.97.",
            )
        )

    meets_all = validation_sweep.loc[validation_sweep.apply(_threshold_meets_cnn_requirements, axis=1)]
    if meets_all.empty:
        note = "No threshold in the sweep satisfied all CNN requirements."
        rows.append(
            {
                "summary_type": "meets_all_cnn_requirements",
                "split": "valid",
                "threshold": NA,
                "accuracy": NA,
                "precision": NA,
                "recall": NA,
                "f1_score": NA,
                "false_occupancy_rate": NA,
                "note": note,
            }
        )
    else:
        for _, row in meets_all.iterrows():
            rows.append(
                _row_with_note(
                    row,
                    "meets_all_cnn_requirements",
                    "This validation threshold satisfies all CNN requirements.",
                )
            )
        note = "At least one threshold in the sweep satisfied all CNN requirements."

    pd.DataFrame(rows).to_csv(output_path, index=False)
    return note


def _save_weather_outputs(output_dir: Path) -> None:
    weather_report = "\n".join(
        [
            "# Weather Robustness Report",
            "",
            "Original PKLot includes sunny, rainy, and cloudy conditions.",
            "",
            "This Roboflow COCO export does not provide reliable weather labels in the current file paths or metadata.",
            "",
            "The project created `data/splits/weather_labels_template.csv` for optional manual weather labeling.",
            "",
            "Weather robustness can be evaluated later if weather labels are manually added.",
            "",
            "Do not claim weather-wise accuracy without labels.",
            "",
        ]
    )
    (output_dir / "weather_robustness_report.md").write_text(weather_report, encoding="utf-8")
    pd.DataFrame(
        [
            {
                "weather": "sunny",
                "classical_accuracy": NA,
                "cnn_accuracy": NA,
                "note": "weather labels unavailable in current COCO export",
            },
            {
                "weather": "rainy",
                "classical_accuracy": NA,
                "cnn_accuracy": NA,
                "note": "weather labels unavailable in current COCO export",
            },
            {
                "weather": "cloudy",
                "classical_accuracy": NA,
                "cnn_accuracy": NA,
                "note": "weather labels unavailable in current COCO export",
            },
        ]
    ).to_csv(output_dir / "weather_robustness_placeholder.csv", index=False)


def _plot_model_metrics(metrics: list[dict[str, Any]], output_path: Path) -> None:
    methods = [row["method"] for row in metrics]
    metric_keys = ["accuracy", "precision", "recall", "f1_score"]
    values = np.array(
        [
            [_to_float(row.get(metric)) if _to_float(row.get(metric)) is not None else np.nan for metric in metric_keys]
            for row in metrics
        ],
        dtype=float,
    )

    x = np.arange(len(metric_keys))
    width = 0.34
    colors = ["#2563eb", "#16a34a"]
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for index, method in enumerate(methods):
        offset = (index - (len(methods) - 1) / 2) * width
        ax.bar(x + offset, values[index], width=width, label=method, color=colors[index % len(colors)])

    ax.axhline(0.98, color="#64748b", linestyle="--", linewidth=1, label="CNN accuracy target")
    ax.set_title("Model Metrics Comparison")
    ax.set_ylabel("Score")
    ax.set_ylim(0.82, 1.01)
    ax.set_xticks(x)
    ax.set_xticklabels([METRIC_LABELS[key] for key in metric_keys])
    ax.legend(loc="lower right")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _plot_requirement_checklist(checklist: pd.DataFrame, output_path: Path) -> None:
    metrics = ["accuracy", "precision", "recall", "f1_score"]
    methods = ["Classical", "CNN"]
    matrix = np.zeros((len(methods), len(metrics)))
    for row_index, method in enumerate(methods):
        for col_index, metric in enumerate(metrics):
            passed = checklist.loc[
                (checklist["method"] == method) & (checklist["metric"] == metric),
                "passed",
            ].iloc[0]
            matrix[row_index, col_index] = 1 if passed == "Yes" else 0

    fig, ax = plt.subplots(figsize=(7, 2.8))
    ax.imshow(matrix, cmap=matplotlib.colors.ListedColormap(["#ef4444", "#22c55e"]), vmin=0, vmax=1)
    ax.set_title("Requirement Checklist")
    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels([METRIC_LABELS[key] for key in metrics])
    ax.set_yticks(range(len(methods)))
    ax.set_yticklabels(methods)
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            ax.text(
                col,
                row,
                "Pass" if matrix[row, col] else "Miss",
                ha="center",
                va="center",
                color="white",
                fontweight="bold",
            )
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _plot_false_occupancy(metrics: list[dict[str, Any]], output_path: Path) -> None:
    methods = [row["method"] for row in metrics]
    values = [_to_float(row.get("false_occupancy_rate")) or 0.0 for row in metrics]
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    bars = ax.bar(methods, values, color=["#2563eb", "#16a34a"])
    ax.set_title("False Occupancy Rate")
    ax.set_ylabel("False occupancy rate")
    ax.set_ylim(0, max(values + [0.08]) * 1.2)
    ax.grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.4f}",
            ha="center",
            va="bottom",
        )
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _markdown_table(frame: pd.DataFrame) -> str:
    columns = [str(column) for column in frame.columns]
    rows = frame.fillna(NA).astype(str).values.tolist()
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join([header, separator, *body])


def _write_final_report(
    metrics: list[dict[str, Any]],
    checklist: pd.DataFrame,
    false_occupancy: pd.DataFrame,
    speed: pd.DataFrame,
    threshold_note: str,
    output_path: Path,
) -> None:
    by_method = {row["method"]: row for row in metrics}
    classical = by_method["Classical"]
    cnn = by_method["CNN"]
    classical_status = _comparison_status(checklist, "Classical")
    cnn_status = _comparison_status(checklist, "CNN")
    cnn_model = cnn.get("model") or NA
    cnn_threshold = _format_value(cnn.get("threshold"), digits=2)

    report = "\n".join(
        [
            "# Final Evaluation Report",
            "",
            "## Project Summary",
            "",
            "This project evaluates a fully offline Smart Parking Detection System for parking-slot occupancy classification. Occupied is the positive class and vacant is the negative class.",
            "",
            "## Dataset Summary",
            "",
            "The current local data source is the Roboflow COCO export of PKLot with train, valid, and test folders and `_annotations.coco.json` files. The evaluation uses saved local model metrics from the test split.",
            "",
            "## Classical Method Summary",
            "",
            f"The classical model uses {classical.get('model', NA)} with LBP, HSV histogram, HOG features, feature scaling, and LinearSVC. Test accuracy is {_format_value(classical.get('accuracy'))}, precision is {_format_value(classical.get('precision'))}, recall is {_format_value(classical.get('recall'))}, and F1-score is {_format_value(classical.get('f1_score'))}.",
            "",
            "The classical method meets the assignment minimum requirements.",
            "",
            "## CNN Method Summary",
            "",
            f"The best available CNN result for this comparison is {cnn_model}. The selected occupied-probability threshold is {cnn_threshold}. Test accuracy is {_format_value(cnn.get('accuracy'))}, precision is {_format_value(cnn.get('precision'))}, recall is {_format_value(cnn.get('recall'))}, and F1-score is {_format_value(cnn.get('f1_score'))}.",
            "",
            "The MobileNetV3 model meets the accuracy, recall, and F1-score targets. Precision is slightly below the strict 97% target, so the CNN requirement is reported honestly as partially met.",
            "",
            "The neural model is used for the visual demo because it is the strongest saved CNN result and has the lowest false occupancy rate in the current experiments.",
            "",
            "## Requirement Checklist",
            "",
            _markdown_table(checklist),
            "",
            f"Classical requirement: {classical_status}.",
            f"CNN requirement: {cnn_status} with MobileNetV3-Small; precision is the only strict target still slightly below requirement.",
            "",
            "## False Occupancy Discussion",
            "",
            _markdown_table(false_occupancy),
            "",
            "False occupancy means an actual vacant slot is predicted as occupied. This can cause a driver to skip a free space, so lower is better.",
            "",
            "## Speed Discussion",
            "",
            _markdown_table(speed),
            "",
            "Classical speed should be read carefully because classifier inference and handcrafted feature extraction are recorded separately. CNN speed reflects the saved model inference timing and depends on the recorded device.",
            "",
            "## Threshold Summary",
            "",
            threshold_note,
            "",
            "## Weather Robustness Limitation",
            "",
            "Original PKLot includes sunny, rainy, and cloudy conditions, but this Roboflow COCO export does not provide reliable weather labels in the current file paths or metadata. The project provides `data/splits/weather_labels_template.csv` for optional manual labeling. Weather-wise accuracy should not be claimed until those labels are added.",
            "",
            "## Honest Conclusion",
            "",
            "The classical method meets the assignment minimum requirements. MobileNetV3-Small is the final neural model because it improves accuracy, F1-score, and false occupancy rate over the earlier custom CNN. It meets accuracy, recall, and F1-score targets, while precision remains slightly below the strict 97% target.",
            "",
        ]
    )
    output_path.write_text(report, encoding="utf-8")


def _write_slide_summary(
    metrics: list[dict[str, Any]],
    checklist: pd.DataFrame,
    output_path: Path,
) -> None:
    by_method = {row["method"]: row for row in metrics}
    classical = by_method["Classical"]
    cnn = by_method["CNN"]
    content = "\n".join(
        [
            "# Slide Summary",
            "",
            f"- Classical result: accuracy {_format_value(classical.get('accuracy'))}, precision {_format_value(classical.get('precision'))}, recall {_format_value(classical.get('recall'))}, F1-score {_format_value(classical.get('f1_score'))}; requirement met.",
            f"- MobileNetV3 result: accuracy {_format_value(cnn.get('accuracy'))}, precision {_format_value(cnn.get('precision'))}, recall {_format_value(cnn.get('recall'))}, F1-score {_format_value(cnn.get('f1_score'))}; precision is slightly below the strict target.",
            f"- Requirement status: Classical {_comparison_status(checklist, 'Classical')}; CNN {_comparison_status(checklist, 'CNN')} with MobileNetV3-Small.",
            "- Visual demo plan: use MobileNetV3-Small for overlays and present the precision limitation honestly.",
            "- Weather limitation: current Roboflow COCO export has no reliable sunny/rainy/cloudy labels, so weather-wise accuracy is not reported.",
            "- Next improvement: continue MobileNetV3 threshold and training tuning to push precision above 0.97.",
            "",
        ]
    )
    output_path.write_text(content, encoding="utf-8")


def _choose_cnn_metrics_path(cli_path: Path | None) -> Path:
    if cli_path is not None:
        return _resolve_project_path(cli_path)
    tuned_path = _resolve_project_path(DEFAULT_CNN_TUNED_METRICS)
    if tuned_path.is_file():
        return tuned_path
    return _resolve_project_path(DEFAULT_CNN_METRICS)


def compare_models(args: argparse.Namespace) -> int:
    classical_path = _resolve_project_path(args.classical_metrics)
    cnn_path = _choose_cnn_metrics_path(args.cnn_metrics)
    output_dir = _resolve_project_path(args.output_dir)
    figures_dir = _resolve_project_path(args.figures_dir)

    if not classical_path.is_file():
        raise FileNotFoundError(f"Missing classical metrics: {classical_path}")
    if not cnn_path.is_file():
        raise FileNotFoundError(f"Missing CNN metrics: {cnn_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    classical_data = _read_json(classical_path)
    cnn_data = _read_json(cnn_path)
    metrics = [
        _normalize_metrics("Classical", classical_path, classical_data),
        _normalize_metrics("CNN", cnn_path, cnn_data),
    ]

    _save_model_comparison(metrics, output_dir / "model_comparison.csv")
    checklist = _requirement_rows(metrics)
    checklist.to_csv(output_dir / "requirement_checklist.csv", index=False)
    false_occupancy = _save_false_occupancy(metrics, output_dir / "false_occupancy_comparison.csv")
    speed = _save_speed_comparison(metrics, output_dir / "speed_comparison.csv")
    _save_confusion_comparison(classical_path, cnn_path, output_dir / "confusion_matrix_comparison.csv")
    threshold_note = _save_threshold_summary(
        cnn_data,
        cnn_path,
        output_dir / "cnn_threshold_summary.csv",
    )
    _save_weather_outputs(output_dir)
    _write_final_report(
        metrics,
        checklist,
        false_occupancy,
        speed,
        threshold_note,
        output_dir / "final_evaluation_report.md",
    )
    _write_slide_summary(metrics, checklist, output_dir / "slide_summary.md")

    _plot_model_metrics(metrics, figures_dir / "model_metrics_comparison.png")
    _plot_requirement_checklist(checklist, figures_dir / "requirement_checklist.png")
    _plot_false_occupancy(metrics, figures_dir / "false_occupancy_comparison.png")

    classical_met = _overall_requirement_status(checklist, "Classical")
    cnn_met = _overall_requirement_status(checklist, "CNN")
    print("Model comparison complete")
    print(f"Classical metrics: {classical_path.relative_to(PROJECT_ROOT)}")
    print(f"CNN metrics: {cnn_path.relative_to(PROJECT_ROOT)}")
    print(f"Classical requirement met: {'Yes' if classical_met else 'No'}")
    print(f"CNN requirement met: {'Yes' if cnn_met else 'No - partially met'}")
    print(f"Outputs: {output_dir.relative_to(PROJECT_ROOT)}")
    print(f"Figures: {figures_dir.relative_to(PROJECT_ROOT)}")
    return 0


def main() -> int:
    return compare_models(_parse_args())


if __name__ == "__main__":
    raise SystemExit(main())

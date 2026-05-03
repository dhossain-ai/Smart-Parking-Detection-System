from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import joblib
import numpy as np
import pandas as pd
import torch

from src.classical.dataset import SlotRecord, load_manifest_records
from src.classical.features import (
    FeatureConfig,
    crop_slot_image,
    extract_combined_features,
    resolve_image_path,
)
from src.neural.dataset import PKLotSlotDataset
from src.neural.model import build_model
from src.utils.config import PROJECT_ROOT, RESULT_IMAGES_DIR
from src.visualization.draw_overlays import (
    create_side_by_side,
    draw_legend,
    draw_slot_overlays,
    draw_summary_panel,
)


DEFAULT_MANIFEST = Path("data/splits/test_slots.csv")
DEFAULT_CNN_MODEL = Path("models/cnn/best_cnn_model_v2.pth")
DEFAULT_CLASSICAL_MODEL = Path("models/classical/classical_lbp_hsv_hog_svm.joblib")
DEFAULT_CNN_METRICS = Path("results/metrics/cnn_tuned/cnn_metrics.json")
DEFAULT_OUTPUT_DIR = Path("results/metrics/demo_image")
LABEL_BY_TARGET = {0: "vacant", 1: "occupied"}


class CNNPredictor:
    def __init__(self, model_path: Path, threshold: float) -> None:
        self.model_path = model_path
        self.threshold = threshold
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.model_version = _load_cnn_model(model_path, self.device)
        self.model_name = f"CNN {self.model_version}"

    @torch.no_grad()
    def predict(self, records: list[SlotRecord], image_rgb: np.ndarray) -> list[dict[str, Any]]:
        crops = [_crop_record(image_rgb, record, image_size=64) for record in records]
        batch = torch.stack([_tensor_from_crop(crop) for crop in crops], dim=0).to(self.device)
        logits = self.model(batch)
        probabilities = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()

        predictions = []
        for slot_index, (record, occupied_probability) in enumerate(zip(records, probabilities), start=1):
            label = "occupied" if float(occupied_probability) >= self.threshold else "vacant"
            predictions.append(
                _prediction_row(
                    slot_index=slot_index,
                    record=record,
                    predicted_label=label,
                    occupied_probability=float(occupied_probability),
                    score=None,
                    threshold=self.threshold,
                    model_type="cnn",
                )
            )
        return predictions


class ClassicalPredictor:
    def __init__(self, model_path: Path) -> None:
        self.model_path = model_path
        self.model, self.feature_config, self.feature_set = _load_classical_model(model_path)
        self.model_name = "Classical LBP+HSV+HOG"

    def predict(self, records: list[SlotRecord], image_rgb: np.ndarray) -> list[dict[str, Any]]:
        predictions = []
        for slot_index, record in enumerate(records, start=1):
            crop = _crop_record(image_rgb, record, image_size=self.feature_config.image_size)
            vector = extract_combined_features(crop, config=self.feature_config, feature_set=self.feature_set)
            features = vector.reshape(1, -1)
            target = int(self.model.predict(features)[0])
            label = LABEL_BY_TARGET.get(target, "occupied" if target == 1 else "vacant")
            predictions.append(
                _prediction_row(
                    slot_index=slot_index,
                    record=record,
                    predicted_label=label,
                    occupied_probability=_probability_for_occupied(self.model, features),
                    score=_score_for_occupied(self.model, features),
                    threshold=None,
                    model_type="classical",
                )
            )
        return predictions


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create an annotated smart-parking image demo from known slot boxes."
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--image-path", type=Path, default=None)
    parser.add_argument("--model-type", choices=["cnn", "classical"], default="cnn")
    parser.add_argument("--cnn-model", type=Path, default=DEFAULT_CNN_MODEL)
    parser.add_argument("--classical-model", type=Path, default=DEFAULT_CLASSICAL_MODEL)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--output-image", type=Path, default=None)
    parser.add_argument("--output-side-by-side", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-labels", type=int, default=80)
    return parser.parse_args()


def _resolve_project_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def _read_threshold_from_metrics(default: float = 0.48) -> float:
    metrics_path = _resolve_project_path(DEFAULT_CNN_METRICS)
    if not metrics_path.is_file():
        return default
    try:
        data = json.loads(metrics_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return default

    for section in [data.get("test", {}), data.get("valid", {}), data.get("extra", {})]:
        if not isinstance(section, dict):
            continue
        for key in ["decision_threshold", "threshold"]:
            value = section.get(key)
            if value not in (None, ""):
                return float(value)
    return default


def read_cnn_threshold(default: float = 0.48) -> float:
    return _read_threshold_from_metrics(default=default)


def _default_output_image(model_type: str) -> Path:
    return RESULT_IMAGES_DIR / f"demo_image_{model_type}_output.jpg"


def _default_side_by_side(model_type: str) -> Path:
    return RESULT_IMAGES_DIR / f"demo_image_{model_type}_side_by_side.jpg"


def _relative_or_absolute(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def _select_records(
    records: list[SlotRecord],
    image_path: Path | None,
) -> tuple[str, Path, list[SlotRecord]]:
    if image_path is not None:
        requested_path = resolve_image_path(image_path)
        requested_name = image_path.name
        matches = []
        for record in records:
            record_path = resolve_image_path(record.image_path)
            if (
                record_path == requested_path
                or Path(record.image_path).as_posix() == image_path.as_posix()
                or record.file_name == requested_name
            ):
                matches.append(record)
        if not matches:
            raise ValueError(f"No slot annotations found for image: {image_path}")
        return matches[0].image_path, resolve_image_path(matches[0].image_path), matches

    counts = Counter(record.image_path for record in records)
    for selected_path, _ in counts.most_common():
        resolved_path = resolve_image_path(selected_path)
        if resolved_path.is_file():
            selected = [record for record in records if record.image_path == selected_path]
            return selected_path, resolved_path, selected

    raise FileNotFoundError("No manifest image paths could be resolved locally.")


def prepare_image_records(
    records: list[SlotRecord],
    image_path: Path | None = None,
) -> tuple[str, Path, list[SlotRecord]]:
    return _select_records(records, image_path)


def _load_image_pair(image_path: Path) -> tuple[np.ndarray, np.ndarray]:
    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    return image_bgr, image_rgb


def load_image_pair(image_path: Path) -> tuple[np.ndarray, np.ndarray]:
    return _load_image_pair(image_path)


def _zero_crop(image_size: int) -> np.ndarray:
    return np.zeros((image_size, image_size, 3), dtype=np.uint8)


def _crop_record(image_rgb: np.ndarray, record: SlotRecord, image_size: int = 64) -> np.ndarray:
    crop = crop_slot_image(
        image_rgb,
        x1=record.x1,
        y1=record.y1,
        x2=record.x2,
        y2=record.y2,
    )
    if crop is None or crop.size == 0:
        return _zero_crop(image_size)
    return cv2.resize(crop, (image_size, image_size), interpolation=cv2.INTER_AREA)


def _tensor_from_crop(crop_rgb: np.ndarray) -> torch.Tensor:
    return PKLotSlotDataset._to_tensor(crop_rgb)


def _clean_state_dict(state_dict: dict[str, Any]) -> dict[str, Any]:
    cleaned = {}
    for key, value in state_dict.items():
        cleaned[key.removeprefix("module.")] = value
    return cleaned


def _checkpoint_state(checkpoint: Any) -> tuple[dict[str, Any], str | None, float | None]:
    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            return checkpoint["model_state_dict"], checkpoint.get("model_version"), checkpoint.get("dropout")
        if "state_dict" in checkpoint:
            return checkpoint["state_dict"], checkpoint.get("model_version"), checkpoint.get("dropout")
        if checkpoint and all(torch.is_tensor(value) for value in checkpoint.values()):
            return checkpoint, None, None
    raise ValueError("Unsupported CNN checkpoint format.")


def _load_cnn_model(model_path: Path, device: torch.device) -> tuple[torch.nn.Module, str]:
    checkpoint = torch.load(model_path, map_location=device)
    state_dict, checkpoint_version, checkpoint_dropout = _checkpoint_state(checkpoint)
    state_dict = _clean_state_dict(state_dict)

    versions = []
    if checkpoint_version:
        versions.append(str(checkpoint_version))
    if "v2" not in versions:
        versions.append("v2")
    if "v1" not in versions:
        versions.append("v1")

    last_error: Exception | None = None
    for version in versions:
        dropout = float(checkpoint_dropout) if checkpoint_dropout is not None else (0.3 if version == "v2" else 0.35)
        model = build_model(num_classes=2, model_version=version, dropout=dropout).to(device)
        try:
            model.load_state_dict(state_dict)
        except RuntimeError as error:
            last_error = error
            continue
        model.eval()
        return model, version

    raise RuntimeError(f"Could not load CNN checkpoint: {last_error}")


def load_cnn_predictor(model_path: Path, threshold: float) -> CNNPredictor:
    return CNNPredictor(model_path=model_path, threshold=threshold)


def load_classical_predictor(model_path: Path) -> ClassicalPredictor:
    return ClassicalPredictor(model_path=model_path)


def predict_slots_for_image(
    records: list[SlotRecord],
    image_rgb: np.ndarray,
    predictor: CNNPredictor | ClassicalPredictor,
) -> tuple[list[dict[str, Any]], str]:
    return predictor.predict(records, image_rgb), predictor.model_name


@torch.no_grad()
def _predict_cnn(
    records: list[SlotRecord],
    image_rgb: np.ndarray,
    model_path: Path,
    threshold: float,
) -> tuple[list[dict[str, Any]], str]:
    return predict_slots_for_image(records, image_rgb, load_cnn_predictor(model_path, threshold))


def _load_classical_model(model_path: Path) -> tuple[Any, FeatureConfig, str]:
    payload = joblib.load(model_path)
    if isinstance(payload, dict):
        model = payload.get("model")
        feature_config = payload.get("feature_config") or FeatureConfig()
        feature_set = payload.get("feature_set") or "lbp_hsv_hog"
    else:
        model = payload
        feature_config = FeatureConfig()
        feature_set = "lbp_hsv_hog"
    if model is None:
        raise ValueError(f"Classical model file did not contain a model: {model_path}")
    return model, feature_config, feature_set


def _score_for_occupied(model: Any, features: np.ndarray) -> float | None:
    if not hasattr(model, "decision_function"):
        return None
    scores = model.decision_function(features)
    scores_array = np.asarray(scores)
    if scores_array.ndim == 0:
        return float(scores_array)
    if scores_array.ndim == 1:
        return float(scores_array[0])
    classes = getattr(model, "classes_", None)
    if classes is not None and 1 in list(classes):
        occupied_index = list(classes).index(1)
        return float(scores_array[0, occupied_index])
    return float(scores_array[0, -1])


def _probability_for_occupied(model: Any, features: np.ndarray) -> float | None:
    if not hasattr(model, "predict_proba"):
        return None
    probabilities = np.asarray(model.predict_proba(features))
    classes = getattr(model, "classes_", None)
    if classes is not None and 1 in list(classes):
        occupied_index = list(classes).index(1)
    else:
        occupied_index = probabilities.shape[1] - 1
    return float(probabilities[0, occupied_index])


def _predict_classical(
    records: list[SlotRecord],
    image_rgb: np.ndarray,
    model_path: Path,
) -> tuple[list[dict[str, Any]], str]:
    return predict_slots_for_image(records, image_rgb, load_classical_predictor(model_path))


def _prediction_row(
    slot_index: int,
    record: SlotRecord,
    predicted_label: str,
    occupied_probability: float | None,
    score: float | None,
    threshold: float | None,
    model_type: str,
) -> dict[str, Any]:
    return {
        "slot_index": slot_index,
        "image_path": record.image_path,
        "x1": record.x1,
        "y1": record.y1,
        "x2": record.x2,
        "y2": record.y2,
        "predicted_label": predicted_label,
        "occupied_probability": occupied_probability if occupied_probability is not None else "NA",
        "score": score if score is not None else "NA",
        "threshold": threshold if threshold is not None else "NA",
        "model_type": model_type,
    }


def _summary(
    model_type: str,
    image_path: str,
    predictions: list[dict[str, Any]],
    threshold: float | None,
    output_image: Path,
    output_side_by_side: Path,
) -> dict[str, Any]:
    occupied_count = sum(row["predicted_label"] == "occupied" for row in predictions)
    vacant_count = sum(row["predicted_label"] == "vacant" for row in predictions)
    total_slots = len(predictions)
    return {
        "model_type": model_type,
        "image_path": image_path,
        "total_slots": total_slots,
        "occupied_count": occupied_count,
        "vacant_count": vacant_count,
        "occupancy_rate": occupied_count / max(1, total_slots),
        "threshold": threshold if threshold is not None else "NA",
        "output_image": _relative_or_absolute(output_image),
        "output_side_by_side": _relative_or_absolute(output_side_by_side),
    }


def _write_outputs(
    predictions: list[dict[str, Any]],
    summary: dict[str, Any],
    output_dir: Path,
    model_type: str,
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    canonical_predictions = output_dir / "demo_image_predictions.csv"
    canonical_summary = output_dir / "demo_image_summary.json"
    model_predictions = output_dir / f"demo_image_{model_type}_predictions.csv"
    model_summary = output_dir / f"demo_image_{model_type}_summary.json"

    frame = pd.DataFrame(predictions)
    frame.to_csv(canonical_predictions, index=False)
    frame.to_csv(model_predictions, index=False)
    text = json.dumps(summary, indent=2)
    canonical_summary.write_text(text, encoding="utf-8")
    model_summary.write_text(text, encoding="utf-8")
    return canonical_predictions, canonical_summary


def run_demo(args: argparse.Namespace) -> int:
    manifest_path = _resolve_project_path(args.manifest)
    records = load_manifest_records(manifest_path)
    if not records:
        raise ValueError(f"No valid slot records found in manifest: {manifest_path}")

    selected_image_path, resolved_image_path, selected_records = _select_records(records, args.image_path)
    original_bgr, image_rgb = _load_image_pair(resolved_image_path)

    output_image = _resolve_project_path(args.output_image) if args.output_image else _default_output_image(args.model_type)
    output_side_by_side = (
        _resolve_project_path(args.output_side_by_side)
        if args.output_side_by_side
        else _default_side_by_side(args.model_type)
    )
    output_dir = _resolve_project_path(args.output_dir)
    output_image.parent.mkdir(parents=True, exist_ok=True)
    output_side_by_side.parent.mkdir(parents=True, exist_ok=True)

    if args.model_type == "cnn":
        threshold = args.threshold if args.threshold is not None else _read_threshold_from_metrics()
        model_path = _resolve_project_path(args.cnn_model)
        predictions, model_name = _predict_cnn(selected_records, image_rgb, model_path, threshold)
    else:
        threshold = None
        model_path = _resolve_project_path(args.classical_model)
        predictions, model_name = _predict_classical(selected_records, image_rgb, model_path)

    summary = _summary(
        model_type=args.model_type,
        image_path=selected_image_path,
        predictions=predictions,
        threshold=threshold,
        output_image=output_image,
        output_side_by_side=output_side_by_side,
    )

    processed = draw_slot_overlays(original_bgr, predictions, max_labels=args.max_labels)
    processed = draw_legend(processed)
    processed = draw_summary_panel(processed, summary, model_name=model_name)
    side_by_side = create_side_by_side(
        original_bgr,
        processed,
        left_title="Original test image",
        right_title=f"{model_name} predictions",
    )
    cv2.imwrite(str(output_image), processed)
    cv2.imwrite(str(output_side_by_side), side_by_side)
    predictions_path, summary_path = _write_outputs(predictions, summary, output_dir, args.model_type)

    print("Image demo complete")
    print(f"Model type: {args.model_type}")
    print(f"Image: {selected_image_path}")
    print(f"Slots: {summary['total_slots']}")
    print(f"Occupied: {summary['occupied_count']}")
    print(f"Vacant: {summary['vacant_count']}")
    print(f"Occupancy rate: {summary['occupancy_rate']:.4f}")
    if threshold is not None:
        print(f"Threshold: {threshold:.2f}")
    print(f"Output image: {_relative_or_absolute(output_image)}")
    print(f"Side by side: {_relative_or_absolute(output_side_by_side)}")
    print(f"Predictions: {_relative_or_absolute(predictions_path)}")
    print(f"Summary: {_relative_or_absolute(summary_path)}")
    return 0


def main() -> int:
    return run_demo(_parse_args())


if __name__ == "__main__":
    raise SystemExit(main())

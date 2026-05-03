from __future__ import annotations

import argparse
import json
import random
from collections import OrderedDict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd

from src.classical.dataset import SlotRecord, load_manifest_records
from src.classical.features import resolve_image_path
from src.utils.config import PROJECT_ROOT, RESULT_VIDEOS_DIR
from src.visualization.demo_image import (
    DEFAULT_CLASSICAL_MODEL,
    DEFAULT_CNN_MODEL,
    DEFAULT_MANIFEST,
    load_classical_predictor,
    load_cnn_predictor,
    load_image_pair,
    predict_slots_for_image,
    read_cnn_threshold,
)
from src.visualization.draw_overlays import (
    create_side_by_side,
    draw_frame_status_panel,
    draw_legend,
    draw_slot_overlays,
)


DEFAULT_OUTPUT_DIR = Path("results/metrics/demo_video")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create an annotated frame-sequence video demo from PKLot test images."
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--model-type", choices=["cnn", "classical"], default="cnn")
    parser.add_argument("--cnn-model", type=Path, default=DEFAULT_CNN_MODEL)
    parser.add_argument("--classical-model", type=Path, default=DEFAULT_CLASSICAL_MODEL)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--num-frames", type=int, default=60)
    parser.add_argument("--min-slots", type=int, default=50)
    parser.add_argument("--fps", type=float, default=5.0)
    parser.add_argument("--max-labels", type=int, default=80)
    parser.add_argument("--output-video", type=Path, default=None)
    parser.add_argument("--output-side-by-side", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _resolve_project_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def _relative_or_absolute(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def _default_output_video(model_type: str) -> Path:
    return RESULT_VIDEOS_DIR / f"demo_video_{model_type}_output.mp4"


def _default_side_by_side(model_type: str) -> Path:
    return RESULT_VIDEOS_DIR / f"demo_video_{model_type}_side_by_side.mp4"


def _group_records_by_image(records: list[SlotRecord]) -> OrderedDict[str, list[SlotRecord]]:
    grouped: OrderedDict[str, list[SlotRecord]] = OrderedDict()
    for record in records:
        grouped.setdefault(record.image_path, []).append(record)
    return grouped


def _select_frame_groups(
    records: list[SlotRecord],
    num_frames: int,
    min_slots: int,
    seed: int,
) -> list[tuple[str, Path, list[SlotRecord]]]:
    grouped = _group_records_by_image(records)
    eligible = []
    for order, (image_path, image_records) in enumerate(grouped.items()):
        if len(image_records) < min_slots:
            continue
        resolved_path = resolve_image_path(image_path)
        if resolved_path.is_file():
            eligible.append((order, image_path, resolved_path, image_records))

    if not eligible:
        raise ValueError(
            f"No test images with at least {min_slots} slot annotations could be resolved."
        )

    rng = random.Random(seed)
    if len(eligible) > num_frames:
        eligible = rng.sample(eligible, num_frames)
    eligible.sort(key=lambda item: item[0])
    return [(image_path, resolved_path, image_records) for _, image_path, resolved_path, image_records in eligible]


def _load_predictor(args: argparse.Namespace) -> tuple[Any, float | None]:
    if args.model_type == "cnn":
        threshold = args.threshold if args.threshold is not None else read_cnn_threshold(default=0.48)
        return load_cnn_predictor(_resolve_project_path(args.cnn_model), threshold), threshold
    return load_classical_predictor(_resolve_project_path(args.classical_model)), None


def _frame_summary(
    frame_index: int,
    image_path: str,
    predictions: list[dict[str, Any]],
    model_type: str,
    threshold: float | None,
) -> dict[str, Any]:
    occupied_count = sum(row["predicted_label"] == "occupied" for row in predictions)
    vacant_count = sum(row["predicted_label"] == "vacant" for row in predictions)
    total_slots = len(predictions)
    return {
        "frame_index": frame_index,
        "image_path": image_path,
        "total_slots": total_slots,
        "occupied_count": occupied_count,
        "vacant_count": vacant_count,
        "occupancy_rate": occupied_count / max(1, total_slots),
        "model_type": model_type,
        "threshold": threshold if threshold is not None else "NA",
    }


def _prediction_rows(
    frame_index: int,
    predictions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    for prediction in predictions:
        row = {"frame_index": frame_index}
        row.update(prediction)
        rows.append(row)
    return rows


def _resize_to_size(frame: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    width, height = size
    if frame.shape[1] == width and frame.shape[0] == height:
        return frame
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)


def _open_video_writer(path: Path, fps: float, size: tuple[int, int]) -> tuple[cv2.VideoWriter, Path]:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    if writer.isOpened():
        return writer, path

    writer.release()
    fallback_path = path.with_suffix(".avi")
    writer = cv2.VideoWriter(str(fallback_path), cv2.VideoWriter_fourcc(*"XVID"), fps, size)
    if writer.isOpened():
        return writer, fallback_path

    writer.release()
    raise RuntimeError(f"Could not open video writer for {path} or {fallback_path}")


def _verify_video(path: Path) -> None:
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise RuntimeError(f"Generated video could not be opened: {path}")
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count <= 0:
            raise RuntimeError(f"Generated video has no readable frames: {path}")
    finally:
        capture.release()


def _write_outputs(
    trend_rows: list[dict[str, Any]],
    prediction_rows: list[dict[str, Any]],
    summary: dict[str, Any],
    output_dir: Path,
    model_type: str,
) -> tuple[Path, Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    trend_csv = output_dir / "demo_video_occupancy_trend.csv"
    prediction_csv = output_dir / "demo_video_predictions.csv"
    summary_json = output_dir / "demo_video_summary.json"
    model_trend_csv = output_dir / f"demo_video_{model_type}_occupancy_trend.csv"
    model_prediction_csv = output_dir / f"demo_video_{model_type}_predictions.csv"
    model_summary_json = output_dir / f"demo_video_{model_type}_summary.json"

    trend_frame = pd.DataFrame(trend_rows)
    prediction_frame = pd.DataFrame(prediction_rows)
    trend_frame.to_csv(trend_csv, index=False)
    trend_frame.to_csv(model_trend_csv, index=False)
    prediction_frame.to_csv(prediction_csv, index=False)
    prediction_frame.to_csv(model_prediction_csv, index=False)
    text = json.dumps(summary, indent=2)
    summary_json.write_text(text, encoding="utf-8")
    model_summary_json.write_text(text, encoding="utf-8")
    return trend_csv, prediction_csv, summary_json


def _summary(
    model_type: str,
    fps: float,
    threshold: float | None,
    trend_rows: list[dict[str, Any]],
    prediction_rows: list[dict[str, Any]],
    output_video: Path,
    output_side_by_side: Path,
    trend_csv: Path,
    prediction_csv: Path,
) -> dict[str, Any]:
    total_frames = len(trend_rows)
    return {
        "model_type": model_type,
        "demo_type": "Annotated frame-sequence video demo",
        "num_frames": total_frames,
        "fps": fps,
        "total_predictions": len(prediction_rows),
        "average_total_slots": float(np.mean([row["total_slots"] for row in trend_rows])) if trend_rows else 0.0,
        "average_occupied_count": float(np.mean([row["occupied_count"] for row in trend_rows])) if trend_rows else 0.0,
        "average_vacant_count": float(np.mean([row["vacant_count"] for row in trend_rows])) if trend_rows else 0.0,
        "average_occupancy_rate": float(np.mean([row["occupancy_rate"] for row in trend_rows])) if trend_rows else 0.0,
        "threshold": threshold if threshold is not None else "NA",
        "output_video": _relative_or_absolute(output_video),
        "output_side_by_side": _relative_or_absolute(output_side_by_side),
        "trend_csv": _relative_or_absolute(trend_csv),
        "prediction_csv": _relative_or_absolute(prediction_csv),
    }


def run_demo(args: argparse.Namespace) -> int:
    if args.num_frames <= 0:
        raise ValueError("--num-frames must be positive")
    if args.fps <= 0:
        raise ValueError("--fps must be positive")

    records = load_manifest_records(_resolve_project_path(args.manifest))
    frame_groups = _select_frame_groups(
        records,
        num_frames=args.num_frames,
        min_slots=args.min_slots,
        seed=args.seed,
    )
    predictor, threshold = _load_predictor(args)

    output_video = _resolve_project_path(args.output_video) if args.output_video else _default_output_video(args.model_type)
    output_side_by_side = (
        _resolve_project_path(args.output_side_by_side)
        if args.output_side_by_side
        else _default_side_by_side(args.model_type)
    )
    output_dir = _resolve_project_path(args.output_dir)

    trend_rows: list[dict[str, Any]] = []
    all_prediction_rows: list[dict[str, Any]] = []
    video_writer: cv2.VideoWriter | None = None
    side_writer: cv2.VideoWriter | None = None
    actual_output_video = output_video
    actual_side_by_side = output_side_by_side
    video_size: tuple[int, int] | None = None
    side_size: tuple[int, int] | None = None
    total_frames = len(frame_groups)

    try:
        for frame_index, (image_path, resolved_path, image_records) in enumerate(frame_groups, start=1):
            original_bgr, image_rgb = load_image_pair(resolved_path)
            predictions, model_name = predict_slots_for_image(image_records, image_rgb, predictor)
            frame_summary = _frame_summary(
                frame_index=frame_index,
                image_path=image_path,
                predictions=predictions,
                model_type=args.model_type,
                threshold=threshold,
            )
            trend_rows.append(frame_summary)
            all_prediction_rows.extend(_prediction_rows(frame_index, predictions))

            processed = draw_slot_overlays(original_bgr, predictions, max_labels=args.max_labels)
            processed = draw_legend(processed, x=18, y=max(170, processed.shape[0] - 86))
            processed = draw_frame_status_panel(
                processed,
                frame_index=frame_index,
                total_frames=total_frames,
                summary=frame_summary,
                model_name=model_name,
            )
            side_by_side = create_side_by_side(
                original_bgr,
                processed,
                left_title="Original Frame",
                right_title="Processed Frame",
            )

            if video_writer is None:
                video_size = (processed.shape[1], processed.shape[0])
                video_writer, actual_output_video = _open_video_writer(output_video, args.fps, video_size)
            if side_writer is None:
                side_size = (side_by_side.shape[1], side_by_side.shape[0])
                side_writer, actual_side_by_side = _open_video_writer(output_side_by_side, args.fps, side_size)

            video_writer.write(_resize_to_size(processed, video_size))
            side_writer.write(_resize_to_size(side_by_side, side_size))
    finally:
        if video_writer is not None:
            video_writer.release()
        if side_writer is not None:
            side_writer.release()

    _verify_video(actual_output_video)
    _verify_video(actual_side_by_side)

    output_dir.mkdir(parents=True, exist_ok=True)
    trend_csv = output_dir / "demo_video_occupancy_trend.csv"
    prediction_csv = output_dir / "demo_video_predictions.csv"
    summary = _summary(
        model_type=args.model_type,
        fps=args.fps,
        threshold=threshold,
        trend_rows=trend_rows,
        prediction_rows=all_prediction_rows,
        output_video=actual_output_video,
        output_side_by_side=actual_side_by_side,
        trend_csv=trend_csv,
        prediction_csv=prediction_csv,
    )
    trend_csv, prediction_csv, summary_json = _write_outputs(
        trend_rows,
        all_prediction_rows,
        summary,
        output_dir,
        args.model_type,
    )
    summary["trend_csv"] = _relative_or_absolute(trend_csv)
    summary["prediction_csv"] = _relative_or_absolute(prediction_csv)
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output_dir / f"demo_video_{args.model_type}_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print("Annotated frame-sequence video demo complete")
    print(f"Model type: {args.model_type}")
    print(f"Frames processed: {summary['num_frames']}")
    print(f"FPS: {summary['fps']}")
    print(f"Average occupied: {summary['average_occupied_count']:.2f}")
    print(f"Average vacant: {summary['average_vacant_count']:.2f}")
    print(f"Average occupancy rate: {summary['average_occupancy_rate']:.4f}")
    if threshold is not None:
        print(f"Threshold: {threshold:.2f}")
    print(f"Output video: {_relative_or_absolute(actual_output_video)}")
    print(f"Side by side: {_relative_or_absolute(actual_side_by_side)}")
    print(f"Trend CSV: {_relative_or_absolute(trend_csv)}")
    print(f"Predictions CSV: {_relative_or_absolute(prediction_csv)}")
    print(f"Summary JSON: {_relative_or_absolute(summary_json)}")
    return 0


def main() -> int:
    return run_demo(_parse_args())


if __name__ == "__main__":
    raise SystemExit(main())

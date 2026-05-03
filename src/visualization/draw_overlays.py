from __future__ import annotations

from typing import Any

import cv2
import numpy as np


VACANT_COLOR = (44, 166, 74)
OCCUPIED_COLOR = (39, 55, 219)
PANEL_COLOR = (20, 24, 32)
TEXT_COLOR = (255, 255, 255)
MUTED_TEXT_COLOR = (218, 226, 236)


def _as_float(value: Any) -> float | None:
    if value in (None, "", "NA"):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _clip_box(
    image: np.ndarray,
    x1: Any,
    y1: Any,
    x2: Any,
    y2: Any,
) -> tuple[int, int, int, int] | None:
    height, width = image.shape[:2]
    values = [_as_float(value) for value in [x1, y1, x2, y2]]
    if any(value is None for value in values):
        return None

    left = max(0, min(width - 1, int(np.floor(values[0]))))
    top = max(0, min(height - 1, int(np.floor(values[1]))))
    right = max(0, min(width - 1, int(np.ceil(values[2]))))
    bottom = max(0, min(height - 1, int(np.ceil(values[3]))))
    if right <= left or bottom <= top:
        return None
    return left, top, right, bottom


def _slot_color(label: str) -> tuple[int, int, int]:
    return OCCUPIED_COLOR if label.lower() == "occupied" else VACANT_COLOR


def _short_label(label: str) -> str:
    return "Occ" if label.lower() == "occupied" else "Vac"


def _confidence_text(prediction: dict[str, Any]) -> str:
    probability = _as_float(prediction.get("occupied_probability"))
    if probability is None:
        return ""

    label = str(prediction.get("predicted_label", "")).lower()
    confidence = probability if label == "occupied" else 1.0 - probability
    confidence = max(0.0, min(1.0, confidence))
    return f" {confidence * 100:.0f}%"


def make_slot_label(prediction: dict[str, Any]) -> str:
    slot_index = prediction.get("slot_index", "")
    label = str(prediction.get("predicted_label", "vacant"))
    return f"S{slot_index} {_short_label(label)}{_confidence_text(prediction)}"


def draw_slot_overlays(
    image_bgr: np.ndarray,
    predictions: list[dict[str, Any]],
    max_labels: int = 80,
    fill_alpha: float = 0.32,
) -> np.ndarray:
    output = image_bgr.copy()
    overlay = output.copy()

    for prediction in predictions:
        box = _clip_box(
            output,
            prediction.get("x1"),
            prediction.get("y1"),
            prediction.get("x2"),
            prediction.get("y2"),
        )
        if box is None:
            continue
        left, top, right, bottom = box
        color = _slot_color(str(prediction.get("predicted_label", "vacant")))
        cv2.rectangle(overlay, (left, top), (right, bottom), color, thickness=-1)

    output = cv2.addWeighted(overlay, fill_alpha, output, 1.0 - fill_alpha, 0)

    for index, prediction in enumerate(predictions):
        box = _clip_box(
            output,
            prediction.get("x1"),
            prediction.get("y1"),
            prediction.get("x2"),
            prediction.get("y2"),
        )
        if box is None:
            continue
        left, top, right, bottom = box
        color = _slot_color(str(prediction.get("predicted_label", "vacant")))
        cv2.rectangle(output, (left, top), (right, bottom), color, thickness=2)

        if index >= max_labels:
            continue
        draw_text_badge(output, make_slot_label(prediction), left, top, color)

    return output


def draw_text_badge(
    image_bgr: np.ndarray,
    text: str,
    x: int,
    y: int,
    color: tuple[int, int, int],
) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.42
    thickness = 1
    (text_width, text_height), baseline = cv2.getTextSize(text, font, scale, thickness)
    pad_x = 5
    pad_y = 4
    image_height, image_width = image_bgr.shape[:2]

    badge_left = max(0, min(x, image_width - text_width - 2 * pad_x - 1))
    badge_top = max(0, y - text_height - baseline - 2 * pad_y - 2)
    badge_right = min(image_width - 1, badge_left + text_width + 2 * pad_x)
    badge_bottom = min(image_height - 1, badge_top + text_height + baseline + 2 * pad_y)

    cv2.rectangle(
        image_bgr,
        (badge_left, badge_top),
        (badge_right, badge_bottom),
        color,
        thickness=-1,
    )
    cv2.rectangle(
        image_bgr,
        (badge_left, badge_top),
        (badge_right, badge_bottom),
        (255, 255, 255),
        thickness=1,
    )
    cv2.putText(
        image_bgr,
        text,
        (badge_left + pad_x, badge_bottom - baseline - pad_y),
        font,
        scale,
        TEXT_COLOR,
        thickness,
        lineType=cv2.LINE_AA,
    )


def draw_legend(image_bgr: np.ndarray, x: int = 18, y: int = 18) -> np.ndarray:
    output = image_bgr.copy()
    items = [("Vacant", VACANT_COLOR), ("Occupied", OCCUPIED_COLOR)]
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.52
    thickness = 1
    row_height = 24
    panel_width = 150
    panel_height = 16 + row_height * len(items)
    _draw_translucent_panel(output, x, y, panel_width, panel_height, alpha=0.72)

    cursor_y = y + 23
    for label, color in items:
        cv2.rectangle(output, (x + 12, cursor_y - 13), (x + 30, cursor_y + 5), color, -1)
        cv2.rectangle(output, (x + 12, cursor_y - 13), (x + 30, cursor_y + 5), (255, 255, 255), 1)
        cv2.putText(
            output,
            label,
            (x + 40, cursor_y + 3),
            font,
            scale,
            TEXT_COLOR,
            thickness,
            lineType=cv2.LINE_AA,
        )
        cursor_y += row_height
    return output


def draw_summary_panel(
    image_bgr: np.ndarray,
    summary: dict[str, Any],
    model_name: str,
) -> np.ndarray:
    output = image_bgr.copy()
    image_height, image_width = output.shape[:2]
    panel_width = min(360, max(285, image_width - 36))
    panel_height = 150
    x = max(18, image_width - panel_width - 18)
    y = 18
    _draw_translucent_panel(output, x, y, panel_width, panel_height, alpha=0.76)

    total = int(summary.get("total_slots", 0))
    occupied = int(summary.get("occupied_count", 0))
    vacant = int(summary.get("vacant_count", 0))
    occupancy_rate = float(summary.get("occupancy_rate", 0.0))
    lines = [
        ("Smart Parking Demo", 0.62, TEXT_COLOR, 2),
        (f"Model: {model_name}", 0.48, MUTED_TEXT_COLOR, 1),
        (f"Total slots: {total}", 0.48, TEXT_COLOR, 1),
        (f"Occupied: {occupied}   Vacant: {vacant}", 0.48, TEXT_COLOR, 1),
        (f"Occupancy rate: {occupancy_rate * 100:.1f}%", 0.48, TEXT_COLOR, 1),
    ]

    cursor_y = y + 26
    for text, scale, color, thickness in lines:
        cv2.putText(
            output,
            text,
            (x + 14, cursor_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            color,
            thickness,
            lineType=cv2.LINE_AA,
        )
        cursor_y += 27
    return output


def _draw_translucent_panel(
    image_bgr: np.ndarray,
    x: int,
    y: int,
    width: int,
    height: int,
    alpha: float,
) -> None:
    overlay = image_bgr.copy()
    cv2.rectangle(overlay, (x, y), (x + width, y + height), PANEL_COLOR, thickness=-1)
    cv2.addWeighted(overlay, alpha, image_bgr, 1.0 - alpha, 0, dst=image_bgr)
    cv2.rectangle(image_bgr, (x, y), (x + width, y + height), (236, 240, 244), thickness=1)


def create_side_by_side(
    original_bgr: np.ndarray,
    processed_bgr: np.ndarray,
    left_title: str = "Original",
    right_title: str = "Processed",
) -> np.ndarray:
    target_height = max(original_bgr.shape[0], processed_bgr.shape[0])
    left = _resize_to_height(original_bgr, target_height)
    right = _resize_to_height(processed_bgr, target_height)
    title_height = 44
    gap = 8
    canvas = np.full(
        (
            target_height + title_height,
            left.shape[1] + right.shape[1] + gap,
            3,
        ),
        245,
        dtype=np.uint8,
    )
    canvas[title_height:, : left.shape[1]] = left
    canvas[title_height:, left.shape[1] + gap :] = right
    canvas[title_height:, left.shape[1] : left.shape[1] + gap] = 36
    _draw_title(canvas, left_title, 16, 29)
    _draw_title(canvas, right_title, left.shape[1] + gap + 16, 29)
    return canvas


def _resize_to_height(image_bgr: np.ndarray, target_height: int) -> np.ndarray:
    if image_bgr.shape[0] == target_height:
        return image_bgr.copy()
    scale = target_height / image_bgr.shape[0]
    target_width = max(1, int(round(image_bgr.shape[1] * scale)))
    return cv2.resize(image_bgr, (target_width, target_height), interpolation=cv2.INTER_AREA)


def _draw_title(image_bgr: np.ndarray, text: str, x: int, y: int) -> None:
    cv2.putText(
        image_bgr,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (31, 41, 55),
        2,
        lineType=cv2.LINE_AA,
    )

"""Load and apply a fixed four-corner panel perspective transform."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from app.config import PANEL_CORNERS, PANEL_HEIGHT, PANEL_WIDTH, ROI_CONFIG_PATH


def save_roi(points: list[list[float]], path: Path = ROI_CONFIG_PATH) -> None:
    if len(points) != 4 or any(len(point) != 2 for point in points):
        raise ValueError("ROI must contain exactly four [x, y] corner points")
    values = np.asarray(points, dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("ROI corner coordinates must be finite")
    payload = {
        "corners": {
            name: [float(value) for value in point]
            for name, point in zip(PANEL_CORNERS, values.tolist())
        },
        "output_size": [PANEL_WIDTH, PANEL_HEIGHT],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_roi(path: Path = ROI_CONFIG_PATH) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(
            f"Panel ROI configuration is missing: {path}. "
            "Run training/calibrate_roi.py with an image or camera to set the four corners."
        )
    try:
        payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
        corner_values = payload["corners"]
        points = np.asarray(
            [corner_values[name] for name in PANEL_CORNERS],
            dtype=np.float32,
        )
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
        raise ValueError(f"Invalid ROI configuration in {path}: {error}") from error
    if points.shape != (4, 2) or not np.isfinite(points).all():
        raise ValueError(f"ROI in {path} must provide four finite 2D points")
    if len(np.unique(points, axis=0)) != 4:
        raise ValueError(f"ROI in {path} must provide four distinct corners")
    return points


def warp_to_panel(frame: np.ndarray, roi_path: Path = ROI_CONFIG_PATH) -> np.ndarray:
    if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
        raise ValueError("Frame must be a non-empty OpenCV image array")
    points = load_roi(roi_path)
    target = np.asarray(
        [
            [0, 0],
            [PANEL_WIDTH - 1, 0],
            [PANEL_WIDTH - 1, PANEL_HEIGHT - 1],
            [0, PANEL_HEIGHT - 1],
        ],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(points, target)
    return cv2.warpPerspective(frame, matrix, (PANEL_WIDTH, PANEL_HEIGHT))

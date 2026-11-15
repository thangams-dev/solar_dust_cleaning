"""Capture frames from the fixed camera, crop to the panel, classify, and log.

Run from ~/solar_project with the venv active:
    python app/capture_check.py                  # live camera, 3 frames
    python app/capture_check.py --image PATH     # test on one saved photo

Close guvcview first: only one program can use the camera at a time.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "training"))

from app.config import DUSTY_THRESHOLD  # noqa: E402
from app.panel_roi import warp_to_panel  # noqa: E402
from predict import build_transform, labels_path_for, make_model  # noqa: E402

# Stable path (by USB port). Keep the camera in the same port.
CAMERA_PATH = "/dev/v4l/by-path/pci-0000:c4:00.3-usb-0:1:1.0-video-index0"
MODEL_PATH = PROJECT_ROOT / "model" / "model_v2.pt"
OUTPUT_DIR = PROJECT_ROOT / "captures"
LOG_PATH = OUTPUT_DIR / "checks.csv"

FRAME_SIZE = (1280, 720)  # the ROI was calibrated at this size
FRAME_COUNT = 3
FRAME_GAP_S = 2.0
WARMUP_FRAMES = 10
FLUSH_FRAMES = 5  # drop buffered frames so each read is fresh
EDGE_TRIM_PX = 10  # trims the thin orange edge left by the ROI
DARK_LEVEL = 40  # starting value, not tuned: mean pixel brightness 0-255


def grab_frames() -> list[np.ndarray]:
    capture = cv2.VideoCapture(CAMERA_PATH, cv2.CAP_V4L2)
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open camera {CAMERA_PATH}. Is guvcview closed?")
    capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_SIZE[0])
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_SIZE[1])
    frames: list[np.ndarray] = []
    try:
        for _ in range(WARMUP_FRAMES):
            capture.read()
        for index in range(FRAME_COUNT):
            for _ in range(FLUSH_FRAMES):
                capture.read()
            ok, frame = capture.read()
            if not ok or frame is None:
                raise RuntimeError("Camera returned no frame")
            frames.append(frame)
            if index < FRAME_COUNT - 1:
                time.sleep(FRAME_GAP_S)
    finally:
        capture.release()
    return frames


def load_model():
    labels_path = labels_path_for(MODEL_PATH)
    metadata = json.loads(labels_path.read_text(encoding="utf-8"))
    labels = metadata["labels"]
    checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
    architecture = checkpoint.get(
        "architecture", metadata.get("architecture", "mobilenet_v2")
    )
    model = make_model(architecture, len(labels))
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    temperature = float(
        checkpoint.get("temperature", metadata.get("temperature", 1.0))
    )
    transform = build_transform(
        int(metadata["input_size"]),
        metadata["normalization"]["mean"],
        metadata["normalization"]["std"],
    )
    return model, labels, temperature, transform


def crop_panel(frame: np.ndarray) -> np.ndarray:
    panel = warp_to_panel(frame)
    trim = EDGE_TRIM_PX
    return panel[trim:-trim, trim:-trim]


def dusty_probability(model, labels, temperature, transform, panel) -> float:
    rgb = cv2.cvtColor(panel, cv2.COLOR_BGR2RGB)
    tensor = transform(Image.fromarray(rgb)).unsqueeze(0)
    with torch.inference_mode():
        probabilities = torch.softmax(model(tensor) / temperature, dim=1)[0]
    return float(probabilities[labels.index("dusty")])


def append_log(row: dict) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    is_new = not LOG_PATH.is_file()
    with LOG_PATH.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if is_new:
            writer.writeheader()
        writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, help="Test on a saved photo, no camera")
    arguments = parser.parse_args()

    if arguments.image:
        frame = cv2.imread(str(arguments.image))
        if frame is None:
            raise SystemExit(f"Could not read image: {arguments.image}")
        frames = [frame]
    else:
        frames = grab_frames()

    expected = (FRAME_SIZE[1], FRAME_SIZE[0])
    for frame in frames:
        if frame.shape[:2] != expected:
            raise SystemExit(
                f"Frame is {frame.shape[1]}x{frame.shape[0]}, but the ROI was "
                f"calibrated at {FRAME_SIZE[0]}x{FRAME_SIZE[1]}. Skipping."
            )

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    brightness = float(np.mean([frame.mean() for frame in frames]))
    row = {"time": stamp, "brightness": round(brightness, 1)}

    if brightness < DARK_LEVEL:
        row.update(status="skipped_dark", p_dusty_frames="", p_dusty_mean="", label="")
        append_log(row)
        print(f"Skipped: image too dark (brightness {brightness:.1f}).")
        return

    model, labels, temperature, transform = load_model()
    run_dir = OUTPUT_DIR / stamp
    run_dir.mkdir(parents=True, exist_ok=True)
    scores = []
    for index, frame in enumerate(frames):
        panel = crop_panel(frame)
        cv2.imwrite(str(run_dir / f"frame_{index}.png"), panel)
        score = dusty_probability(model, labels, temperature, transform, panel)
        scores.append(score)
        print(f"frame {index}: p_dusty = {score:.3f}")

    mean_score = float(np.mean(scores))
    label = "dusty" if mean_score >= DUSTY_THRESHOLD else "clean"
    print(f"RESULT: {label}  (mean p_dusty {mean_score:.3f}, threshold {DUSTY_THRESHOLD})")
    print(f"Saved crops in {run_dir}")

    row.update(
        status="ok",
        p_dusty_frames=" ".join(f"{score:.3f}" for score in scores),
        p_dusty_mean=round(mean_score, 3),
        label=label,
    )
    append_log(row)


if __name__ == "__main__":
    main()

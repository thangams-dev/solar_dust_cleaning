"""Capture labelled, ROI-cropped webcam images for own-panel data collection."""

from __future__ import annotations

import argparse
import csv
import logging
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Protocol

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OWN_DATA_ROOT = PROJECT_ROOT / "training" / "data" / "own"
DEFAULT_ROI_PATH = PROJECT_ROOT / "app" / "config" / "roi.json"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
LABELS = ("clean", "dusty", "cracked", "shaded")
DUST_LEVELS = ("light", "medium", "heavy")
BLUR_VARIANCE_THRESHOLD = 45.0
GLARE_FRACTION_THRESHOLD = 0.01
METADATA_FIELDS = (
    "captured_at",
    "image_path",
    "session",
    "label",
    "dust_level",
    "camera_index",
    "laplacian_variance",
    "glare_fraction",
    "blur_warning",
    "glare_warning",
    "controls_lock_requested",
    "controls_lock_succeeded",
)
LOGGER = logging.getLogger("capture_own")


class Camera(Protocol):
    def isOpened(self) -> bool: ...

    def read(self) -> tuple[bool, np.ndarray | None]: ...

    def release(self) -> None: ...


@dataclass(frozen=True)
class CaptureLabel:
    session: str
    label: str
    dust_level: str


@dataclass(frozen=True)
class ImageQuality:
    laplacian_variance: float
    glare_fraction: float
    blur_warning: bool
    glare_warning: bool


def validate_capture_label(session: str, label: str, dust_level: str = "") -> CaptureLabel:
    session = session.strip()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}", session):
        raise ValueError(
            "Session name must be 1-80 letters, digits, dots, underscores, or hyphens "
            "and start with a letter or digit."
        )
    if label not in LABELS:
        raise ValueError(f"Label must be one of: {', '.join(LABELS)}")
    if label == "dusty" and dust_level not in DUST_LEVELS:
        raise ValueError(f"Dusty captures need a level: {', '.join(DUST_LEVELS)}")
    if label != "dusty" and dust_level:
        raise ValueError("Dust level is only applicable to the dusty label.")
    return CaptureLabel(session, label, dust_level)


def prompt_capture_label(input_fn: Callable[[str], str] | None = None) -> CaptureLabel:
    if input_fn is None:
        input_fn = input
    session = input_fn("Session name (one day or lighting setup): ").strip()
    label = input_fn(f"Label ({'/'.join(LABELS)}): ").strip().lower()
    dust_level = ""
    if label == "dusty":
        dust_level = input_fn(f"Dust level ({'/'.join(DUST_LEVELS)}): ").strip().lower()
    return validate_capture_label(session, label, dust_level)


def quality_metrics(
    image: np.ndarray,
    blur_threshold: float = BLUR_VARIANCE_THRESHOLD,
    glare_threshold: float = GLARE_FRACTION_THRESHOLD,
) -> ImageQuality:
    if image is None or image.size == 0:
        raise ValueError("Cannot assess an empty image.")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian_variance = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    glare_fraction = float(np.mean((hsv[:, :, 2] >= 245) & (hsv[:, :, 1] <= 45)))
    return ImageQuality(
        laplacian_variance=laplacian_variance,
        glare_fraction=glare_fraction,
        blur_warning=laplacian_variance < blur_threshold,
        glare_warning=glare_fraction >= glare_threshold,
    )


def lock_camera_controls(camera_index: int) -> bool:
    """Disable automatic exposure and white balance via the Linux V4L2 utility."""
    executable = shutil.which("v4l2-ctl")
    if executable is None:
        LOGGER.warning(
            "Could not lock webcam exposure/white balance: v4l2-ctl is unavailable. "
            "Install the v4l-utils system package, then retry."
        )
        return False
    device = f"/dev/video{camera_index}"
    command = [
        executable,
        "--device",
        device,
        "--set-ctrl=auto_exposure=1,white_balance_automatic=0",
    ]
    try:
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except subprocess.TimeoutExpired:
        LOGGER.warning("Timed out locking camera controls for %s.", device)
        return False
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()
        LOGGER.warning("Camera controls were not locked for %s: %s", device, detail)
        return False
    LOGGER.info("Automatic exposure and white balance disabled for %s.", device)
    return True


def _write_image_exclusive(
    image: np.ndarray, output_dir: Path, captured_at: datetime
) -> Path:
    encoded_ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 95])
    if not encoded_ok:
        raise OSError("OpenCV could not encode the captured panel image.")
    timestamp = captured_at.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    suffix = 0
    while True:
        extra = "" if suffix == 0 else f"_{suffix:03d}"
        path = output_dir / f"{timestamp}{extra}.jpg"
        try:
            descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        except FileExistsError:
            suffix += 1
            continue
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(encoded.tobytes())
                stream.flush()
                os.fsync(stream.fileno())
        except Exception:
            path.unlink(missing_ok=True)
            raise
        return path


def _append_metadata(session_dir: Path, record: dict[str, object]) -> None:
    metadata_path = session_dir / "metadata.csv"
    exists = metadata_path.is_file()
    if exists and metadata_path.stat().st_size:
        with metadata_path.open(newline="", encoding="utf-8") as stream:
            header = next(csv.reader(stream), [])
        if tuple(header) != METADATA_FIELDS:
            raise ValueError(f"Unexpected metadata.csv columns in {metadata_path}")
    with metadata_path.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=METADATA_FIELDS)
        if not exists or metadata_path.stat().st_size == 0:
            writer.writeheader()
        writer.writerow(record)
        stream.flush()
        os.fsync(stream.fileno())


def _preview(frame: np.ndarray, quality: ImageQuality) -> np.ndarray:
    preview = frame.copy()
    lines = [
        f"Sharpness (Laplacian variance): {quality.laplacian_variance:.1f}",
        f"Glare: {quality.glare_fraction:.1%}",
        "s: save   q / Esc: finish",
    ]
    if quality.blur_warning:
        lines[0] += "  WARNING: possible blur"
    if quality.glare_warning:
        lines[1] += "  WARNING: glare"
    for index, text in enumerate(lines):
        cv2.putText(
            preview,
            text,
            (12, 26 + index * 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (0, 0, 255) if "WARNING" in text else (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
    return preview


def capture_session(
    camera: Camera,
    camera_index: int,
    capture_label: CaptureLabel,
    roi_path: Path = DEFAULT_ROI_PATH,
    output_root: Path = OWN_DATA_ROOT,
    lock_controls: bool = True,
) -> int:
    if not camera.isOpened():
        camera.release()
        raise OSError(f"Could not open webcam camera index {camera_index}.")
    saved = 0
    session_dir = output_root / capture_label.label / capture_label.session
    try:
        from app.panel_roi import warp_to_panel

        controls_locked = lock_camera_controls(camera_index) if lock_controls else False
        session_dir.mkdir(parents=True, exist_ok=True)
        failed_reads = 0
        window_name = "Solar panel own-photo capture"
        LOGGER.info(
            "Capture ready: label=%s session=%s dust_level=%s; press s to save, q to finish.",
            capture_label.label,
            capture_label.session,
            capture_label.dust_level or "n/a",
        )
        if capture_label.label in ("cracked", "shaded"):
            LOGGER.warning(
                "%s can be recorded for future review but is not a deployed model class.",
                capture_label.label,
            )
        while True:
            ok, frame = camera.read()
            if not ok or frame is None:
                failed_reads += 1
                if failed_reads >= 10:
                    raise OSError("Webcam failed to provide 10 consecutive frames.")
                time.sleep(0.05)
                continue
            failed_reads = 0
            panel = warp_to_panel(frame, roi_path)
            quality = quality_metrics(panel)
            cv2.imshow(window_name, _preview(panel, quality))
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key != ord("s"):
                continue

            captured_at = datetime.now(timezone.utc)
            image_path = _write_image_exclusive(panel, session_dir, captured_at)
            record: dict[str, object] = {
                "captured_at": captured_at.isoformat(),
                "image_path": image_path.relative_to(PROJECT_ROOT).as_posix()
                if image_path.is_relative_to(PROJECT_ROOT)
                else str(image_path),
                "session": capture_label.session,
                "label": capture_label.label,
                "dust_level": capture_label.dust_level,
                "camera_index": camera_index,
                "laplacian_variance": f"{quality.laplacian_variance:.4f}",
                "glare_fraction": f"{quality.glare_fraction:.6f}",
                "blur_warning": quality.blur_warning,
                "glare_warning": quality.glare_warning,
                "controls_lock_requested": lock_controls,
                "controls_lock_succeeded": controls_locked,
            }
            _append_metadata(session_dir, record)
            saved += 1
            LOGGER.info("Saved %s", image_path)
            if quality.blur_warning:
                LOGGER.warning(
                    "Possible blur in %s (Laplacian variance %.1f < %.1f).",
                    image_path.name,
                    quality.laplacian_variance,
                    BLUR_VARIANCE_THRESHOLD,
                )
            if quality.glare_warning:
                LOGGER.warning(
                    "Possible glare in %s (%.1f%% of pixels).",
                    image_path.name,
                    quality.glare_fraction * 100,
                )
    finally:
        camera.release()
        cv2.destroyAllWindows()
    LOGGER.info("Capture finished; saved %d ROI images to %s.", saved, session_dir)
    return saved


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--roi", type=Path, default=DEFAULT_ROI_PATH)
    parser.add_argument("--output-root", type=Path, default=OWN_DATA_ROOT)
    parser.add_argument(
        "--no-lock-controls",
        action="store_true",
        help="Leave automatic camera exposure and white balance unchanged.",
    )
    parser.add_argument("--verbose", action="store_true")
    arguments = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO if arguments.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    capture_label = prompt_capture_label()
    camera = cv2.VideoCapture(arguments.camera_index)
    capture_session(
        camera=camera,
        camera_index=arguments.camera_index,
        capture_label=capture_label,
        roi_path=arguments.roi,
        output_root=arguments.output_root,
        lock_controls=not arguments.no_lock_controls,
    )


if __name__ == "__main__":
    main()

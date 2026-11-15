"""Click the four fixed-panel corners and save a perspective ROI."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import cv2


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.config import PANEL_CORNERS, ROI_CONFIG_PATH  # noqa: E402
from app.panel_roi import save_roi  # noqa: E402


LOGGER = logging.getLogger("calibrate_roi")


def load_frame(image_path: Path | None, camera_index: int) -> cv2.Mat:
    if image_path is not None:
        frame = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if frame is None:
            raise OSError(f"Could not read calibration image: {image_path}")
        return frame
    capture = cv2.VideoCapture(camera_index)
    if not capture.isOpened():
        capture.release()
        raise OSError(f"Could not open camera index {camera_index}")
    try:
        success, frame = capture.read()
        if not success or frame is None:
            raise OSError(f"Could not capture a frame from camera index {camera_index}")
        return frame
    finally:
        capture.release()


def calibrate(frame: cv2.Mat, output_path: Path) -> None:
    points: list[tuple[int, int]] = []
    display = frame.copy()
    window_name = "Click panel corners TL, TR, BR, BL; press q to cancel"

    def handle_click(event: int, x: int, y: int, _flags: int, _data: object) -> None:
        if event != cv2.EVENT_LBUTTONDOWN or len(points) >= len(PANEL_CORNERS):
            return
        points.append((x, y))
        cv2.circle(display, (x, y), 7, (0, 0, 255), -1)
        cv2.putText(
            display,
            PANEL_CORNERS[len(points) - 1],
            (x + 8, y - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )
        cv2.imshow(window_name, display)

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, handle_click)
    cv2.imshow(window_name, display)
    while len(points) < len(PANEL_CORNERS):
        key = cv2.waitKey(25) & 0xFF
        if key == ord("q") or cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            cv2.destroyAllWindows()
            raise RuntimeError("ROI calibration cancelled before four corners were selected")
    cv2.destroyAllWindows()
    save_roi([[float(x), float(y)] for x, y in points], output_path)
    LOGGER.info("Saved panel corners to %s", output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, help="Calibration frame image path.")
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--output", type=Path, default=ROI_CONFIG_PATH)
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    arguments = parse_args()
    frame = load_frame(arguments.image, arguments.camera_index)
    calibrate(frame, arguments.output)


if __name__ == "__main__":
    main()

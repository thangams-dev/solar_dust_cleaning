from __future__ import annotations

import csv
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from app.panel_roi import save_roi  # noqa: E402
from training.capture_own import (  # noqa: E402
    CaptureLabel,
    capture_session,
    prompt_capture_label,
    validate_capture_label,
)


class FakeCamera:
    def __init__(self, frame: np.ndarray) -> None:
        self.frame = frame
        self.keys = [ord("s"), ord("q")]
        self.released = False

    def isOpened(self) -> bool:
        return True

    def read(self) -> tuple[bool, np.ndarray]:
        return True, self.frame.copy()

    def release(self) -> None:
        self.released = True


def test_capture_prompts_and_validates_dust_level() -> None:
    answers = iter(("morning_light", "dusty", "medium"))
    capture_label = prompt_capture_label(lambda _prompt: next(answers))
    assert capture_label == CaptureLabel("morning_light", "dusty", "medium")

    try:
        validate_capture_label("../unsafe", "clean")
    except ValueError as error:
        assert "Session name" in str(error)
    else:
        raise AssertionError("Invalid session names must be rejected.")


def test_capture_uses_fake_camera_roi_and_writes_metadata(
    tmp_path: Path, monkeypatch
) -> None:
    roi_path = tmp_path / "roi.json"
    save_roi([[0, 0], [639, 0], [639, 479], [0, 479]], roi_path)
    output_root = tmp_path / "own"
    camera = FakeCamera(np.full((480, 640, 3), 255, dtype=np.uint8))
    monkeypatch.setattr(cv2, "imshow", lambda *_args: None)
    monkeypatch.setattr(cv2, "waitKey", lambda _delay: camera.keys.pop(0))
    monkeypatch.setattr(cv2, "destroyAllWindows", lambda: None)

    count = capture_session(
        camera,
        camera_index=7,
        capture_label=CaptureLabel("overcast_day", "dusty", "light"),
        roi_path=roi_path,
        output_root=output_root,
        lock_controls=False,
    )

    session_dir = output_root / "dusty" / "overcast_day"
    images = list(session_dir.glob("*.jpg"))
    assert count == 1
    assert len(images) == 1
    assert cv2.imread(str(images[0])).shape == (400, 480, 3)
    assert camera.released
    with (session_dir / "metadata.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 1
    assert rows[0]["label"] == "dusty"
    assert rows[0]["dust_level"] == "light"
    assert rows[0]["camera_index"] == "7"
    assert rows[0]["blur_warning"] == "True"
    assert rows[0]["glare_warning"] == "True"


def test_capture_never_overwrites_same_timestamp(
    tmp_path: Path, monkeypatch
) -> None:
    roi_path = tmp_path / "roi.json"
    save_roi([[0, 0], [639, 0], [639, 479], [0, 479]], roi_path)
    camera = FakeCamera(np.full((480, 640, 3), 100, dtype=np.uint8))
    camera.keys = [ord("s"), ord("s"), ord("q")]
    monkeypatch.setattr(cv2, "imshow", lambda *_args: None)
    monkeypatch.setattr(cv2, "waitKey", lambda _delay: camera.keys.pop(0))
    monkeypatch.setattr(cv2, "destroyAllWindows", lambda: None)
    from training import capture_own

    fixed_time = capture_own.datetime(2025, 1, 2, tzinfo=capture_own.timezone.utc)
    monkeypatch.setattr(
        capture_own,
        "datetime",
        type(
            "FixedDateTime",
            (),
            {"now": staticmethod(lambda _tz: fixed_time)},
        ),
    )

    count = capture_session(
        camera,
        camera_index=0,
        capture_label=CaptureLabel("same_setup", "clean", ""),
        roi_path=roi_path,
        output_root=tmp_path / "own",
        lock_controls=False,
    )

    images = list((tmp_path / "own" / "clean" / "same_setup").glob("*.jpg"))
    assert count == 2
    assert len(images) == 2
    assert len({path.name for path in images}) == 2

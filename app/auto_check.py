"""Automatic check loop, DRY RUN: capture -> analyse -> decide -> log.

It never talks to the ESP32. It only logs what it WOULD do.

Run from ~/solar_project with the venv active:
    python app/auto_check.py                        # every 5 min, live camera
    python app/auto_check.py --once                 # one check, then stop
    python app/auto_check.py --image PHOTO --interval 1 --cooldown 5
                                                    # test the decision rules on a saved photo

Stop with Ctrl-C. Close guvcview first when using the live camera.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.capture_check import (  # noqa: E402
    DARK_LEVEL,
    FRAME_SIZE,
    OUTPUT_DIR,
    crop_panel,
    dusty_probability,
    grab_frames,
    load_model,
)
from app.config import DUSTY_THRESHOLD  # noqa: E402

DECISION_LOG = OUTPUT_DIR / "decisions.csv"
LATEST_CROP = OUTPUT_DIR / "latest.png"

# Starting values from the project plan, not tuned on real data.
CHECK_INTERVAL_S = 300  # demo value; real use is 2-6 hours in daylight
CONFIRM_CHECKS = 2  # dusty this many checks in a row before cleaning
COOLDOWN_S = 1800  # at least 30 min between cleans
MAX_REPEAT_CLEANS = 3  # then raise an alert and stop deciding


class Decider:
    """Turns check results into decisions. No hardware, easy to test."""

    def __init__(self, cooldown_s: float) -> None:
        self.cooldown_s = cooldown_s
        self.streak = 0
        self.repeat = 0
        self.last_clean: float | None = None
        self.locked = False

    def update(self, result: dict) -> str:
        if result["status"] != "ok":
            return f"SKIP ({result['status']})"
        if result["label"] == "clean":
            self.streak = 0
            self.repeat = 0
            return "NO_ACTION"
        self.streak += 1
        if self.locked:
            return "LOCKED (cleaning not working, check the panel)"
        if self.streak < CONFIRM_CHECKS:
            return f"WAIT_CONFIRM {self.streak}/{CONFIRM_CHECKS}"
        now = time.monotonic()
        if self.last_clean is not None and now - self.last_clean < self.cooldown_s:
            return "WAIT_COOLDOWN"
        if self.repeat >= MAX_REPEAT_CLEANS:
            self.locked = True
            return "ALERT (still dusty after repeated cleans)"
        self.repeat += 1
        self.last_clean = now
        self.streak = 0
        return "WOULD_CLEAN (dry run, nothing sent)"


def run_cycle(bundle, image_path: Path | None) -> dict:
    try:
        if image_path:
            frame = cv2.imread(str(image_path))
            if frame is None:
                raise RuntimeError(f"Could not read {image_path}")
            frames = [frame]
        else:
            frames = grab_frames()
    except Exception as error:  # camera busy, unplugged, no frame
        return {"status": "camera_error", "note": str(error)}

    expected = (FRAME_SIZE[1], FRAME_SIZE[0])
    if any(frame.shape[:2] != expected for frame in frames):
        return {"status": "bad_frame_size", "note": "ROI needs 1280x720"}
    brightness = float(np.mean([frame.mean() for frame in frames]))
    if brightness < DARK_LEVEL:
        return {"status": "dark", "brightness": brightness}

    model, labels, temperature, transform = bundle
    crops = [crop_panel(frame) for frame in frames]
    scores = [
        dusty_probability(model, labels, temperature, transform, crop)
        for crop in crops
    ]
    mean_score = float(np.mean(scores))
    return {
        "status": "ok",
        "brightness": brightness,
        "mean": mean_score,
        "label": "dusty" if mean_score >= DUSTY_THRESHOLD else "clean",
        "crop": crops[len(crops) // 2],
    }


def log_decision(result: dict, decision: str) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    row = {
        "time": datetime.now().isoformat(timespec="seconds"),
        "status": result["status"],
        "brightness": round(result.get("brightness", 0.0), 1),
        "p_dusty_mean": round(result.get("mean", 0.0), 3),
        "label": result.get("label", ""),
        "decision": decision,
        "note": result.get("note", ""),
    }
    is_new = not DECISION_LOG.is_file()
    with DECISION_LOG.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if is_new:
            writer.writeheader()
        writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interval", type=float, default=CHECK_INTERVAL_S)
    parser.add_argument("--cooldown", type=float, default=COOLDOWN_S)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--image", type=Path, help="Use a saved photo, no camera")
    arguments = parser.parse_args()

    bundle = load_model()
    decider = Decider(arguments.cooldown)
    print(f"Dry run. Threshold {DUSTY_THRESHOLD}. Ctrl-C to stop.")
    try:
        while True:
            result = run_cycle(bundle, arguments.image)
            decision = decider.update(result)
            if "crop" in result:
                OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(LATEST_CROP), result["crop"])
            log_decision(result, decision)
            stamp = datetime.now().strftime("%H:%M:%S")
            detail = (
                f"p_dusty {result['mean']:.3f} -> {result['label']}"
                if result["status"] == "ok"
                else result["status"]
            )
            print(f"[{stamp}] {detail} | {decision}")
            if arguments.once:
                break
            time.sleep(arguments.interval)
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()

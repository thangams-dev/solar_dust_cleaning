"""Synthetic checks for fixed-ROI and reference-based dust-index helpers."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.dust_index import compare_before_after, dust_index, save_reference  # noqa: E402
from app.panel_roi import load_roi, save_roi, warp_to_panel  # noqa: E402


class PanelHelpersTest(unittest.TestCase):
    def setUp(self) -> None:
        rng = np.random.default_rng(42)
        panel = np.zeros((400, 480, 3), dtype=np.uint8)
        for y in range(0, 400, 40):
            cv2.line(panel, (0, y), (479, y), (100, 130, 155), 1)
        for x in range(0, 480, 48):
            cv2.line(panel, (x, 0), (x, 399), (100, 130, 155), 1)
        noise = rng.normal(0, 8, panel.shape)
        self.clean = np.clip(panel.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    def test_roi_warp_and_missing_config(self) -> None:
        with tempfile.TemporaryDirectory(
            prefix="roi_test_", dir=PROJECT_ROOT / "training" / "reports"
        ) as temporary:
            config_path = Path(temporary) / "roi.json"
            points = [[100, 80], [590, 110], [560, 520], [120, 500]]
            save_roi(points, config_path)
            self.assertEqual(load_roi(config_path).shape, (4, 2))
            transform = cv2.getPerspectiveTransform(
                np.asarray([[0, 0], [479, 0], [479, 399], [0, 399]], dtype=np.float32),
                np.asarray(points, dtype=np.float32),
            )
            frame = cv2.warpPerspective(self.clean, transform, (700, 600))
            crop = warp_to_panel(frame, config_path)
            self.assertEqual(crop.shape, (400, 480, 3))
            with self.assertRaises(FileNotFoundError):
                load_roi(Path(temporary) / "absent.json")

    def test_clean_reference_and_synthetic_haze(self) -> None:
        with tempfile.TemporaryDirectory(
            prefix="dust_test_", dir=PROJECT_ROOT / "training" / "reports"
        ) as temporary:
            reference = Path(temporary) / "clean.png"
            save_reference(self.clean, reference, already_warped=True)
            clean_score = dust_index(self.clean, reference)
            haze = np.clip(
                self.clean.astype(np.float32) * 0.68 + 150.0 * 0.32,
                0,
                255,
            ).astype(np.uint8)
            haze_score = dust_index(haze, reference)
            comparison = compare_before_after(haze, self.clean, reference)
            self.assertLess(clean_score, 2.0)
            self.assertGreater(haze_score, clean_score)
            self.assertGreater(comparison["improvement"], 0)


if __name__ == "__main__":
    unittest.main()

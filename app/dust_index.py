"""Exposure-normalized comparison with a saved clean-panel reference image."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from app.config import CLEAN_REFERENCE_PATH, HEAVY_DUST_REFERENCE_PATH
from app.panel_roi import warp_to_panel


def _validate_frame(frame: np.ndarray) -> np.ndarray:
    if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
        raise ValueError("Expected a non-empty OpenCV image array")
    if frame.ndim not in (2, 3):
        raise ValueError(f"Expected a grayscale or color image; got shape {frame.shape}")
    return frame


def _to_gray_float(frame: np.ndarray) -> np.ndarray:
    frame = _validate_frame(frame)
    if frame.ndim == 3:
        if frame.shape[2] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2GRAY)
        else:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(
        frame,
        (480, 400),
        interpolation=cv2.INTER_AREA if frame.shape[1] > 480 else cv2.INTER_LINEAR,
    ).astype(np.float32)
    return gray


def save_reference(
    frame: np.ndarray,
    reference_path: Path = CLEAN_REFERENCE_PATH,
    *,
    already_warped: bool = False,
) -> Path:
    """Save a clean panel frame; raw camera frames are warped using the calibrated ROI."""
    valid = _validate_frame(frame)
    panel = valid if already_warped else warp_to_panel(valid)
    reference_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(reference_path), panel):
        raise OSError(f"Could not write clean reference image to {reference_path}")
    return reference_path


def save_heavy_dust_reference(
    frame: np.ndarray,
    reference_path: Path = HEAVY_DUST_REFERENCE_PATH,
    *,
    already_warped: bool = False,
) -> Path:
    """Optionally save a heavy-dust example for later visual/calibration use."""
    valid = _validate_frame(frame)
    panel = valid if already_warped else warp_to_panel(valid)
    reference_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(reference_path), panel):
        raise OSError(f"Could not write heavy-dust reference image to {reference_path}")
    return reference_path


def _exposure_normalize(candidate: np.ndarray, reference: np.ndarray) -> np.ndarray:
    candidate_mean = float(np.mean(candidate))
    reference_mean = float(np.mean(reference))
    gain = reference_mean / max(candidate_mean, 1.0)
    return np.clip(candidate * gain, 0, 255)


def _ssim_score(first: np.ndarray, second: np.ndarray) -> float:
    first_blur = cv2.GaussianBlur(first, (11, 11), 1.5)
    second_blur = cv2.GaussianBlur(second, (11, 11), 1.5)
    first_sq = cv2.GaussianBlur(first * first, (11, 11), 1.5)
    second_sq = cv2.GaussianBlur(second * second, (11, 11), 1.5)
    product = cv2.GaussianBlur(first * second, (11, 11), 1.5)
    first_variance = np.maximum(first_sq - first_blur * first_blur, 0)
    second_variance = np.maximum(second_sq - second_blur * second_blur, 0)
    covariance = product - first_blur * second_blur
    constant_1 = (0.01 * 255) ** 2
    constant_2 = (0.03 * 255) ** 2
    numerator = (2 * first_blur * second_blur + constant_1) * (
        2 * covariance + constant_2
    )
    denominator = (first_blur**2 + second_blur**2 + constant_1) * (
        first_variance + second_variance + constant_2
    )
    return float(np.mean(numerator / np.maximum(denominator, 1e-8)))


def dust_index(
    frame: np.ndarray,
    reference_path: Path = CLEAN_REFERENCE_PATH,
    *,
    already_warped: bool = True,
) -> float:
    """Return a heuristic 0-100 dust-like difference score versus a clean reference.

    The candidate mean brightness is normalized to the clean-reference mean first.
    The score is 70% normalized structural dissimilarity and 30% relative contrast
    loss. It is not a calibrated physical dust measurement; lighting, reflections,
    shadows, camera motion, and panel changes can alter it.
    """
    if not reference_path.is_file():
        raise FileNotFoundError(
            f"Clean-panel reference is missing: {reference_path}. "
            "Capture a clean reference with save_reference() first."
        )
    candidate_frame = _validate_frame(frame)
    if not already_warped:
        candidate_frame = warp_to_panel(candidate_frame)
    reference_image = cv2.imread(str(reference_path), cv2.IMREAD_COLOR)
    if reference_image is None:
        raise OSError(f"Could not decode clean reference image: {reference_path}")

    reference = _to_gray_float(reference_image)
    candidate = _to_gray_float(candidate_frame)
    normalized_candidate = _exposure_normalize(candidate, reference)
    similarity = np.clip(_ssim_score(reference, normalized_candidate), -1.0, 1.0)
    reference_contrast = float(np.std(reference))
    candidate_contrast = float(np.std(normalized_candidate))
    relative_contrast_loss = np.clip(
        (reference_contrast - candidate_contrast) / max(reference_contrast, 1.0),
        0.0,
        1.0,
    )
    structural_difference = np.clip((1.0 - similarity) / 2.0, 0.0, 1.0)
    score = 100.0 * (0.70 * structural_difference + 0.30 * relative_contrast_loss)
    return float(np.clip(score, 0.0, 100.0))


def compare_before_after(
    before: np.ndarray,
    after: np.ndarray,
    reference_path: Path = CLEAN_REFERENCE_PATH,
    *,
    already_warped: bool = True,
) -> dict[str, float]:
    before_index = dust_index(
        before, reference_path, already_warped=already_warped
    )
    after_index = dust_index(
        after, reference_path, already_warped=already_warped
    )
    return {
        "before": before_index,
        "after": after_index,
        "improvement": before_index - after_index,
    }

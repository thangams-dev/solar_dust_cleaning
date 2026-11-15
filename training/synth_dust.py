"""Generate controllable synthetic dust effects for panel-image experiments."""

from __future__ import annotations

import argparse
import csv
import hashlib
import logging
import random
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
from PIL import Image, ImageOps


ROOT = Path(__file__).resolve().parent
OWN_SYNTH_ROOT = ROOT / "data" / "own_synth"
REPORT_ROOT = ROOT / "reports"
IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
LEVELS = ("light", "medium", "heavy")
LOGGER = logging.getLogger("synth_dust")

LEVEL_PARAMETERS: dict[str, dict[str, float | int]] = {
    "light": {
        "haze": 0.07,
        "contrast": 0.94,
        "desaturation": 0.12,
        "speckles": 45,
        "blotches": 3,
        "streaks": 2,
    },
    "medium": {
        "haze": 0.15,
        "contrast": 0.86,
        "desaturation": 0.24,
        "speckles": 110,
        "blotches": 6,
        "streaks": 4,
    },
    "heavy": {
        "haze": 0.25,
        "contrast": 0.75,
        "desaturation": 0.38,
        "speckles": 220,
        "blotches": 10,
        "streaks": 7,
    },
}


def _validate_rgb(image: np.ndarray) -> np.ndarray:
    if image is None or image.ndim != 3 or image.shape[2] != 3 or image.size == 0:
        raise ValueError("Expected a non-empty RGB uint8 image")
    if image.dtype != np.uint8:
        raise ValueError(f"Expected uint8 RGB input, got {image.dtype}")
    return image


def _low_frequency_field(
    height: int, width: int, rng: np.random.Generator, rows: int = 9, columns: int = 9
) -> np.ndarray:
    coarse = rng.normal(0.0, 1.0, (rows, columns)).astype(np.float32)
    field = cv2.resize(coarse, (width, height), interpolation=cv2.INTER_CUBIC)
    field = cv2.GaussianBlur(field, (0, 0), sigmaX=max(width, height) / 18)
    field -= float(field.mean())
    deviation = float(field.std())
    if deviation > 1e-6:
        field /= deviation
    return field


def _dust_mask(
    height: int,
    width: int,
    parameters: dict[str, float | int],
    rng: np.random.Generator,
) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.float32)
    blob_count = int(parameters["blotches"])
    for _ in range(blob_count):
        center = (int(rng.integers(width)), int(rng.integers(height)))
        axes = (
            int(rng.integers(max(4, width // 30), max(5, width // 7))),
            int(rng.integers(max(3, height // 35), max(4, height // 8))),
        )
        rotation = float(rng.uniform(0, 180))
        value = float(rng.uniform(0.2, 0.75))
        cv2.ellipse(mask, center, axes, rotation, 0, 360, value, -1, cv2.LINE_AA)
    mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=max(width, height) / 70)
    maximum = float(mask.max())
    if maximum > 1e-6:
        mask /= maximum
    return mask


def synthesize_image(image: np.ndarray, level: str, seed: int) -> np.ndarray:
    """Apply low-frequency haze, contrast/saturation loss, particles, blotches, and streaks."""
    rgb = _validate_rgb(image)
    if level not in LEVEL_PARAMETERS:
        raise ValueError(f"Unknown dust level {level!r}; choose from {LEVELS}")
    height, width = rgb.shape[:2]
    parameters = LEVEL_PARAMETERS[level]
    rng = np.random.default_rng(seed)
    original = rgb.astype(np.float32)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)[:, :, None]

    desaturation = float(parameters["desaturation"])
    base = original * (1.0 - desaturation) + gray * desaturation
    mean = base.mean(axis=(0, 1), keepdims=True)
    base = (base - mean) * float(parameters["contrast"]) + mean

    field = _low_frequency_field(height, width, rng)
    blotch_mask = _dust_mask(height, width, parameters, rng)
    intensity = float(parameters["haze"])
    haze_alpha = np.clip(intensity * (0.72 + 0.12 * field + 0.75 * blotch_mask), 0.0, 0.65)
    haze_alpha = cv2.GaussianBlur(haze_alpha.astype(np.float32), (0, 0), sigmaX=1.2)
    beige = np.asarray([184.0, 176.0, 158.0], dtype=np.float32)
    result = base * (1.0 - haze_alpha[:, :, None]) + beige * haze_alpha[:, :, None]

    speckle_count = int(parameters["speckles"])
    for _ in range(speckle_count):
        center = (int(rng.integers(width)), int(rng.integers(height)))
        radius = int(rng.choice([1, 1, 1, 2]))
        shade = float(rng.uniform(75, 225))
        opacity = float(rng.uniform(0.18, 0.62))
        color = np.asarray([shade, shade * 0.98, shade * 0.91], dtype=np.float32)
        particle_mask = np.zeros((height, width), dtype=np.float32)
        cv2.circle(particle_mask, center, radius, 1.0, -1, cv2.LINE_AA)
        particle_mask = cv2.GaussianBlur(particle_mask, (3, 3), 0.45)
        alpha = (particle_mask * opacity)[:, :, None]
        result = result * (1.0 - alpha) + color * alpha

    streak_count = int(parameters["streaks"])
    for _ in range(streak_count):
        start = (int(rng.integers(width)), int(rng.integers(height)))
        length = int(rng.integers(max(5, width // 18), max(6, width // 4)))
        angle = float(rng.uniform(-0.25, 0.25))
        end = (
            int(np.clip(start[0] + length * np.cos(angle), 0, width - 1)),
            int(np.clip(start[1] + length * np.sin(angle), 0, height - 1)),
        )
        streak = np.zeros((height, width), dtype=np.float32)
        cv2.line(streak, start, end, 1.0, int(rng.choice([1, 1, 2])), cv2.LINE_AA)
        streak = cv2.GaussianBlur(streak, (0, 0), sigmaX=1.0)
        alpha = (streak * float(rng.uniform(0.08, 0.22)))[:, :, None]
        color = np.asarray([153.0, 145.0, 129.0], dtype=np.float32)
        result = result * (1.0 - alpha) + color * alpha

    return np.clip(result, 0, 255).astype(np.uint8)


def discover_images(folder: Path) -> list[Path]:
    if not folder.is_dir():
        raise NotADirectoryError(f"Input folder does not exist: {folder}")
    return sorted(
        path
        for path in folder.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def _seed_for_path(seed: int, relative_path: str, level: str) -> int:
    digest = hashlib.sha256(f"{seed}\0{relative_path}\0{level}".encode()).digest()
    return int.from_bytes(digest[:4], "big")


def generate_folder(
    input_folder: Path,
    output_folder: Path,
    levels: Iterable[str] = LEVELS,
    seed: int = 42,
) -> list[Path]:
    selected_levels = tuple(levels)
    invalid_levels = set(selected_levels) - set(LEVELS)
    if invalid_levels:
        raise ValueError(f"Invalid dust levels: {sorted(invalid_levels)}")
    sources = discover_images(input_folder)
    if not sources:
        raise ValueError(f"No supported images found in {input_folder}")

    output_folder.mkdir(parents=True, exist_ok=True)
    manifest_path = output_folder / "manifest.csv"
    outputs: list[Path] = []
    with manifest_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=("output_path", "source_path", "level", "seed"),
        )
        writer.writeheader()
        for source_path in sources:
            relative = source_path.relative_to(input_folder)
            try:
                with Image.open(source_path) as opened:
                    rgb = np.asarray(ImageOps.exif_transpose(opened).convert("RGB"))
            except (OSError, ValueError) as error:
                raise OSError(f"Could not decode clean input {source_path}: {error}") from error
            for level in selected_levels:
                sample_seed = _seed_for_path(seed, relative.as_posix(), level)
                synthetic = synthesize_image(rgb, level, sample_seed)
                destination = (
                    output_folder
                    / level
                    / relative.parent
                    / f"{relative.stem}_{level}{relative.suffix.lower()}"
                )
                destination.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(synthetic).save(destination)
                outputs.append(destination)
                writer.writerow(
                    {
                        "output_path": destination.relative_to(output_folder).as_posix(),
                        "source_path": relative.as_posix(),
                        "level": level,
                        "seed": sample_seed,
                    }
                )
    LOGGER.info(
        "Generated %d synthetic images from %d clean inputs in %s",
        len(outputs),
        len(sources),
        output_folder,
    )
    return outputs


def save_example_grids(
    source_images: list[np.ndarray],
    output_directory: Path,
    seed: int = 42,
    samples_per_level: int = 16,
) -> dict[str, Path]:
    if not source_images:
        raise ValueError("At least one clean source image is required for example grids")
    output_directory.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    selected = rng.sample(source_images, k=min(samples_per_level, len(source_images)))
    paths: dict[str, Path] = {}
    for level in LEVELS:
        images = [
            synthesize_image(selected[index % len(selected)], level, seed + index)
            for index in range(samples_per_level)
        ]
        columns = 4
        rows = (len(images) + columns - 1) // columns
        import matplotlib.pyplot as plt

        figure, axes = plt.subplots(rows, columns, figsize=(columns * 3, rows * 3), squeeze=False)
        for axis, image in zip(axes.flat, images):
            axis.imshow(image)
            axis.axis("off")
        for axis in list(axes.flat)[len(images) :]:
            axis.axis("off")
        figure.suptitle(f"Synthetic dust examples: {level}")
        figure.tight_layout()
        path = output_directory / f"synthetic_dust_{level}.png"
        figure.savefig(path, dpi=120)
        plt.close(figure)
        paths[level] = path
    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-folder", type=Path, required=True, help="Clean image folder.")
    parser.add_argument(
        "--output-folder",
        type=Path,
        help="Output folder. Defaults to training/data/own_synth/<session>.",
    )
    parser.add_argument(
        "--session",
        default="webcam_session",
        help="Session folder under training/data/own_synth when output-folder is omitted.",
    )
    parser.add_argument("--level", choices=LEVELS + ("all",), default="all")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    arguments = parse_args()
    if arguments.output_folder is None:
        session = Path(arguments.session)
        if session.name != arguments.session or arguments.session in {"", ".", ".."}:
            raise ValueError("--session must be a single safe directory name")
        output_folder = OWN_SYNTH_ROOT / session
    else:
        output_folder = arguments.output_folder
    levels = LEVELS if arguments.level == "all" else (arguments.level,)
    generate_folder(arguments.input_folder, output_folder, levels, arguments.seed)


if __name__ == "__main__":
    main()

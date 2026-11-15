"""Sample and conservatively filter close-up patches from the raw source images."""

from __future__ import annotations

import csv
import hashlib
import logging
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError


ROOT = Path(__file__).resolve().parent
MANIFEST_PATH = ROOT / "data" / "manifest.csv"
PATCH_ROOT = ROOT / "data" / "patches"
PATCH_MANIFEST_PATH = ROOT / "data" / "patches_manifest.csv"
GRID_ROOT = ROOT / "reports" / "patches"
LABELS = ("clean", "dusty", "cracked")
PATCH_SIZE = (224, 224)
SEED = 42
LOGGER = logging.getLogger("build_patches")


@dataclass(frozen=True)
class ParentImage:
    source: str
    label: str
    path: Path
    group_id: str


@dataclass(frozen=True)
class CropCandidate:
    image: Image.Image
    area_fraction: float
    bbox: tuple[int, int, int, int]


def load_parents() -> list[ParentImage]:
    if not MANIFEST_PATH.is_file():
        raise FileNotFoundError(f"Cleaned source manifest is missing: {MANIFEST_PATH}")
    parents: list[ParentImage] = []
    with MANIFEST_PATH.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if row["role"] != "train_candidate" or row["label"] not in LABELS:
                continue
            image_path = ROOT / row["source_path"]
            if not image_path.is_file():
                raise FileNotFoundError(f"Source image listed in manifest is missing: {image_path}")
            parents.append(
                ParentImage(
                    source=row["source"],
                    label=row["label"],
                    path=image_path,
                    group_id=row["group_id"],
                )
            )
    return sorted(parents, key=lambda parent: (parent.label, parent.source, str(parent.path)))


def square_crop_dimensions(
    image_size: tuple[int, int], area_fraction: float, aspect_ratio: float
) -> tuple[int, int] | None:
    image_width, image_height = image_size
    crop_height = int(round(np.sqrt(area_fraction * image_width * image_height / aspect_ratio)))
    crop_width = int(round(crop_height * aspect_ratio))
    if crop_width > image_width or crop_height > image_height:
        return None
    return max(1, crop_width), max(1, crop_height)


def sample_crop(image: Image.Image, rng: random.Random) -> CropCandidate | None:
    image_width, image_height = image.size
    for _ in range(40):
        area_fraction = rng.uniform(0.15, 0.60)
        aspect_ratio = rng.uniform(0.90, 1.10)
        dimensions = square_crop_dimensions(
            (image_width, image_height), area_fraction, aspect_ratio
        )
        if dimensions is None:
            continue
        crop_width, crop_height = dimensions
        actual_area = crop_width * crop_height / (image_width * image_height)
        if not 0.15 <= actual_area <= 0.60:
            continue
        left = rng.randint(0, image_width - crop_width)
        top = rng.randint(0, image_height - crop_height)
        bbox = (left, top, left + crop_width, top + crop_height)
        patch = image.crop(bbox).resize(PATCH_SIZE, Image.Resampling.LANCZOS)
        return CropCandidate(patch, actual_area, bbox)
    return None


def text_or_watermark_dominated(rgb: np.ndarray) -> bool:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    threshold = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        7,
    )
    count, _, stats, _ = cv2.connectedComponentsWithStats(threshold, 8)
    candidates: list[tuple[int, int, int]] = []
    coverage = 0
    for index in range(1, count):
        x, y, width, height, area = (int(value) for value in stats[index])
        if 2 <= width <= 38 and 4 <= height <= 34 and 5 <= area <= 260:
            candidates.append((x, y, area))
            coverage += area
    if len(candidates) < 70 or not 0.025 <= coverage / (rgb.shape[0] * rgb.shape[1]) <= 0.35:
        return False
    y_values = np.asarray([point[1] for point in candidates])
    x_values = np.asarray([point[0] for point in candidates])
    if np.ptp(x_values) < rgb.shape[1] * 0.55:
        return False
    row_bins = np.histogram(y_values, bins=8, range=(0, rgb.shape[0]))[0]
    return int(np.count_nonzero(row_bins >= 8)) >= 2


def patch_drop_reasons(image: Image.Image) -> set[str]:
    rgb = np.asarray(image.convert("RGB"))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 80, 160)
    edge_density = float(np.mean(edges > 0))
    reasons: set[str] = set()

    overexposed_fraction = float(np.mean(np.all(rgb >= 250, axis=2)))
    if overexposed_fraction >= 0.85:
        reasons.add("overexposed")

    if float(np.std(gray)) < 4.0 and edge_density < 0.012:
        reasons.add("extremely_uniform")

    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    sky_mask = (
        (hsv[:, :, 0] >= 90)
        & (hsv[:, :, 0] <= 140)
        & (hsv[:, :, 1] >= 55)
        & (hsv[:, :, 2] >= 150)
    )
    if float(np.mean(sky_mask)) >= 0.88 and edge_density < 0.025:
        reasons.add("mostly_sky")

    if text_or_watermark_dominated(rgb):
        reasons.add("text_or_watermark_dominated")
    return reasons


def parent_key(parent: ParentImage) -> str:
    identity = f"{parent.source}\0{parent.path}\0{parent.group_id}"
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16]


def create_candidate_crops(
    parents: list[ParentImage], rng: random.Random
) -> list[dict[str, object]]:
    candidates: list[dict[str, object]] = []
    for parent in parents:
        try:
            with Image.open(parent.path) as opened:
                original = ImageOps.exif_transpose(opened).convert("RGB")
                patch_count = rng.randint(3, 6)
                for crop_index in range(patch_count):
                    crop = sample_crop(original, rng)
                    if crop is None:
                        LOGGER.warning(
                            "No near-square crop in requested area range for %s",
                            parent.path,
                        )
                        break
                    candidates.append(
                        {
                            "parent": parent,
                            "crop_index": crop_index,
                            "crop": crop,
                            "patch_id": f"{parent_key(parent)}_{crop_index:02d}",
                            "all_reasons": patch_drop_reasons(crop.image),
                        }
                    )
        except (OSError, UnidentifiedImageError, ValueError) as error:
            LOGGER.error("Cannot read source image %s: %s", parent.path, error)
            raise
    return candidates


def filter_candidates(
    candidates: list[dict[str, object]], disabled_reasons: set[str]
) -> tuple[list[dict[str, object]], Counter[tuple[str, str]], Counter[tuple[str, str]]]:
    kept: list[dict[str, object]] = []
    kept_counts: Counter[tuple[str, str]] = Counter()
    dropped_counts: Counter[tuple[str, str]] = Counter()
    for candidate in candidates:
        parent = candidate["parent"]
        crop = candidate["crop"]
        assert isinstance(parent, ParentImage)
        assert isinstance(crop, CropCandidate)
        all_reasons = candidate["all_reasons"]
        assert isinstance(all_reasons, set)
        reasons = all_reasons - disabled_reasons
        candidate["reasons"] = sorted(reasons)
        key = (parent.source, parent.label)
        if reasons:
            dropped_counts[key] += 1
        else:
            kept.append(candidate)
            kept_counts[key] += 1
    return kept, kept_counts, dropped_counts


def class_drop_rates(
    candidates: list[dict[str, object]], disabled_reasons: set[str]
) -> dict[str, float]:
    totals: Counter[str] = Counter()
    dropped: Counter[str] = Counter()
    for candidate in candidates:
        parent = candidate["parent"]
        assert isinstance(parent, ParentImage)
        totals[parent.label] += 1
        all_reasons = candidate["all_reasons"]
        assert isinstance(all_reasons, set)
        if all_reasons - disabled_reasons:
            dropped[parent.label] += 1
    return {
        label: dropped[label] / totals[label]
        for label in totals
        if totals[label]
    }


def relax_filter_if_skewed(
    candidates: list[dict[str, object]],
) -> set[str]:
    disabled: set[str] = set()
    all_reasons = {
        reason
        for candidate in candidates
        for reason in candidate["all_reasons"]  # type: ignore[union-attr]
    }
    while True:
        rates = class_drop_rates(candidates, disabled)
        if len(rates) < 2 or max(rates.values()) - min(rates.values()) <= 0.10:
            return disabled
        remaining = all_reasons - disabled
        if not remaining:
            LOGGER.warning(
                "Filter drop-rate difference remains above 10 percentage points "
                "after all filter reasons were disabled: %s",
                rates,
            )
            return disabled
        reason_skew: dict[str, float] = {}
        for reason in remaining:
            per_class: dict[str, float] = {}
            for candidate in candidates:
                parent = candidate["parent"]
                assert isinstance(parent, ParentImage)
                reasons = candidate["all_reasons"]
                assert isinstance(reasons, set)
                if reason in reasons:
                    per_class[parent.label] = per_class.get(parent.label, 0.0) + 1
            class_totals = Counter(
                candidate["parent"].label
                for candidate in candidates
                if isinstance(candidate["parent"], ParentImage)
            )
            for label, class_total in class_totals.items():
                per_class[label] = per_class.get(label, 0) / class_total
            if per_class:
                reason_skew[reason] = max(per_class.values()) - min(
                    per_class.values()
                )
        if not reason_skew:
            return disabled
        to_disable = max(reason_skew, key=reason_skew.get)
        disabled.add(to_disable)
        LOGGER.warning(
            "Relaxing patch filter %s because class drop rates differ by more "
            "than 10 percentage points: %s",
            to_disable,
            rates,
        )


def write_patch_manifest(candidates: list[dict[str, object]]) -> None:
    fields = [
        "patch_path",
        "parent_path",
        "group_id",
        "label",
        "source",
        "crop_index",
        "crop_area_fraction",
        "left",
        "top",
        "right",
        "bottom",
        "status",
        "drop_reasons",
    ]
    current_patch_paths: set[Path] = set()
    with PATCH_MANIFEST_PATH.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for candidate in candidates:
            parent = candidate["parent"]
            crop = candidate["crop"]
            assert isinstance(parent, ParentImage)
            assert isinstance(crop, CropCandidate)
            patch_path = ""
            reasons = candidate["reasons"]
            if not reasons:
                patch_file = (
                    PATCH_ROOT
                    / parent.label
                    / f"{parent.source}__{candidate['patch_id']}.jpg"
                )
                patch_file.parent.mkdir(parents=True, exist_ok=True)
                crop.image.save(patch_file, format="JPEG", quality=94, optimize=True)
                patch_path = str(patch_file.relative_to(ROOT))
                current_patch_paths.add(patch_file.resolve())
            left, top, right, bottom = crop.bbox
            writer.writerow(
                {
                    "patch_path": patch_path,
                    "parent_path": str(parent.path.relative_to(ROOT)),
                    "group_id": parent.group_id,
                    "label": parent.label,
                    "source": parent.source,
                    "crop_index": candidate["crop_index"],
                    "crop_area_fraction": f"{crop.area_fraction:.6f}",
                    "left": left,
                    "top": top,
                    "right": right,
                    "bottom": bottom,
                    "status": "dropped" if reasons else "kept",
                    "drop_reasons": ";".join(reasons),
                }
            )
    patch_root = PATCH_ROOT.resolve()
    for existing_path in PATCH_ROOT.rglob("*.jpg"):
        resolved_path = existing_path.resolve()
        if resolved_path.is_relative_to(patch_root) and resolved_path not in current_patch_paths:
            resolved_path.unlink()


def save_sample_grids(kept: list[dict[str, object]]) -> None:
    from PIL import ImageDraw

    GRID_ROOT.mkdir(parents=True, exist_ok=True)
    grouped: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for candidate in kept:
        parent = candidate["parent"]
        assert isinstance(parent, ParentImage)
        grouped[(parent.label, parent.source)].append(candidate)
    for (label, source), items in sorted(grouped.items()):
        rng = random.Random(f"{SEED}:{label}:{source}")
        chosen = sorted(rng.sample(items, min(16, len(items))), key=lambda item: str(item["patch_id"]))
        grid = Image.new("RGB", (4 * 240, 4 * 260), (245, 245, 245))
        draw = ImageDraw.Draw(grid)
        for index, candidate in enumerate(chosen):
            crop = candidate["crop"]
            parent = candidate["parent"]
            assert isinstance(crop, CropCandidate)
            assert isinstance(parent, ParentImage)
            x = (index % 4) * 240
            y = (index // 4) * 260
            grid.paste(crop.image, (x + 8, y + 4))
            draw.text(
                (x + 5, y + 232),
                Path(parent.path).name[:36],
                fill=(20, 20, 20),
            )
        grid.save(GRID_ROOT / f"{label}_{source}.png")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    PATCH_ROOT.mkdir(parents=True, exist_ok=True)
    PATCH_MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    GRID_ROOT.mkdir(parents=True, exist_ok=True)

    parents = load_parents()
    LOGGER.info("Eligible original images: %d", len(parents))
    for (source, label), count in sorted(
        Counter((parent.source, parent.label) for parent in parents).items()
    ):
        LOGGER.info("parents source=%s class=%s count=%d", source, label, count)

    candidates = create_candidate_crops(parents, random.Random(SEED))
    LOGGER.info("Candidate crops sampled from full-resolution originals: %d", len(candidates))
    disabled_reasons = relax_filter_if_skewed(candidates)
    if disabled_reasons:
        LOGGER.warning("Filters relaxed by class-balance guard: %s", sorted(disabled_reasons))
    kept, kept_counts, dropped_counts = filter_candidates(candidates, disabled_reasons)
    write_patch_manifest(candidates)
    save_sample_grids(kept)

    classes = sorted({parent.label for parent in parents})
    for label in classes:
        class_kept = sum(count for (source, item_label), count in kept_counts.items() if item_label == label)
        class_dropped = sum(count for (source, item_label), count in dropped_counts.items() if item_label == label)
        total = class_kept + class_dropped
        LOGGER.info(
            "class=%s kept=%d dropped=%d drop_rate=%.1f%%",
            label,
            class_kept,
            class_dropped,
            100 * class_dropped / total if total else 0.0,
        )
    for source, label in sorted(set(kept_counts) | set(dropped_counts)):
        LOGGER.info(
            "source=%s class=%s kept=%d dropped=%d",
            source,
            label,
            kept_counts[(source, label)],
            dropped_counts[(source, label)],
        )
    reason_counts = Counter(
        reason
        for candidate in candidates
        for reason in candidate["reasons"]  # type: ignore[union-attr]
    )
    LOGGER.info("Final kept patches: %d", len(kept))
    LOGGER.info("Dropped patches by reason: %s", dict(sorted(reason_counts.items())))
    LOGGER.info("Patch manifest: %s", PATCH_MANIFEST_PATH)
    LOGGER.info("Review grids: %s", GRID_ROOT)


if __name__ == "__main__":
    main()

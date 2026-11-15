"""Rebuild experimental cracked patches using larger crops from raw images."""

from __future__ import annotations

import csv
import logging
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageOps, UnidentifiedImageError

from build_patches import PATCH_SIZE, ROOT, patch_drop_reasons


PATCH_ROOT = ROOT / "data" / "cracked_patches"
PATCH_MANIFEST = ROOT / "data" / "cracked_patches_manifest.csv"
EXCLUDED_MANIFEST = ROOT / "data" / "cracked_parents_excluded.csv"
GRID_PATH = ROOT / "reports" / "patches" / "cracked_experimental_review.png"
SEED = 42
LOGGER = logging.getLogger("build_cracked_experiment")


@dataclass(frozen=True)
class Parent:
    source: str
    path: Path
    group_id: str


def load_parents() -> list[Parent]:
    path = ROOT / "data" / "patches_manifest.csv"
    parents: dict[str, Parent] = {}
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if row["label"] != "cracked" or row["status"] != "kept":
                continue
            parent_path = row["parent_path"]
            parents[parent_path] = Parent(
                source=row["source"],
                path=ROOT / parent_path,
                group_id=row["group_id"],
            )
    return sorted(parents.values(), key=lambda parent: str(parent.path))


def sample_large_crop(
    image: Image.Image, rng: random.Random
) -> tuple[Image.Image, tuple[int, int, int, int], float] | None:
    width, height = image.size
    for _ in range(120):
        area_fraction = rng.uniform(0.50, 1.0)
        aspect_ratio = rng.uniform(0.90, 1.10)
        crop_height = round((area_fraction * width * height / aspect_ratio) ** 0.5)
        crop_width = round(crop_height * aspect_ratio)
        if crop_width > width or crop_height > height:
            continue
        actual_area = crop_width * crop_height / (width * height)
        if not 0.50 <= actual_area <= 1.0:
            continue
        left = rng.randint(0, width - crop_width)
        top = rng.randint(0, height - crop_height)
        bbox = (left, top, left + crop_width, top + crop_height)
        patch = image.crop(bbox).resize(PATCH_SIZE, Image.Resampling.LANCZOS)
        return patch, bbox, actual_area
    return None


def save_patch_grid(samples: list[tuple[Path, Image.Image]]) -> None:
    from PIL import ImageDraw

    grid = Image.new("RGB", (4 * 240, 4 * 260), (245, 245, 245))
    draw = ImageDraw.Draw(grid)
    for index, (parent_path, patch) in enumerate(samples):
        x = (index % 4) * 240
        y = (index // 4) * 260
        grid.paste(patch, (x + 8, y + 4))
        draw.text((x + 5, y + 232), parent_path.name[:36], fill=(20, 20, 20))
    GRID_PATH.parent.mkdir(parents=True, exist_ok=True)
    grid.save(GRID_PATH)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    PATCH_ROOT.mkdir(parents=True, exist_ok=True)
    parents = load_parents()
    rng = random.Random(SEED)
    rows: list[dict[str, str]] = []
    excluded: list[dict[str, str]] = []
    samples_by_parent: dict[Path, list[Image.Image]] = {}
    generated_patch_paths: set[Path] = set()
    kept_counts: Counter[str] = Counter()
    dropped_counts: Counter[str] = Counter()
    parent_patch_counts: Counter[str] = Counter()
    patch_index = 0

    for parent in parents:
        try:
            with Image.open(parent.path) as opened:
                original = ImageOps.exif_transpose(opened).convert("RGB")
                first_crop = sample_large_crop(original, rng)
                if first_crop is None:
                    excluded.append(
                        {
                            "parent_path": str(parent.path.relative_to(ROOT)),
                            "group_id": parent.group_id,
                            "reason": "cannot fit a near-square crop with area >= 0.50",
                        }
                    )
                    continue
                crops = [first_crop]
                for _ in range(rng.randint(2, 5)):
                    crop = sample_large_crop(original, rng)
                    if crop is not None:
                        crops.append(crop)
                for crop_index, (patch, bbox, area_fraction) in enumerate(crops):
                    reasons = patch_drop_reasons(patch) - {"text_or_watermark_dominated"}
                    output_path = PATCH_ROOT / f"{parent.group_id}_{patch_index:04d}.jpg"
                    if not reasons:
                        patch.save(output_path, format="JPEG", quality=94, optimize=True)
                        generated_patch_paths.add(output_path.resolve())
                        kept_counts[parent.source] += 1
                        parent_patch_counts[str(parent.path)] += 1
                        samples_by_parent.setdefault(parent.path, []).append(patch.copy())
                    else:
                        dropped_counts[parent.source] += 1
                    left, top, right, bottom = bbox
                    rows.append(
                        {
                            "patch_path": (
                                str(output_path.relative_to(ROOT)) if not reasons else ""
                            ),
                            "parent_path": str(parent.path.relative_to(ROOT)),
                            "group_id": parent.group_id,
                            "label": "cracked",
                            "source": parent.source,
                            "crop_index": str(crop_index),
                            "crop_area_fraction": f"{area_fraction:.6f}",
                            "left": str(left),
                            "top": str(top),
                            "right": str(right),
                            "bottom": str(bottom),
                            "status": "dropped" if reasons else "kept",
                            "drop_reasons": ";".join(sorted(reasons)),
                        }
                    )
                    patch_index += 1
        except (OSError, UnidentifiedImageError, ValueError) as error:
            LOGGER.error("Cannot read cracked source image %s: %s", parent.path, error)
            raise

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
    with PATCH_MANIFEST.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    with EXCLUDED_MANIFEST.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["parent_path", "group_id", "reason"])
        writer.writeheader()
        writer.writerows(excluded)
    patch_root = PATCH_ROOT.resolve()
    for existing_path in PATCH_ROOT.rglob("*.jpg"):
        resolved_path = existing_path.resolve()
        if resolved_path.is_relative_to(patch_root) and resolved_path not in generated_patch_paths:
            resolved_path.unlink()
    sample_rng = random.Random(SEED)
    sample_parents = sorted(samples_by_parent)
    sample_rng.shuffle(sample_parents)
    samples = [
        (parent, sample_rng.choice(samples_by_parent[parent]))
        for parent in sample_parents[:16]
    ]
    save_patch_grid(samples)

    LOGGER.info("Cracked parent images eligible before large-crop check: %d", len(parents))
    LOGGER.info("Parents excluded because near-square 0.50 area crops do not fit: %d", len(excluded))
    LOGGER.info("Parents with large-crop patches: %d", len(parent_patch_counts))
    LOGGER.info("Kept cracked patches: %d", sum(kept_counts.values()))
    LOGGER.info("Dropped cracked patches: %d", sum(dropped_counts.values()))
    LOGGER.info("Per-source kept patches: %s", dict(sorted(kept_counts.items())))
    LOGGER.info("Per-source dropped patches: %s", dict(sorted(dropped_counts.items())))
    LOGGER.info("Review grid: %s", GRID_PATH)
    LOGGER.info("Patch manifest: %s", PATCH_MANIFEST)


if __name__ == "__main__":
    main()

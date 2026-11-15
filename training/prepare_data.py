"""Validate, de-duplicate, resize, and inventory the downloaded image datasets."""

from __future__ import annotations

import csv
import hashlib
import logging
import random
import shutil
from dataclasses import dataclass
from pathlib import Path

import imagehash
from PIL import Image, ImageDraw, ImageOps, UnidentifiedImageError


ROOT = Path(__file__).resolve().parent
RAW_ROOT = ROOT / "data" / "raw"
PROCESSED_ROOT = ROOT / "data" / "processed"
HELDOUT_ROOT = ROOT / "data" / "heldout"
REPORTS_ROOT = ROOT / "reports"
MANIFEST_PATH = ROOT / "data" / "manifest.csv"
REJECTIONS_PATH = ROOT / "data" / "rejected.csv"
IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
LOG = logging.getLogger("prepare_data")

SOURCE_URLS = {
    "pv_defect": "https://www.kaggle.com/datasets/alicjalena/pv-panel-defect-dataset",
    "dust_detection": "https://www.kaggle.com/datasets/hemanthsai7/solar-panel-dust-detection",
}
SOURCE_LICENSES = {
    "pv_defect": "CC-BY-NC-SA-4.0",
    "dust_detection": "CC0-1.0",
}
CLASS_MAPPING = {
    "clean": "clean",
    "dusty": "dusty",
    "physical-damage": "cracked",
    "bird-drop": "bird-drop",
    "snow-covered": "snow-covered",
    "electrical-damage": "electrical-damage",
}
HELDOUT_CLASSES = {
    "ambiguous-near-duplicate",
    "bird-drop",
    "snow-covered",
    "electrical-damage",
    "unclear-physical-damage",
}

# These are obvious elevated/aerial views from the visual inspection grids.
VISUAL_REJECT_NAMES = {
    "Imgclean_1189_0.jpg",
    "Imgclean_1205_0.jpg",
    "Imgclean_25_0.jpg",
    "Imgclean_442_0.jpg",
    "Imgdirty_836_1.jpg",
    "Imgdirty_989_1.jpg",
    "Physical (35).jpg",
    "Physical (40).jpg",
}
VISUAL_HELDOUT_NAMES = {
    "Physical (30).jpg",
    "Physical (38).jpg",
    "Physical (62).jpg",
    "Physical (115).jpg",
}


@dataclass(frozen=True)
class ImageRecord:
    source: str
    source_class: str
    label: str
    path: Path
    exact_hash: str
    perceptual_hash: int
    width: int
    height: int


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_class_for(path: Path, dataset_root: Path) -> str:
    relative_parts = path.relative_to(dataset_root).parts
    if len(relative_parts) < 2:
        return "uncategorized"
    return relative_parts[-2].strip().lower().replace("_", "-").replace(" ", "-")


def source_for(dataset_root: Path) -> str:
    return dataset_root.name


def to_rgb(opened: Image.Image) -> Image.Image:
    image = ImageOps.exif_transpose(opened)
    if image.mode == "P" and "transparency" in image.info:
        rgba = image.convert("RGBA")
        background = Image.new("RGB", rgba.size, "white")
        background.paste(rgba, mask=rgba.getchannel("A"))
        return background
    return image.convert("RGB")


def scan_records() -> tuple[list[ImageRecord], list[dict[str, str]]]:
    records: list[ImageRecord] = []
    rejected: list[dict[str, str]] = []
    for dataset_root in sorted(p for p in RAW_ROOT.iterdir() if p.is_dir()):
        for path in sorted(dataset_root.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            source_class = source_class_for(path, dataset_root)
            label = CLASS_MAPPING.get(source_class)
            if label is None:
                rejected.append(
                    {"path": str(path), "reason": f"unmapped class: {source_class}"}
                )
                continue
            if path.name in VISUAL_REJECT_NAMES:
                rejected.append({"path": str(path), "reason": "visually identified aerial view"})
                continue
            if path.name in VISUAL_HELDOUT_NAMES:
                label = "unclear-physical-damage"
            try:
                with Image.open(path) as opened:
                    image = to_rgb(opened)
                    width, height = image.size
                    if width < 64 or height < 64:
                        rejected.append({"path": str(path), "reason": "image under 64 px"})
                        continue
                    perceptual_hash = int(str(imagehash.phash(image)), 16)
                records.append(
                    ImageRecord(
                        source=source_for(dataset_root),
                        source_class=source_class,
                        label=label,
                        path=path,
                        exact_hash=sha256_file(path),
                        perceptual_hash=perceptual_hash,
                        width=width,
                        height=height,
                    )
                )
            except (OSError, UnidentifiedImageError, ValueError) as error:
                rejected.append({"path": str(path), "reason": f"unreadable image: {error}"})
    return records, rejected


def deduplicate_exact(
    records: list[ImageRecord],
) -> tuple[list[ImageRecord], list[dict[str, str]]]:
    by_hash: dict[str, list[ImageRecord]] = {}
    for record in records:
        by_hash.setdefault(record.exact_hash, []).append(record)
    retained: list[ImageRecord] = []
    rejected: list[dict[str, str]] = []
    for duplicates in by_hash.values():
        duplicates.sort(key=lambda item: (item.source, str(item.path)))
        labels = {item.label for item in duplicates}
        if len(labels) > 1:
            rejected.extend(
                {"path": str(item.path), "reason": "exact duplicate has conflicting labels"}
                for item in duplicates
            )
            continue
        retained.append(duplicates[0])
        rejected.extend(
            {"path": str(item.path), "reason": f"exact duplicate of {duplicates[0].path}"}
            for item in duplicates[1:]
        )
    return retained, rejected


def near_duplicate_groups(records: list[ImageRecord]) -> dict[str, str]:
    parent = list(range(len(records)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(first: int, second: int) -> None:
        first_root = find(first)
        second_root = find(second)
        if first_root != second_root:
            parent[second_root] = first_root

    for index, record in enumerate(records):
        for previous in range(index):
            if (record.perceptual_hash ^ records[previous].perceptual_hash).bit_count() <= 4:
                union(index, previous)

    members: dict[int, list[int]] = {}
    for index in range(len(records)):
        members.setdefault(find(index), []).append(index)
    group_ids: dict[str, str] = {}
    for indices in members.values():
        group_key = min(records[index].exact_hash for index in indices)
        group_id = hashlib.sha256(group_key.encode("ascii")).hexdigest()[:16]
        for index in indices:
            group_ids[records[index].exact_hash] = group_id
    return group_ids


def processed_path(record: ImageRecord, label: str) -> Path:
    if label in HELDOUT_CLASSES:
        return (
            PROCESSED_ROOT
            / "heldout"
            / label
            / record.source
            / f"{record.exact_hash[:16]}.jpg"
        )
    return (
        PROCESSED_ROOT
        / label
        / record.source
        / f"{record.exact_hash[:16]}.jpg"
    )


def cache_image(record: ImageRecord, target: Path) -> None:
    with Image.open(record.path) as opened:
        image = to_rgb(opened)
        width, height = image.size
        scale = 256 / min(width, height)
        size = (round(width * scale), round(height * scale))
        image.resize(size, Image.Resampling.LANCZOS).save(
            target, format="JPEG", quality=92, optimize=True
        )


def create_sample_grids(rows: list[dict[str, str]]) -> None:
    by_label: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        by_label.setdefault(row["label"], []).append(row)
    for label, label_rows in sorted(by_label.items()):
        label_rows.sort(key=lambda row: row["processed_path"])
        chosen = random.Random(42).sample(label_rows, min(16, len(label_rows)))
        canvas = Image.new("RGB", (4 * 320, 4 * 250), (245, 245, 245))
        draw = ImageDraw.Draw(canvas)
        for index, row in enumerate(chosen):
            x = (index % 4) * 320
            y = (index // 4) * 250
            with Image.open(ROOT / row["processed_path"]) as opened:
                image = opened.convert("RGB")
                image.thumbnail((304, 210))
                canvas.paste(image, (x + (304 - image.width) // 2 + 8, y + 4))
            draw.text(
                (x + 5, y + 218),
                f'{row["source"]}/{Path(row["source_path"]).name}'[:45],
                fill=(20, 20, 20),
            )
        canvas.save(REPORTS_ROOT / f"sample_{label}.png")


def write_manifest(rows: list[dict[str, str]]) -> None:
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "processed_path",
        "source_path",
        "source",
        "source_class",
        "label",
        "group_id",
        "width",
        "height",
        "role",
    ]
    with MANIFEST_PATH.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def remove_stale_cached_images(previous_paths: set[str], current_paths: set[str]) -> None:
    processed_root = PROCESSED_ROOT.resolve()
    for relative_path in previous_paths - current_paths:
        stale_path = (ROOT / relative_path).resolve()
        if stale_path.is_relative_to(processed_root) and stale_path.is_file():
            stale_path.unlink()


def write_rejections(rejected: list[dict[str, str]]) -> None:
    with REJECTIONS_PATH.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["path", "reason"])
        writer.writeheader()
        writer.writerows(rejected)


def write_dataset_notes(rows: list[dict[str, str]], rejected: list[dict[str, str]]) -> None:
    counts: dict[tuple[str, str], int] = {}
    source_class_counts: dict[tuple[str, str, str], int] = {}
    for row in rows:
        counts[(row["source"], row["label"])] = (
            counts.get((row["source"], row["label"]), 0) + 1
        )
        key = (row["source"], row["label"], row["role"])
        source_class_counts[key] = source_class_counts.get(key, 0) + 1
    lines = [
        "# Dataset inventory",
        "",
        "| Source | URL | License reported by Kaggle | Images after validation and exact de-duplication | Classes | Decision |",
        "|---|---|---|---:|---|---|",
    ]
    decisions = {
        "pv_defect": "Accepted selectively: clean, dusty, and physical-damage (mapped to cracked); bird-drop, snow, and electrical damage are held out. Visual inspection found watermarks and non-crack physical-damage examples; selected unclear examples were held out.",
        "dust_detection": "Accepted selectively: clean and dusty. RGB panel images are present, but some samples are elevated/aerial views; inspected aerial examples were excluded.",
    }
    for source in sorted(SOURCE_URLS):
        source_rows = [row for row in rows if row["source"] == source]
        source_classes = sorted({row["label"] for row in source_rows})
        source_count = len(source_rows)
        lines.append(
            f"| {source} | {SOURCE_URLS[source]} | {SOURCE_LICENSES[source]} | "
            f"{source_count} | {', '.join(source_classes)} | {decisions[source]} |"
        )
    lines.extend(
        [
            "",
            "## Counts by source and mapped class",
            "",
            "| Source | Mapped class | Role | Images |",
            "|---|---|---|---:|",
        ]
    )
    for (source, label, role), count in sorted(source_class_counts.items()):
        lines.append(f"| {source} | {label} | {role} | {count} |")
    rejection_counts: dict[str, int] = {}
    for item in rejected:
        reason = item["reason"].split(" of ", maxsplit=1)[0]
        rejection_counts[reason] = rejection_counts.get(reason, 0) + 1
    lines.extend(
        [
            "",
            "Counts in this inventory include valid decoded images after exact-duplicate removal. Images under 64 px, unreadable files, known aerial examples, and exact duplicates are excluded. Near-duplicate images are retained and have group IDs in `data/manifest.csv` so a later split can keep each group together.",
            "",
            "Exclusions recorded: "
            + "; ".join(
                f"{reason}: {count}"
                for reason, count in sorted(rejection_counts.items())
            )
            + ".",
            "",
            "The public class names are mapped as follows: `Clean` → `clean`, `Dusty` → `dusty`, and visibly cracked `Physical-Damage` examples → `cracked`. `Bird-drop`, `Snow-Covered`, `Electrical-damage`, selected visually unclear physical-damage examples, and near-duplicate groups carrying conflicting labels are kept in `data/heldout/` and are not intended for training. A small visual sample of each retained/held-out label is in `reports/sample_*.png`.",
            "",
            "Both Kaggle license identifiers are recorded as displayed by the Kaggle CLI. The CC-BY-NC-SA-4.0 dataset is non-commercial and share-alike; intended deployment/commercial use must be checked against its terms before use.",
            "",
            "Rejected-file details are recorded in `data/rejected.csv`.",
            "",
        ]
    )
    (ROOT / "DATASETS.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    for directory in (PROCESSED_ROOT, HELDOUT_ROOT, REPORTS_ROOT):
        directory.mkdir(parents=True, exist_ok=True)

    previous_paths: set[str] = set()
    if MANIFEST_PATH.is_file():
        with MANIFEST_PATH.open(newline="", encoding="utf-8") as stream:
            previous_paths = {
                row["processed_path"] for row in csv.DictReader(stream)
            }

    records, invalid = scan_records()
    LOG.info("Decoded valid images before exact de-duplication: %d", len(records))
    records, duplicate_rejections = deduplicate_exact(records)
    rejected = invalid + duplicate_rejections
    LOG.info(
        "Kept after exact de-duplication: %d; excluded: %d",
        len(records),
        len(rejected),
    )
    groups = near_duplicate_groups(records)
    labels_by_group: dict[str, set[str]] = {}
    for record in records:
        labels_by_group.setdefault(groups[record.exact_hash], set()).add(record.label)
    ambiguous_groups = {
        group_id for group_id, labels in labels_by_group.items() if len(labels) > 1
    }
    rows: list[dict[str, str]] = []
    for record in records:
        group_id = groups[record.exact_hash]
        label = (
            "ambiguous-near-duplicate"
            if group_id in ambiguous_groups
            else record.label
        )
        cached_path = processed_path(record, label)
        cached_path.parent.mkdir(parents=True, exist_ok=True)
        cache_image(record, cached_path)
        role = "heldout" if label in HELDOUT_CLASSES else "train_candidate"
        if role == "heldout":
            heldout_path = (
                HELDOUT_ROOT
                / record.source
                / label
                / f"{record.exact_hash[:16]}{record.path.suffix.lower()}"
            )
            heldout_path.parent.mkdir(parents=True, exist_ok=True)
            if not heldout_path.exists():
                shutil.copy2(record.path, heldout_path)
        rows.append(
            {
                "processed_path": str(cached_path.relative_to(ROOT)),
                "source_path": str(record.path.relative_to(ROOT)),
                "source": record.source,
                "source_class": record.source_class,
                "label": label,
                "group_id": group_id,
                "width": str(record.width),
                "height": str(record.height),
                "role": role,
            }
        )

    current_paths = {row["processed_path"] for row in rows}
    remove_stale_cached_images(previous_paths, current_paths)
    write_manifest(rows)
    write_rejections(rejected)
    create_sample_grids(rows)
    write_dataset_notes(rows, rejected)

    counts: dict[tuple[str, str], int] = {}
    group_members: dict[str, int] = {}
    for row in rows:
        key = (row["source"], row["label"])
        counts[key] = counts.get(key, 0) + 1
        group_members[row["group_id"]] = group_members.get(row["group_id"], 0) + 1
    for (source, label), count in sorted(counts.items()):
        LOG.info("source=%s class=%s count=%d", source, label, count)
    LOG.info(
        "Near-duplicate groups with multiple retained images: %d",
        sum(size > 1 for size in group_members.values()),
    )
    LOG.info("Excluded records: %d", len(rejected))
    rejection_reasons: dict[str, int] = {}
    for item in rejected:
        reason = item["reason"].split(" of ", maxsplit=1)[0]
        rejection_reasons[reason] = rejection_reasons.get(reason, 0) + 1
    for reason, count in sorted(rejection_reasons.items()):
        LOG.info("excluded reason=%s count=%d", reason, count)


if __name__ == "__main__":
    main()

"""Grouped OOF validation for prioritizing possible label issues."""

from __future__ import annotations

import csv
import json
import logging
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageOps
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from torchvision.models import MobileNet_V2_Weights
from torchvision.transforms import InterpolationMode


ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
SPLIT_ROOT = ROOT / "splits" / "main"
REPORT_ROOT = ROOT / "reports"
LABELS = ("clean", "dusty")
LABEL_TO_INDEX = {label: index for index, label in enumerate(LABELS)}
INPUT_SIZE = 224
SEED = 42
FOLDS = 5
BATCH_SIZE = 48
LOGGER = logging.getLogger("cross_validate_suspects")


@dataclass(frozen=True)
class CVRow:
    patch_path: Path
    relative_patch_path: str
    parent_path: str
    group_id: str
    label: int
    source: str


class PatchDataset(Dataset[torch.Tensor]):
    def __init__(self, rows: list[CVRow]) -> None:
        self.rows = rows
        self.transform = transforms.Compose(
            [
                transforms.Resize(256, interpolation=InterpolationMode.BILINEAR),
                transforms.CenterCrop(INPUT_SIZE),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.485, 0.456, 0.406),
                    (0.229, 0.224, 0.225),
                ),
            ]
        )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> torch.Tensor:
        with Image.open(self.rows[index].patch_path) as opened:
            image = ImageOps.exif_transpose(opened).convert("RGB")
        return self.transform(image)


def load_train_val_rows() -> list[CVRow]:
    rows: list[CVRow] = []
    for split_name in ("train", "val"):
        split_path = SPLIT_ROOT / f"{split_name}.csv"
        if not split_path.is_file():
            raise FileNotFoundError(f"Required {split_name} split not found: {split_path}")
        with split_path.open(newline="", encoding="utf-8") as stream:
            for item in csv.DictReader(stream):
                if item["label"] not in LABEL_TO_INDEX:
                    raise ValueError(f"Unexpected label {item['label']!r} in {split_path}")
                relative_path = item["patch_path"]
                image_path = ROOT / relative_path
                if not image_path.is_file():
                    raise FileNotFoundError(f"Split image missing: {image_path}")
                rows.append(
                    CVRow(
                        patch_path=image_path,
                        relative_patch_path=relative_path,
                        parent_path=item["parent_path"],
                        group_id=item["group_id"],
                        label=LABEL_TO_INDEX[item["label"]],
                        source=item["source"],
                    )
                )
    if not rows:
        raise ValueError("No train+validation patches were loaded")

    parent_metadata: dict[str, tuple[int, str, str]] = {}
    group_parents: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        metadata = (row.label, row.source, row.group_id)
        if row.parent_path in parent_metadata and parent_metadata[row.parent_path] != metadata:
            raise ValueError(f"Inconsistent labels/source/group for parent {row.parent_path}")
        parent_metadata[row.parent_path] = metadata
        group_parents[row.group_id].add(row.parent_path)
    if any(len(parents) < 1 for parents in group_parents.values()):
        raise ValueError("Found an empty duplicate group")
    return rows


def extract_imagenet_features(
    rows: list[CVRow], device: torch.device
) -> np.ndarray:
    weights = MobileNet_V2_Weights.DEFAULT
    backbone = models.mobilenet_v2(weights=weights).features.to(device)
    backbone.eval()
    pool = nn.AdaptiveAvgPool2d((1, 1))
    loader = DataLoader(
        PatchDataset(rows),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )
    output: list[np.ndarray] = []
    with torch.inference_mode():
        for batch_index, images in enumerate(loader, start=1):
            embeddings = pool(backbone(images.to(device, non_blocking=True))).flatten(1)
            output.append(embeddings.cpu().numpy().astype(np.float32, copy=False))
            if batch_index % 20 == 0:
                LOGGER.info(
                    "feature_batches=%d/%d",
                    batch_index,
                    (len(rows) + BATCH_SIZE - 1) // BATCH_SIZE,
                )
    features = np.concatenate(output, axis=0)
    if features.shape[0] != len(rows):
        raise RuntimeError(f"Expected {len(rows)} features; got {features.shape[0]}")
    return features


def grouped_oof_predictions(
    rows: list[CVRow], features: np.ndarray
) -> tuple[np.ndarray, list[int]]:
    labels = np.asarray([row.label for row in rows], dtype=np.int64)
    groups = np.asarray([row.group_id for row in rows])
    splitter = StratifiedGroupKFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
    probabilities = np.full((len(rows), len(LABELS)), np.nan, dtype=np.float32)
    fold_ids = np.full(len(rows), -1, dtype=np.int8)

    for fold_number, (train_indices, heldout_indices) in enumerate(
        splitter.split(features, labels, groups), start=1
    ):
        train_groups = set(groups[train_indices])
        heldout_groups = set(groups[heldout_indices])
        if train_groups & heldout_groups:
            raise RuntimeError(f"Duplicate-group leakage in fold {fold_number}")
        train_parents = {rows[index].parent_path for index in train_indices}
        heldout_parents = {rows[index].parent_path for index in heldout_indices}
        if train_parents & heldout_parents:
            raise RuntimeError(f"Parent-image leakage in fold {fold_number}")

        classifier = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                class_weight="balanced",
                max_iter=1000,
                random_state=SEED,
            ),
        )
        classifier.fit(features[train_indices], labels[train_indices])
        fold_probabilities = classifier.predict_proba(features[heldout_indices])
        class_indices = classifier[-1].classes_.astype(np.int64)
        probabilities[np.ix_(heldout_indices, class_indices)] = fold_probabilities
        fold_ids[heldout_indices] = fold_number
        fold_accuracy = accuracy_score(
            labels[heldout_indices], fold_probabilities.argmax(axis=1)
        )
        LOGGER.info(
            "fold=%d train_patches=%d heldout_patches=%d heldout_parents=%d "
            "accuracy=%.4f",
            fold_number,
            len(train_indices),
            len(heldout_indices),
            len(heldout_parents),
            fold_accuracy,
        )

    if np.isnan(probabilities).any() or np.any(fold_ids < 1):
        raise RuntimeError("Cross-validation did not create exactly one OOF prediction per row")
    return probabilities, fold_ids.tolist()


def aggregate_parents(
    rows: list[CVRow], probabilities: np.ndarray
) -> list[dict[str, Any]]:
    indices_by_parent: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        indices_by_parent[row.parent_path].append(index)

    parent_records: list[dict[str, Any]] = []
    for parent_path, indices in sorted(indices_by_parent.items()):
        labels = {rows[index].label for index in indices}
        groups = {rows[index].group_id for index in indices}
        sources = {rows[index].source for index in indices}
        if len(labels) != 1 or len(groups) != 1 or len(sources) != 1:
            raise ValueError(f"Inconsistent parent metadata for {parent_path}")
        parent_probabilities = probabilities[indices].mean(axis=0)
        label = labels.pop()
        parent_records.append(
            {
                "parent_path": parent_path,
                "label": label,
                "source": sources.pop(),
                "group_id": groups.pop(),
                "patch_indices": indices,
                "probabilities": parent_probabilities,
                "predicted": int(parent_probabilities.argmax()),
                "self_confidence": float(parent_probabilities[label]),
            }
        )
    return parent_records


def thresholds_for(labels: np.ndarray, probabilities: np.ndarray) -> np.ndarray:
    thresholds = np.empty(len(LABELS), dtype=np.float64)
    for label in range(len(LABELS)):
        class_probabilities = probabilities[labels == label, label]
        if not len(class_probabilities):
            raise ValueError(f"No examples for class {LABELS[label]}")
        thresholds[label] = float(class_probabilities.mean())
    return thresholds


def mark_and_rank_suspects(
    rows: list[CVRow],
    probabilities: np.ndarray,
    parent_records: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    labels = np.asarray([row.label for row in rows], dtype=np.int64)
    patch_thresholds = thresholds_for(labels, probabilities)
    parent_labels = np.asarray([parent["label"] for parent in parent_records])
    parent_probabilities = np.stack(
        [parent["probabilities"] for parent in parent_records], axis=0
    )
    parent_thresholds = thresholds_for(parent_labels, parent_probabilities)

    patch_records: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        predicted = int(probabilities[index].argmax())
        self_confidence = float(probabilities[index, row.label])
        alternate_confidence = float(probabilities[index, predicted])
        confident_mismatch = (
            predicted != row.label
            and alternate_confidence >= patch_thresholds[predicted]
            and self_confidence < patch_thresholds[row.label]
        )
        patch_records.append(
            {
                "row_index": index,
                "label": row.label,
                "predicted": predicted,
                "probabilities": probabilities[index],
                "self_confidence": self_confidence,
                "alternate_confidence": float(probabilities[index, 1 - row.label]),
                "confident_mismatch": bool(confident_mismatch),
                "score": float(1.0 - self_confidence),
            }
        )

    parent_by_path = {parent["parent_path"]: parent for parent in parent_records}
    for parent in parent_records:
        label = int(parent["label"])
        predicted = int(parent["predicted"])
        confidence = float(parent["probabilities"][predicted])
        parent["confident_mismatch"] = bool(
            predicted != label
            and confidence >= parent_thresholds[predicted]
            and parent["self_confidence"] < parent_thresholds[label]
        )
        parent["score"] = float(1.0 - parent["self_confidence"])

    ranked_parents = sorted(
        parent_records,
        key=lambda parent: (
            not parent["confident_mismatch"],
            -parent["score"],
            parent["parent_path"],
        ),
    )
    for parent_rank, parent in enumerate(ranked_parents, start=1):
        parent["parent_rank"] = parent_rank

    # Select one highly suspicious patch per parent for the 150-image review grid.
    ranked_patch_records: list[dict[str, Any]] = []
    best_patch_by_parent: dict[str, dict[str, Any]] = {}
    for patch in patch_records:
        row = rows[patch["row_index"]]
        parent = parent_by_path[row.parent_path]
        patch["parent_rank"] = parent["parent_rank"]
        patch["parent_score"] = parent["score"]
        patch["parent_confident_mismatch"] = parent["confident_mismatch"]
        patch["parent_probabilities"] = parent["probabilities"]
        patch["grid_candidate"] = parent["parent_rank"] <= 150
        existing = best_patch_by_parent.get(row.parent_path)
        if existing is None or (
            patch["confident_mismatch"],
            patch["score"],
        ) > (
            existing["confident_mismatch"],
            existing["score"],
        ):
            best_patch_by_parent[row.parent_path] = patch

    ranked_patch_records = sorted(
        patch_records,
        key=lambda patch: (
            not patch["confident_mismatch"],
            -patch["score"],
            -patch["parent_score"],
            patch["parent_rank"],
            rows[patch["row_index"]].relative_patch_path,
        ),
    )
    for patch_rank, patch in enumerate(ranked_patch_records, start=1):
        patch["patch_rank"] = patch_rank

    grid_candidates = [
        best_patch_by_parent[parent["parent_path"]] for parent in ranked_parents[:150]
    ]
    grid_candidates.sort(key=lambda patch: patch["parent_rank"])

    def confident_joint(
        example_labels: np.ndarray,
        example_probabilities: np.ndarray,
        thresholds: np.ndarray,
    ) -> list[list[int]]:
        joint = np.zeros((len(LABELS), len(LABELS)), dtype=np.int64)
        predicted = example_probabilities.argmax(axis=1)
        for index, predicted_label in enumerate(predicted):
            if example_probabilities[index, predicted_label] >= thresholds[predicted_label]:
                joint[example_labels[index], predicted_label] += 1
        return joint.tolist()

    summary = {
        "method": (
            "Confident-learning-style OOF ranking: class thresholds are mean "
            "out-of-fold probability assigned to each observed class; the "
            "confident joint counts OOF predictions exceeding their predicted-class threshold."
        ),
        "patch_class_thresholds": {
            label: float(patch_thresholds[index])
            for index, label in enumerate(LABELS)
        },
        "parent_class_thresholds": {
            label: float(parent_thresholds[index])
            for index, label in enumerate(LABELS)
        },
        "patch_confident_joint_observed_by_predicted": confident_joint(
            labels, probabilities, patch_thresholds
        ),
        "parent_confident_joint_observed_by_predicted": confident_joint(
            parent_labels, parent_probabilities, parent_thresholds
        ),
        "patch_confident_mismatch_count": int(
            sum(record["confident_mismatch"] for record in patch_records)
        ),
        "parent_confident_mismatch_count": int(
            sum(parent["confident_mismatch"] for parent in parent_records)
        ),
        "review_grid_parents": len(grid_candidates),
    }
    return ranked_patch_records, grid_candidates, summary


def cv_metrics(
    rows: list[CVRow], probabilities: np.ndarray, fold_ids: list[int]
) -> dict[str, Any]:
    labels = np.asarray([row.label for row in rows], dtype=np.int64)
    predictions = probabilities.argmax(axis=1)
    parents = aggregate_parents(rows, probabilities)
    parent_labels = np.asarray([parent["label"] for parent in parents])
    parent_predictions = np.asarray([parent["predicted"] for parent in parents])
    per_fold = {}
    for fold in sorted(set(fold_ids)):
        indices = np.flatnonzero(np.asarray(fold_ids) == fold)
        per_fold[str(fold)] = {
            "patches": int(len(indices)),
            "parents": len({rows[index].parent_path for index in indices}),
            "patch_accuracy": float(accuracy_score(labels[indices], predictions[indices])),
        }
    return {
        "patches": len(rows),
        "parents": len(parents),
        "patch_accuracy": float(accuracy_score(labels, predictions)),
        "patch_macro_f1": float(f1_score(labels, predictions, average="macro")),
        "patch_confusion_matrix": confusion_matrix(
            labels, predictions, labels=list(range(len(LABELS)))
        ).tolist(),
        "parent_accuracy": float(accuracy_score(parent_labels, parent_predictions)),
        "parent_macro_f1": float(
            f1_score(parent_labels, parent_predictions, average="macro")
        ),
        "parent_confusion_matrix": confusion_matrix(
            parent_labels, parent_predictions, labels=list(range(len(LABELS)))
        ).tolist(),
        "folds": per_fold,
    }


def save_review_grid(
    rows: list[CVRow], candidates: list[dict[str, Any]], path: Path
) -> None:
    columns = 10
    tile_width = 190
    tile_height = 190
    rows_count = (len(candidates) + columns - 1) // columns
    figure, axes = plt.subplots(
        rows_count,
        columns,
        figsize=(columns * 2.0, rows_count * 2.05),
        squeeze=False,
    )
    for axis, candidate in zip(axes.flat, candidates):
        row = rows[candidate["row_index"]]
        with Image.open(row.patch_path) as opened:
            image = ImageOps.exif_transpose(opened).convert("RGB")
            image.thumbnail((tile_width, tile_height))
            axis.imshow(image)
        truth = LABELS[row.label]
        predicted = LABELS[candidate["predicted"]]
        axis.set_title(
            f'P{candidate["parent_rank"]} {truth}→{predicted} '
            f'(p(label)={candidate["self_confidence"]:.2f})\n'
            f'{Path(row.parent_path).name[:23]}',
            fontsize=7,
        )
        axis.axis("off")
    for axis in list(axes.flat)[len(candidates) :]:
        axis.axis("off")
    figure.suptitle(
        "Top 150 suspected parent images (OOF; observed labels shown)", fontsize=14
    )
    figure.tight_layout(rect=(0, 0, 1, 0.99))
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=90)
    plt.close(figure)


def write_suspect_csv(
    rows: list[CVRow],
    ranked_patches: list[dict[str, Any]],
    fold_ids: list[int],
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "patch_rank",
        "parent_rank",
        "in_top_150_parents",
        "patch_path",
        "parent_path",
        "group_id",
        "source",
        "observed_label",
        "oof_predicted_label",
        "oof_probability_observed_label",
        "oof_probability_alternate_label",
        "oof_probability_clean",
        "oof_probability_dusty",
        "parent_oof_probability_clean",
        "parent_oof_probability_dusty",
        "patch_suspect_score",
        "patch_confident_mismatch",
        "parent_suspect_score",
        "parent_confident_mismatch",
        "fold",
    ]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for patch in ranked_patches:
            row = rows[patch["row_index"]]
            writer.writerow(
                {
                    "patch_rank": patch["patch_rank"],
                    "parent_rank": patch["parent_rank"],
                    "in_top_150_parents": patch["grid_candidate"],
                    "patch_path": row.relative_patch_path,
                    "parent_path": row.parent_path,
                    "group_id": row.group_id,
                    "source": row.source,
                    "observed_label": LABELS[row.label],
                    "oof_predicted_label": LABELS[patch["predicted"]],
                    "oof_probability_observed_label": f'{patch["self_confidence"]:.8f}',
                    "oof_probability_alternate_label": f'{patch["alternate_confidence"]:.8f}',
                    "oof_probability_clean": f'{patch["probabilities"][0]:.8f}',
                    "oof_probability_dusty": f'{patch["probabilities"][1]:.8f}',
                    "parent_oof_probability_clean": f'{patch["parent_probabilities"][0]:.8f}',
                    "parent_oof_probability_dusty": f'{patch["parent_probabilities"][1]:.8f}',
                    "patch_suspect_score": f'{patch["score"]:.8f}',
                    "patch_confident_mismatch": patch["confident_mismatch"],
                    "parent_suspect_score": f'{patch["parent_score"]:.8f}',
                    "parent_confident_mismatch": patch["parent_confident_mismatch"],
                    "fold": fold_ids[patch["row_index"]],
                }
            )


def write_cv_report(
    metrics: dict[str, Any],
    suspect_summary: dict[str, Any],
    output_path: Path,
    elapsed_seconds: float,
) -> None:
    payload = {
        "data_scope": "training/splits/main/{train,val}.csv only; test split not read",
        "model": (
            "Frozen ImageNet MobileNetV2 pooled features followed by a fold-specific "
            "StandardScaler + class-balanced LogisticRegression linear probe"
        ),
        "grouping": "StratifiedGroupKFold by duplicate group_id; parent paths asserted disjoint",
        "seed": SEED,
        "fold_count": FOLDS,
        "runtime_seconds": elapsed_seconds,
        "oof_metrics": metrics,
        "confident_learning_review": suspect_summary,
        "automatic_relabeling": False,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    markdown_path = output_path.with_suffix(".md")
    lines = [
        "# Grouped train+validation OOF label review",
        "",
        "- Data scope: train and validation split CSVs only; test split was not opened.",
        "- Validation design: five stratified folds grouped by duplicate group ID; parent images are asserted not to cross folds.",
        "- Model: frozen ImageNet MobileNetV2 pooled embeddings and fold-specific class-balanced logistic regression.",
        "- This is a fixed feature-probe ranking for human review, not an automatic relabeling decision.",
        f"- Runtime: {elapsed_seconds:.1f} seconds.",
        "",
        "## Out-of-fold metrics",
        "",
        f"- Patches: {metrics['patches']}; patch accuracy {metrics['patch_accuracy']:.4f}; macro F1 {metrics['patch_macro_f1']:.4f}.",
        f"- Parents: {metrics['parents']}; parent accuracy {metrics['parent_accuracy']:.4f}; macro F1 {metrics['parent_macro_f1']:.4f}.",
        "",
        "### Patch confusion matrix (observed rows, predicted columns)",
        "",
        f"`{metrics['patch_confusion_matrix']}`",
        "",
        "### Parent confusion matrix (observed rows, predicted columns)",
        "",
        f"`{metrics['parent_confusion_matrix']}`",
        "",
        "## Fold metrics",
        "",
        "| Fold | Patches | Parents | Patch accuracy |",
        "|---:|---:|---:|---:|",
    ]
    for fold, result in metrics["folds"].items():
        lines.append(
            f"| {fold} | {result['patches']} | {result['parents']} | "
            f"{result['patch_accuracy']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Confident-learning review ranking",
            "",
            suspect_summary["method"],
            "",
            f"- Patch-level confident mismatches: {suspect_summary['patch_confident_mismatch_count']}.",
            f"- Parent-level confident mismatches: {suspect_summary['parent_confident_mismatch_count']}.",
            f"- Top review grid: {suspect_summary['review_grid_parents']} unique parents.",
            "- Ranking scores are OOF model disagreement signals and may reflect domain shift, "
            "ambiguous dust levels, or crop selection rather than incorrect labels. Review visually; "
            "no labels or source data were changed.",
            "",
            "## Class thresholds",
            "",
            f"- Patch: `{json.dumps(suspect_summary['patch_class_thresholds'], sort_keys=True)}`",
            f"- Parent: `{json.dumps(suspect_summary['parent_class_thresholds'], sort_keys=True)}`",
            "",
            "The complete patch-level ranking and parent ranks are in `suspect_labels.csv`.",
            "",
        ]
    )
    markdown_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(min(8, torch.get_num_threads()))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    LOGGER.info(
        "device=%s torch=%s torchvision=%s",
        device,
        torch.__version__,
        __import__("torchvision").__version__,
    )
    rows = load_train_val_rows()
    LOGGER.info(
        "loaded train+val only: patches=%d parents=%d duplicate_groups=%d",
        len(rows),
        len({row.parent_path for row in rows}),
        len({row.group_id for row in rows}),
    )
    started = time.perf_counter()
    features = extract_imagenet_features(rows, device)
    probabilities, fold_ids = grouped_oof_predictions(rows, features)
    parents = aggregate_parents(rows, probabilities)
    ranked_patches, grid_candidates, suspect_summary = mark_and_rank_suspects(
        rows, probabilities, parents
    )
    metrics = cv_metrics(rows, probabilities, fold_ids)
    elapsed_seconds = time.perf_counter() - started

    save_review_grid(
        rows,
        grid_candidates,
        REPORT_ROOT / "suspect_labels_top150.png",
    )
    write_suspect_csv(
        rows,
        ranked_patches,
        fold_ids,
        REPORT_ROOT / "suspect_labels.csv",
    )
    write_cv_report(
        metrics,
        suspect_summary,
        REPORT_ROOT / "suspect_labels_cv.json",
        elapsed_seconds,
    )
    LOGGER.info("OOF metrics: %s", json.dumps(metrics, sort_keys=True))
    LOGGER.info(
        "Confident-learning review: %s",
        json.dumps(suspect_summary, sort_keys=True),
    )
    LOGGER.info("CV runtime seconds=%.1f", elapsed_seconds)
    LOGGER.info("Review artifacts: %s", REPORT_ROOT)


if __name__ == "__main__":
    main()

"""Evaluate the frozen main-model test split at patch and parent-image levels."""

from __future__ import annotations

import csv
import importlib.metadata
import importlib.metadata
import json
import logging
import platform
import platform
import random
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageFilter, ImageOps, ImageDraw
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from torchvision.transforms import InterpolationMode


ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
TEST_PATH = ROOT / "splits" / "main" / "test.csv"
CHECKPOINT_PATH = ROOT / "checkpoints" / "main_best.pt"
DATA_MANIFEST_PATH = ROOT / "data" / "manifest.csv"
REPORT_ROOT = ROOT / "reports"
LOG_ROOT = ROOT / "logs"
MODEL_ROOT = PROJECT_ROOT / "model"
LABELS = ("clean", "dusty")
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
INPUT_SIZE = 224
BOOTSTRAP_ITERATIONS = 2000
SEED = 42
LOGGER = logging.getLogger("evaluate")


@dataclass(frozen=True)
class TestRow:
    image_path: Path
    parent_path: str
    group_id: str
    source: str
    true_label: int


class EvaluationDataset(Dataset[torch.Tensor]):
    def __init__(
        self,
        rows: list[TestRow],
        transform: transforms.Compose,
        brightness: float = 1.0,
        blur_radius: float = 0.0,
    ) -> None:
        self.rows = rows
        self.transform = transform
        self.brightness = brightness
        self.blur_radius = blur_radius

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> torch.Tensor:
        with Image.open(self.rows[index].image_path) as opened:
            image = ImageOps.exif_transpose(opened).convert("RGB")
        if self.brightness != 1.0:
            image = ImageEnhance.Brightness(image).enhance(self.brightness)
        if self.blur_radius > 0:
            image = image.filter(ImageFilter.GaussianBlur(radius=self.blur_radius))
        return self.transform(image)


def build_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize(256, interpolation=InterpolationMode.BILINEAR),
            transforms.CenterCrop(INPUT_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ]
    )


def make_model() -> nn.Module:
    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(LABELS))
    return model


def load_test_rows() -> list[TestRow]:
    if not TEST_PATH.is_file():
        raise FileNotFoundError(f"Test split not found: {TEST_PATH}")
    rows: list[TestRow] = []
    with TEST_PATH.open(newline="", encoding="utf-8") as stream:
        for item in csv.DictReader(stream):
            image_path = PROJECT_ROOT / "training" / item["patch_path"]
            if not image_path.is_file():
                raise FileNotFoundError(f"Test image is missing: {image_path}")
            rows.append(
                TestRow(
                    image_path=image_path,
                    parent_path=item["parent_path"],
                    group_id=item["group_id"],
                    source=item["source"],
                    true_label=LABELS.index(item["label"]),
                )
            )
    return rows


def infer(
    model: nn.Module,
    rows: list[TestRow],
    device: torch.device,
    batch_size: int = 64,
    brightness: float = 1.0,
    blur_radius: float = 0.0,
) -> np.ndarray:
    dataset = EvaluationDataset(
        rows,
        build_transform(),
        brightness=brightness,
        blur_radius=blur_radius,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )
    model.eval()
    probabilities: list[np.ndarray] = []
    with torch.inference_mode():
        for images in loader:
            logits = model(images.to(device, non_blocking=True))
            probabilities.append(torch.softmax(logits, dim=1).cpu().numpy())
    if not probabilities:
        raise ValueError("Cannot evaluate an empty dataset")
    return np.concatenate(probabilities, axis=0)


def metrics_for(
    true_labels: np.ndarray,
    predicted_labels: np.ndarray,
    confidences: np.ndarray | None = None,
) -> dict[str, Any]:
    precision, recall, f1, support = precision_recall_fscore_support(
        true_labels,
        predicted_labels,
        labels=list(range(len(LABELS))),
        zero_division=0,
    )
    result: dict[str, Any] = {
        "samples": int(len(true_labels)),
        "accuracy": float(accuracy_score(true_labels, predicted_labels)),
        "per_class": {
            label: {
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "support": int(support[index]),
            }
            for index, label in enumerate(LABELS)
        },
        "confusion_matrix": confusion_matrix(
            true_labels, predicted_labels, labels=list(range(len(LABELS)))
        ).tolist(),
    }
    if confidences is not None:
        correct = predicted_labels == true_labels
        result["average_confidence_correct"] = (
            float(confidences[correct].mean()) if np.any(correct) else None
        )
        result["average_confidence_wrong"] = (
            float(confidences[~correct].mean()) if np.any(~correct) else None
        )
    return result


def parent_aggregate(
    rows: list[TestRow], probabilities: np.ndarray
) -> tuple[list[dict[str, Any]], np.ndarray, np.ndarray]:
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        grouped[row.parent_path].append(index)
    parents: list[dict[str, Any]] = []
    for parent_path, indices in sorted(grouped.items()):
        labels = {rows[index].true_label for index in indices}
        sources = {rows[index].source for index in indices}
        groups = {rows[index].group_id for index in indices}
        if len(labels) != 1 or len(sources) != 1 or len(groups) != 1:
            raise ValueError(f"Inconsistent test parent metadata for {parent_path}")
        average_probability = probabilities[indices].mean(axis=0)
        parents.append(
            {
                "parent_path": parent_path,
                "source": rows[indices[0]].source,
                "group_id": rows[indices[0]].group_id,
                "true": labels.pop(),
                "probabilities": average_probability,
                "predicted": int(np.argmax(average_probability)),
                "confidence": float(np.max(average_probability)),
                "patch_count": len(indices),
            }
        )
    true = np.asarray([parent["true"] for parent in parents], dtype=np.int64)
    predicted = np.asarray([parent["predicted"] for parent in parents], dtype=np.int64)
    confidence = np.asarray([parent["confidence"] for parent in parents], dtype=np.float64)
    return parents, true, predicted


def bootstrap_intervals(
    parents: list[dict[str, Any]], iterations: int = BOOTSTRAP_ITERATIONS
) -> dict[str, Any]:
    rng = np.random.default_rng(SEED)
    labels = np.asarray([parent["true"] for parent in parents], dtype=np.int64)
    predictions = np.asarray([parent["predicted"] for parent in parents], dtype=np.int64)
    n = len(parents)
    samples: dict[str, list[float]] = {
        "accuracy": [],
        **{
            f"{label}_{metric}": []
            for label in LABELS
            for metric in ("precision", "recall", "f1")
        },
    }
    for _ in range(iterations):
        indices = rng.integers(0, n, size=n)
        y_true = labels[indices]
        y_pred = predictions[indices]
        samples["accuracy"].append(float(np.mean(y_true == y_pred)))
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true,
            y_pred,
            labels=list(range(len(LABELS))),
            zero_division=0,
        )
        for index, label in enumerate(LABELS):
            samples[f"{label}_precision"].append(float(precision[index]))
            samples[f"{label}_recall"].append(float(recall[index]))
            samples[f"{label}_f1"].append(float(f1[index]))
    return {
        key: [
            float(np.percentile(values, 2.5)),
            float(np.percentile(values, 97.5)),
        ]
        for key, values in samples.items()
    }


def split_by_source(
    rows: list[TestRow], probabilities: np.ndarray
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for source in sorted({row.source for row in rows}):
        indices = [index for index, row in enumerate(rows) if row.source == source]
        y_true = np.asarray([rows[index].true_label for index in indices], dtype=np.int64)
        y_pred = np.argmax(probabilities[indices], axis=1)
        confidence = np.max(probabilities[indices], axis=1)
        parent_subset = [row for row in rows if row.source == source]
        parent_probs = probabilities[indices]
        parents, parent_true, parent_pred = parent_aggregate(parent_subset, parent_probs)
        result[source] = {
            "patch": metrics_for(y_true, y_pred, confidence),
            "parent": metrics_for(
                parent_true,
                parent_pred,
                np.asarray([parent["confidence"] for parent in parents]),
            ),
        }
    return result


def evaluate_heldout(
    model: nn.Module, device: torch.device
) -> dict[str, dict[str, Any]]:
    heldout: dict[str, list[tuple[str, Path]]] = defaultdict(list)
    with DATA_MANIFEST_PATH.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if row["role"] != "heldout":
                continue
            if row["label"] not in {
                "bird-drop",
                "snow-covered",
                "electrical-damage",
            }:
                continue
            image_path = ROOT / row["processed_path"]
            if not image_path.is_file():
                raise FileNotFoundError(f"Held-out image is missing: {image_path}")
            heldout[row["label"]].append((row["source"], image_path))
    results: dict[str, dict[str, Any]] = {}
    results: dict[str, dict[str, Any]] = {}
    for label, items in sorted(heldout.items()):
        dataset_rows = [
            TestRow(
                image_path=path,
                parent_path=str(path),
                group_id=str(path),
                source=source,
                true_label=0,
            )
            for source, path in items
        ]
        probabilities = infer(model, dataset_rows, device)
        prediction_counts = Counter(
            LABELS[index] for index in np.argmax(probabilities, axis=1)
        )
        results[label] = {
            "images": len(items),
            "predicted_labels": dict(prediction_counts),
            "average_probability": {
                LABELS[index]: float(probabilities[:, index].mean())
                for index in range(len(LABELS))
            },
        }
    return results


def robustness_metrics(
    model: nn.Module,
    rows: list[TestRow],
    baseline_patch_accuracy: float,
    baseline_parent_accuracy: float,
    device: torch.device,
) -> dict[str, Any]:
    cases = {
        "brightness_minus_30_percent": {"brightness": 0.7, "blur_radius": 0.0},
        "brightness_plus_30_percent": {"brightness": 1.3, "blur_radius": 0.0},
        "gaussian_blur_radius_1_5": {"brightness": 1.0, "blur_radius": 1.5},
    }
    results: dict[str, Any] = {}
    for name, options in cases.items():
        probabilities = infer(model, rows, device, **options)
        patch_prediction = probabilities.argmax(axis=1)
        patch_true = np.asarray([row.true_label for row in rows], dtype=np.int64)
        patch_metrics = metrics_for(
            patch_true,
            patch_prediction,
            probabilities.max(axis=1),
        )
        parent_results, parent_true, parent_prediction = parent_aggregate(rows, probabilities)
        robust_parent_metrics = metrics_for(
            parent_true,
            parent_prediction,
            np.asarray([parent["confidence"] for parent in parent_results]),
        )
        results[name] = {
            "patch": {
                "accuracy": patch_metrics["accuracy"],
                "drop_from_baseline": baseline_patch_accuracy - patch_metrics["accuracy"],
            },
            "parent": {
                "accuracy": robust_parent_metrics["accuracy"],
                "drop_from_baseline": baseline_parent_accuracy - robust_parent_metrics["accuracy"],
            },
        }
    results["baseline"] = {
        "patch_accuracy": baseline_patch_accuracy,
        "parent_accuracy": baseline_parent_accuracy,
    }
    return results


def save_confusion_matrix(metrics: dict[str, Any], path: Path) -> None:
    matrix = np.asarray(metrics["confusion_matrix"])
    figure, axis = plt.subplots(figsize=(6, 5))
    image = axis.imshow(matrix, cmap="Blues")
    axis.set(
        xticks=range(len(LABELS)),
        yticks=range(len(LABELS)),
        xticklabels=LABELS,
        yticklabels=LABELS,
        xlabel="Predicted label",
        ylabel="True label",
        title="Main model test confusion matrix (parent images)",
    )
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            axis.text(column, row, str(matrix[row, column]), ha="center", va="center")
    figure.colorbar(image, ax=axis)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=160)
    plt.close(figure)


def save_wrong_images(
    parents: list[dict[str, Any]], path: Path, csv_path: Path
) -> dict[str, int]:
    wrong_clean_to_dusty = [
        parent
        for parent in parents
        if parent["true"] == LABELS.index("clean")
        and parent["predicted"] == LABELS.index("dusty")
    ]
    wrong_dusty_to_clean = [
        parent
        for parent in parents
        if parent["true"] == LABELS.index("dusty")
        and parent["predicted"] == LABELS.index("clean")
    ]
    selected = wrong_clean_to_dusty[:6] + wrong_dusty_to_clean[:6]
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=(
                "parent_image",
                "true_label",
                "predicted_label",
                "confidence",
                "patch_count",
            ),
        )
        writer.writeheader()
        for parent in selected:
            writer.writerow(
                {
                    "parent_image": parent["parent_path"],
                    "true_label": LABELS[parent["true"]],
                    "predicted_label": LABELS[parent["predicted"]],
                    "confidence": f'{parent["confidence"]:.6f}',
                    "patch_count": parent["patch_count"],
                }
            )
    figure, axes = plt.subplots(3, 4, figsize=(16, 11))
    for axis, parent in zip(axes.flat, selected):
        image_path = PROJECT_ROOT / "training" / parent["parent_path"]
        with Image.open(image_path) as opened:
            image = ImageOps.exif_transpose(opened).convert("RGB")
            image.thumbnail((360, 270))
            axis.imshow(image)
        axis.set_title(
            f'{LABELS[parent["true"]]} → {LABELS[parent["predicted"]]} '
            f'({parent["confidence"]:.2f})\n{Path(parent["parent_path"]).name}',
            fontsize=9,
        )
        axis.axis("off")
    for axis in list(axes.flat)[len(selected) :]:
        axis.axis("off")
    figure.suptitle("Misclassified main-model test parent images", fontsize=15)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return {
        "clean_predicted_dusty": len(wrong_clean_to_dusty),
        "dusty_predicted_clean": len(wrong_dusty_to_clean),
        "shown": len(selected),
        "examples": [
            {
                "parent_image": parent["parent_path"],
                "true_label": LABELS[parent["true"]],
                "predicted_label": LABELS[parent["predicted"]],
                "confidence": parent["confidence"],
                "patch_count": parent["patch_count"],
            }
            for parent in selected
        ],
    }


def write_metrics_report(
    output: dict[str, Any],
    json_path: Path,
    markdown_path: Path,
    model_card_path: Path,
    training_seconds: float,
) -> None:
    json_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    lines = [
        "# Main model evaluation",
        "",
        f"- Test patches: {output['patch']['samples']}",
        f"- Test parent images: {output['parent']['samples']}",
        f"- Measured training time: {training_seconds:.1f} seconds",
        "- Parent-image results are primary; patch results are correlated observations.",
        "- 95% intervals are percentile bootstrap intervals resampling parent images.",
        "",
        "## Parent-image metrics (primary)",
        "",
        "| Class | Precision | Recall | F1 | Parents | Precision 95% CI | Recall 95% CI | F1 95% CI |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    parent_intervals = output["parent_bootstrap_95"]
    for label in LABELS:
        item = output["parent"]["per_class"][label]
        precision_lower, precision_upper = parent_intervals[f"{label}_precision"]
        recall_lower, recall_upper = parent_intervals[f"{label}_recall"]
        lower, upper = parent_intervals[f"{label}_f1"]
        lines.append(
            f"| {label} | {item['precision']:.4f} | {item['recall']:.4f} | "
            f"{item['f1']:.4f} | {item['support']} | "
            f"[{precision_lower:.4f}, {precision_upper:.4f}] | "
            f"[{recall_lower:.4f}, {recall_upper:.4f}] | "
            f"[{lower:.4f}, {upper:.4f}] |"
        )
    lower, upper = parent_intervals["accuracy"]
    lines.extend(
        [
            "",
            f"Parent-image accuracy: {output['parent']['accuracy']:.4f} "
            f"(95% bootstrap CI [{lower:.4f}, {upper:.4f}]).",
            "",
            "## Patch-level metrics",
            "",
            "| Class | Precision | Recall | F1 | Support (patches) |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for label in LABELS:
        item = output["patch"]["per_class"][label]
        lines.append(
            f"| {label} | {item['precision']:.4f} | {item['recall']:.4f} | "
            f"{item['f1']:.4f} | {item['support']} |"
        )
    lines.extend(
        [
            "",
            "## Confidence summary",
            "",
            "Mean maximum class probability for correct versus incorrect predictions:",
            "",
            f"- Parent images: {output['parent']['average_confidence_correct']:.4f} correct; "
            f"{output['parent']['average_confidence_wrong']:.4f} incorrect.",
            f"- Patches: {output['patch']['average_confidence_correct']:.4f} correct; "
            f"{output['patch']['average_confidence_wrong']:.4f} incorrect.",
            "- The planned 0.80 controller threshold is not calibrated by this public test; "
            "select a threshold using own-panel validation data and account for the cost "
            "of false dusty decisions.",
        ]
    )
    lines.extend(["", "## Per-source metrics", ""])
    for source, metrics in sorted(output["per_source"].items()):
        lines.append(f"### {source}")
        lines.append("")
        lines.append("| Class | Precision | Recall | F1 | Parents |")
        lines.append("|---|---:|---:|---:|---:|")
        for label in LABELS:
            item = metrics["parent"]["per_class"][label]
            lines.append(
                f"| {label} | {item['precision']:.4f} | {item['recall']:.4f} | "
                f"{item['f1']:.4f} | {item['support']} |"
            )
        lines.extend(["", "Patch level:", ""])
        lines.append("| Class | Precision | Recall | F1 | Patches |")
        lines.append("|---|---:|---:|---:|---:|")
        for label in LABELS:
            item = metrics["patch"]["per_class"][label]
            lines.append(
                f"| {label} | {item['precision']:.4f} | {item['recall']:.4f} | "
                f"{item['f1']:.4f} | {item['support']} |"
            )
        lines.append("")
        if source == "pv_defect":
            pv_dusty = metrics["patch"]["per_class"]["dusty"]["support"]
            lines.append(
                f"Warning: PV dusty test metrics are based on only {pv_dusty} patches."
            )
            lines.append("")
    lines.extend(
        [
            "## Robustness",
            "",
            "Drop is baseline accuracy minus condition accuracy; negative values mean "
            "accuracy increased in that perturbation.",
            "",
            "| Condition | Parent accuracy | Parent drop | Patch accuracy | Patch drop |",
            "|---|---:|---:|---:|---:|",
            f"| Baseline | {output['robustness']['baseline']['parent_accuracy']:.4f} | "
            f"0.0000 | {output['robustness']['baseline']['patch_accuracy']:.4f} | 0.0000 |",
        ]
    )
    for name, values in output["robustness"].items():
        if name == "baseline":
            continue
        lines.append(
            f"| {name} | {values['parent']['accuracy']:.4f} | "
            f"{values['parent']['drop_from_baseline']:.4f} | "
            f"{values['patch']['accuracy']:.4f} | "
            f"{values['patch']['drop_from_baseline']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Held-out classes (informative only; not trained)",
            "",
            "| Class | Images | Predicted labels |",
            "|---|---:|---|",
        ]
    )
    for label, result in sorted(output["heldout"].items()):
        lines.append(
            f"| {label} | {result['images']} | "
            f"{json.dumps(result['predicted_labels'], sort_keys=True)} |"
        )
    lines.extend(
        [
            "",
            "## Test errors shown",
            "",
            f"Selected {output['wrong_parent_counts']['shown']} parent images "
            "(up to six clean→dusty and six dusty→clean). The full list is in "
            "`main_test_errors_12.csv`; the grid is `main_test_errors_12.png`.",
            "",
            "## Limitations",
            "",
            "- Public images are mostly broad panel/array views and do not match the fixed close-up webcam deployment.",
            "- The PV dusty test subset contains only six patches; its source-specific result is highly uncertain.",
            "- Cracked was not trained in the main classifier. Its 70-source-image experiment was skipped after visual review.",
            "- The main metrics are not a substitute for testing on the user's own panel; own photos are pending.",
            "- Bootstrap intervals resample parent images and do not account for dataset source or collection bias.",
            "",
        ]
    )
    markdown_path.write_text("\n".join(lines), encoding="utf-8")
    write_model_card(output, model_card_path, training_seconds)


def split_sizes() -> dict[str, dict[str, int]]:
    sizes: dict[str, dict[str, int]] = {}
    for split_name in ("train", "val", "test"):
        split_path = ROOT / "splits" / "main" / f"{split_name}.csv"
        with split_path.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        sizes[split_name] = {
            "patches": len(rows),
            "parents": len({row["parent_path"] for row in rows}),
        }
    return sizes


def write_model_card(
    output: dict[str, Any], path: Path, training_seconds: float
) -> None:
    versions = {
        display_name: importlib.metadata.version(package_name)
        for display_name, package_name in {
            "torch": "torch",
            "torchvision": "torchvision",
            "numpy": "numpy",
            "scikit-learn": "scikit-learn",
            "Pillow": "Pillow",
            "opencv-python": "opencv-python",
            "matplotlib": "matplotlib",
            "pandas": "pandas",
            "ImageHash": "ImageHash",
            "tqdm": "tqdm",
        }.items()
    }
    lower, upper = output["parent_bootstrap_95"]["accuracy"]
    lines = [
        "# Solar-panel clean/dusty model card",
        "",
        f"- Generated: {time.strftime('%Y-%m-%d')}",
        "- Task: two-class image classification (`clean`, `dusty`).",
        "- Architecture: MobileNetV2 with ImageNet initialization.",
        f"- Input: {INPUT_SIZE}×{INPUT_SIZE} RGB; ImageNet mean/std normalization.",
        f"- Training time: {training_seconds:.1f} seconds "
        f"({training_seconds / 60:.2f} minutes; full fit, smoke test excluded).",
        f"- Split sizes: {json.dumps(output['split_sizes'], sort_keys=True)}",
        "- Public sources and reported licenses: "
        "[solar-panel-dust-detection](https://www.kaggle.com/datasets/hemanthsai7/solar-panel-dust-detection) "
        "(CC0-1.0); "
        "[pv-panel-defect-dataset](https://www.kaggle.com/datasets/alicjalena/pv-panel-defect-dataset) "
        "(CC-BY-NC-SA-4.0). The latter is non-commercial/share-alike; check its "
        "terms before deployment.",
        "- Cracked training was skipped after the user rejected its patch grid; "
        "cracked is not in this model's labels.",
        "",
        "## Public test results",
        "",
        f"- Parent-image accuracy: {output['parent']['accuracy']:.4f}; 95% parent-bootstrap "
        f"CI [{lower:.4f}, {upper:.4f}].",
        f"- Patch accuracy: {output['patch']['accuracy']:.4f}.",
        "- Parent-image results are primary. See "
        "[the complete evaluation](../training/reports/evaluation.md) for per-class, "
        "per-source, robustness, held-out, and confidence results.",
        "- Own-panel metrics: pending; no own webcam photos were available.",
        "",
        "## Runtime versions",
        "",
        f"- Python: {platform.python_version()}",
    ]
    lines.extend(f"- {name}: {version}" for name, version in sorted(versions.items()))
    lines.extend(
        [
            "",
            "## Parent-image test metrics",
            "",
            "| Class | Precision | Recall | F1 | Support |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for label in LABELS:
        item = output["parent"]["per_class"][label]
        lines.append(
            f"| {label} | {item['precision']:.4f} | {item['recall']:.4f} | "
            f"{item['f1']:.4f} | {item['support']} |"
        )
    lines.extend(
        [
            "",
            "## Per-source metrics",
            "",
            "| Source | Level | Class | Precision | Recall | F1 | Support |",
            "|---|---|---|---:|---:|---:|---:|",
        ]
    )
    for source, source_metrics in sorted(output["per_source"].items()):
        for level in ("parent", "patch"):
            for label in LABELS:
                item = source_metrics[level]["per_class"][label]
                lines.append(
                    f"| {source} | {level} | {label} | {item['precision']:.4f} | "
                    f"{item['recall']:.4f} | {item['f1']:.4f} | {item['support']} |"
                )
    lines.extend(
        [
            "",
            "Own-photo metrics: pending. No webcam photos were available; public "
            "metrics are not a deployment substitute.",
        ]
    )
    lines.extend(
        [
            "",
            "## Known limits",
            "",
            "- Public images are predominantly broad panel/array views and may differ "
            "substantially from one close-up fixed webcam view.",
            "- The PV dusty test result is based on only six patches and is not stable.",
            "- Public test results do not establish deployment accuracy. Collect "
            "session-separated photos from the actual panel and treat own-test results "
            "as the deployment measure.",
            "- The classifier has no `cracked` or `shaded` output.",
            "- The reference-based dust index is heuristic and uncalibrated until "
            "measured using real dust on the deployment panel.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    MODEL_ROOT.mkdir(parents=True, exist_ok=True)
    if not CHECKPOINT_PATH.is_file():
        raise FileNotFoundError(f"Best checkpoint is missing: {CHECKPOINT_PATH}")
    torch.set_num_threads(min(8, torch.get_num_threads()))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=True)
    model = make_model()
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)

    rows = load_test_rows()
    test_parent_count = len({row.parent_path for row in rows})
    LOGGER.info("Evaluating frozen test split: patches=%d parents=%d", len(rows), test_parent_count)
    probabilities = infer(model, rows, device)
    patch_true = np.asarray([row.true_label for row in rows], dtype=np.int64)
    patch_predicted = probabilities.argmax(axis=1)
    patch_confidence = probabilities.max(axis=1)
    patch_metrics = metrics_for(patch_true, patch_predicted, patch_confidence)

    parents, parent_true, parent_predicted = parent_aggregate(rows, probabilities)
    parent_confidence = np.asarray([parent["confidence"] for parent in parents])
    parent_metrics = metrics_for(parent_true, parent_predicted, parent_confidence)
    parent_metrics["samples"] = len(parents)
    bootstrap = bootstrap_intervals(parents)
    per_source = split_by_source(rows, probabilities)
    robustness = robustness_metrics(
        model,
        rows,
        patch_metrics["accuracy"],
        parent_metrics["accuracy"],
        device,
    )
    heldout = evaluate_heldout(model, device)
    wrong_counts = save_wrong_images(
        parents,
        REPORT_ROOT / "main_test_errors_12.png",
        REPORT_ROOT / "main_test_errors_12.csv",
    )
    save_confusion_matrix(
        parent_metrics, REPORT_ROOT / "main_test_confusion_matrix.png"
    )

    training_log_matches = sorted(LOG_ROOT.glob("main_training_*.log"))
    training_seconds = 0.0
    if training_log_matches:
        for line in training_log_matches[-1].read_text(encoding="utf-8").splitlines():
            if "full_training_seconds=" in line:
                try:
                    training_seconds = float(line.split("full_training_seconds=", 1)[1].split()[0])
                except ValueError:
                    LOGGER.warning("Could not parse runtime from %s", training_log_matches[-1])

    output: dict[str, Any] = {
        "labels": list(LABELS),
        "test_split": str(TEST_PATH.relative_to(PROJECT_ROOT)),
        "patch": patch_metrics,
        "parent": parent_metrics,
        "parent_bootstrap_95": bootstrap,
        "bootstrap_iterations": BOOTSTRAP_ITERATIONS,
        "per_source": per_source,
        "robustness": robustness,
        "heldout": heldout,
        "wrong_parent_counts": wrong_counts,
        "split_sizes": split_sizes(),
        "model_confidence_threshold_reference": 0.80,
        "training_seconds": training_seconds,
    }
    write_metrics_report(
        output,
        REPORT_ROOT / "evaluation.json",
        REPORT_ROOT / "evaluation.md",
        MODEL_ROOT / "model_card.md",
        training_seconds,
    )
    LOGGER.info("Patch metrics: %s", json.dumps(patch_metrics, sort_keys=True))
    LOGGER.info("Parent metrics: %s", json.dumps(parent_metrics, sort_keys=True))
    LOGGER.info("Parent bootstrap 95%% intervals: %s", json.dumps(bootstrap, sort_keys=True))
    LOGGER.info("Per-source parent metrics: %s", json.dumps(per_source, sort_keys=True))
    LOGGER.info("Robustness: %s", json.dumps(robustness, sort_keys=True))
    LOGGER.info("Held-out predictions: %s", json.dumps(heldout, sort_keys=True))
    LOGGER.info("Confidence threshold reference 0.80; see evaluation report for measured summaries.")
    LOGGER.info("Wrong parent images selected: %s", wrong_counts)
    LOGGER.info("Reports: %s", REPORT_ROOT)


if __name__ == "__main__":
    main()

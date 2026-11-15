"""Compare pretrained architectures and synthetic dust using validation only."""

from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
import logging
import math
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
from PIL import Image, ImageOps
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from torchvision.models import (
    EfficientNet_B0_Weights,
    MobileNet_V2_Weights,
    ResNet18_Weights,
)
from torchvision.transforms import InterpolationMode

if __package__:
    from .synth_dust import LEVELS, save_example_grids, synthesize_image
else:
    from synth_dust import LEVELS, save_example_grids, synthesize_image


ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
SPLIT_ROOT = ROOT / "splits" / "main"
REPORT_ROOT = ROOT / "reports"
CHECKPOINT_ROOT = ROOT / "checkpoints" / "public_compare"
MODEL_ROOT = PROJECT_ROOT / "model"
LABELS = ("clean", "dusty")
LABEL_TO_INDEX = {label: index for index, label in enumerate(LABELS)}
INPUT_SIZE = 224
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
SEED = 42
BATCH_SIZE = 32
LOGGER = logging.getLogger("compare_models")
ARCHITECTURES = ("mobilenet_v2", "efficientnet_b0", "resnet18")


@dataclass(frozen=True)
class ImageRow:
    image_path: Path
    relative_path: str
    label: int
    parent_path: str
    group_id: str
    source: str
    synthetic_level: str = ""


class PatchDataset(Dataset[tuple[torch.Tensor, int]]):
    def __init__(self, rows: list[ImageRow], augment: bool) -> None:
        self.rows = rows
        operations: list[Any] = []
        if augment:
            operations.extend(
                [
                    transforms.RandomResizedCrop(
                        INPUT_SIZE,
                        scale=(0.4, 1.0),
                        ratio=(0.9, 1.1),
                        interpolation=InterpolationMode.BILINEAR,
                    ),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomRotation(10, interpolation=InterpolationMode.BILINEAR),
                    transforms.ColorJitter(
                        brightness=0.25, contrast=0.25, saturation=0.20
                    ),
                    transforms.RandomApply(
                        [transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 1.5))],
                        p=0.20,
                    ),
                ]
            )
        else:
            operations.extend(
                [
                    transforms.Resize(256, interpolation=InterpolationMode.BILINEAR),
                    transforms.CenterCrop(INPUT_SIZE),
                ]
            )
        operations.extend(
            [
                transforms.ToTensor(),
                transforms.Normalize(MEAN, STD),
            ]
        )
        self.transform = transforms.Compose(operations)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        row = self.rows[index]
        with Image.open(row.image_path) as opened:
            image = ImageOps.exif_transpose(opened).convert("RGB")
        return self.transform(image), row.label


def load_split(name: str) -> list[ImageRow]:
    path = SPLIT_ROOT / f"{name}.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Required {name} split does not exist: {path}")
    rows: list[ImageRow] = []
    with path.open(newline="", encoding="utf-8") as stream:
        for item in csv.DictReader(stream):
            if item["label"] not in LABEL_TO_INDEX:
                raise ValueError(f"Unexpected label in {path}: {item['label']}")
            relative_path = item["patch_path"]
            image_path = ROOT / relative_path
            if not image_path.is_file():
                raise FileNotFoundError(f"Split image is missing: {image_path}")
            rows.append(
                ImageRow(
                    image_path=image_path,
                    relative_path=relative_path,
                    label=LABEL_TO_INDEX[item["label"]],
                    parent_path=item["parent_path"],
                    group_id=item["group_id"],
                    source=item["source"],
                )
            )
    return rows


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_model(architecture: str, pretrained: bool = True) -> nn.Module:
    if architecture == "mobilenet_v2":
        weights = MobileNet_V2_Weights.DEFAULT if pretrained else None
        model = models.mobilenet_v2(weights=weights)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(LABELS))
    elif architecture == "efficientnet_b0":
        weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(LABELS))
    elif architecture == "resnet18":
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, len(LABELS))
    else:
        raise ValueError(f"Unsupported architecture: {architecture}")
    return model


def classifier_parameters(model: nn.Module, architecture: str) -> list[nn.Parameter]:
    head = model.fc if architecture == "resnet18" else model.classifier
    return list(head.parameters())


def configure_trainable(
    model: nn.Module, architecture: str, fine_tune: bool
) -> list[nn.Parameter]:
    for parameter in model.parameters():
        parameter.requires_grad = False
    parameters = classifier_parameters(model, architecture)
    for parameter in parameters:
        parameter.requires_grad = True
    if fine_tune:
        if architecture == "mobilenet_v2":
            unfrozen_layers = [model.features[-1]]
        elif architecture == "efficientnet_b0":
            unfrozen_layers = [model.features[-1]]
        else:
            unfrozen_layers = [model.layer4]
        for layer in unfrozen_layers:
            for parameter in layer.parameters():
                parameter.requires_grad = True
            parameters.extend(parameter for parameter in layer.parameters())
    return list({id(parameter): parameter for parameter in parameters}.values())


def set_training_mode(model: nn.Module) -> None:
    model.train()
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.eval()


def class_weights(rows: list[ImageRow], device: torch.device) -> torch.Tensor:
    counts = Counter(row.label for row in rows)
    if any(counts[label] == 0 for label in range(len(LABELS))):
        raise ValueError(f"Training rows must contain both labels; counts={counts}")
    weights = torch.tensor(
        [1.0 / counts[label] for label in range(len(LABELS))],
        dtype=torch.float32,
        device=device,
    )
    return weights / weights.mean()


def make_loader(
    rows: list[ImageRow], augment: bool, shuffle: bool
) -> DataLoader[tuple[torch.Tensor, int]]:
    return DataLoader(
        PatchDataset(rows, augment),
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=False,
    )


def run_epoch(
    model: nn.Module,
    loader: DataLoader[tuple[torch.Tensor, int]],
    device: torch.device,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
) -> tuple[float, float]:
    if optimizer is None:
        model.eval()
    else:
        set_training_mode(model)
    loss_total = 0.0
    correct_total = 0
    item_total = 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(optimizer is not None):
            logits = model(images)
            loss = criterion(logits, labels)
            if optimizer is not None:
                loss.backward()
                optimizer.step()
        batch_count = labels.size(0)
        loss_total += float(loss.item()) * batch_count
        correct_total += int((logits.argmax(dim=1) == labels).sum().item())
        item_total += batch_count
    if item_total == 0:
        raise ValueError("Cannot evaluate/training on an empty loader")
    return loss_total / item_total, correct_total / item_total


def predict_logits(
    model: nn.Module, rows: list[ImageRow], device: torch.device
) -> np.ndarray:
    loader = make_loader(rows, augment=False, shuffle=False)
    model.eval()
    batches: list[np.ndarray] = []
    with torch.inference_mode():
        for images, _labels in loader:
            logits = model(images.to(device, non_blocking=True))
            batches.append(logits.cpu().numpy().astype(np.float64, copy=False))
    if not batches:
        raise ValueError("Cannot predict an empty split")
    return np.concatenate(batches, axis=0)


def parent_level(
    rows: list[ImageRow], logits: np.ndarray
) -> tuple[list[dict[str, Any]], np.ndarray]:
    indices_by_parent: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        indices_by_parent[row.parent_path].append(index)
    parents: list[dict[str, Any]] = []
    for parent_path, indices in sorted(indices_by_parent.items()):
        labels = {rows[index].label for index in indices}
        if len(labels) != 1:
            raise ValueError(f"Mixed labels within parent {parent_path}")
        source_values = {rows[index].source for index in indices}
        if len(source_values) != 1:
            raise ValueError(f"Mixed sources within parent {parent_path}")
        mean_probabilities = softmax(logits[indices]).mean(axis=0)
        calibration_logits = np.log(np.clip(mean_probabilities, 1e-12, 1.0))
        parents.append(
            {
                "parent_path": parent_path,
                "label": labels.pop(),
                "source": source_values.pop(),
                "logits": calibration_logits,
                "probabilities": mean_probabilities,
                "prediction": int(np.argmax(mean_probabilities)),
            }
        )
    return parents, np.stack([parent["logits"] for parent in parents], axis=0)


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    exp_logits = np.exp(shifted)
    return exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)


def binary_metrics(
    parents: list[dict[str, Any]], probabilities: np.ndarray | None = None
) -> dict[str, Any]:
    labels = np.asarray([parent["label"] for parent in parents], dtype=np.int64)
    if probabilities is None:
        probabilities = np.stack(
            [parent["probabilities"] for parent in parents], axis=0
        )
    predictions = (probabilities[:, 1] >= 0.5).astype(np.int64)
    matrix = confusion_matrix(labels, predictions, labels=[0, 1])
    clean_total = int(np.sum(labels == 0))
    true_dusty = int(matrix[1, 1])
    dusty_total = int(np.sum(labels == 1))
    false_dusty = int(matrix[0, 1])
    return {
        "parents": len(parents),
        "accuracy": float(accuracy_score(labels, predictions)),
        "dusty_recall": float(true_dusty / dusty_total) if dusty_total else 0.0,
        "false_dusty_rate_on_clean": (
            float(false_dusty / clean_total) if clean_total else 0.0
        ),
        "dusty_precision": float(precision_score(labels, predictions, pos_label=1, zero_division=0)),
        "confusion_matrix": matrix.tolist(),
    }


def fit_temperature(parent_logits: np.ndarray, labels: np.ndarray) -> float:
    logits = torch.tensor(parent_logits, dtype=torch.float64)
    targets = torch.tensor(labels, dtype=torch.long)
    log_temperature = torch.zeros((), dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.LBFGS(
        [log_temperature], lr=0.1, max_iter=100, line_search_fn="strong_wolfe"
    )

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        temperature = log_temperature.exp().clamp(0.05, 20.0)
        loss = nn.functional.cross_entropy(logits / temperature, targets)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(log_temperature.detach().exp().clamp(0.05, 20.0).item())


def expected_calibration_error(
    probabilities: np.ndarray, labels: np.ndarray, bins: int = 10
) -> tuple[float, list[dict[str, float | int]]]:
    confidences = probabilities.max(axis=1)
    predictions = probabilities.argmax(axis=1)
    bin_edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    reliability: list[dict[str, float | int]] = []
    for index in range(bins):
        if index == bins - 1:
            selected = (confidences >= bin_edges[index]) & (
                confidences <= bin_edges[index + 1]
            )
        else:
            selected = (confidences >= bin_edges[index]) & (
                confidences < bin_edges[index + 1]
            )
        count = int(selected.sum())
        if count == 0:
            reliability.append(
                {
                    "bin_lower": float(bin_edges[index]),
                    "bin_upper": float(bin_edges[index + 1]),
                    "count": 0,
                    "mean_confidence": 0.0,
                    "accuracy": 0.0,
                }
            )
            continue
        mean_confidence = float(confidences[selected].mean())
        accuracy = float(np.mean(predictions[selected] == labels[selected]))
        ece += count / len(labels) * abs(accuracy - mean_confidence)
        reliability.append(
            {
                "bin_lower": float(bin_edges[index]),
                "bin_upper": float(bin_edges[index + 1]),
                "count": count,
                "mean_confidence": mean_confidence,
                "accuracy": accuracy,
            }
        )
    return float(ece), reliability


def threshold_metrics(
    probabilities: np.ndarray, labels: np.ndarray, thresholds: tuple[float, ...]
) -> dict[str, dict[str, float | int]]:
    results: dict[str, dict[str, float | int]] = {}
    for threshold in thresholds:
        predictions = (probabilities[:, 1] >= threshold).astype(np.int64)
        matrix = confusion_matrix(labels, predictions, labels=[0, 1])
        tp = int(matrix[1, 1])
        fp = int(matrix[0, 1])
        fn = int(matrix[1, 0])
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        results[f"{threshold:.1f}"] = {
            "threshold": threshold,
            "precision": precision,
            "recall": recall,
            "predicted_dusty": int(predictions.sum()),
            "confusion_matrix": matrix.tolist(),
        }
    return results


def plot_reliability(
    reliability: list[dict[str, float | int]], path: Path
) -> None:
    points = [
        item
        for item in reliability
        if int(item["count"]) > 0
    ]
    figure, axis = plt.subplots(figsize=(6, 6))
    axis.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect calibration")
    axis.plot(
        [float(item["mean_confidence"]) for item in points],
        [float(item["accuracy"]) for item in points],
        marker="o",
        label="Validation parent predictions",
    )
    axis.set(
        xlabel="Mean confidence",
        ylabel="Accuracy",
        title="Validation reliability diagram",
        xlim=(0, 1),
        ylim=(0, 1),
    )
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=160)
    plt.close(figure)


def save_state(
    model: nn.Module,
    architecture: str,
    path: Path,
    *,
    temperature: float = 1.0,
    metadata: dict[str, Any] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "architecture": architecture,
            "state_dict": {
                key: value.detach().cpu() for key, value in model.state_dict().items()
            },
            "labels": list(LABELS),
            "input_size": INPUT_SIZE,
            "mean": list(MEAN),
            "std": list(STD),
            "temperature": temperature,
            "metadata": metadata or {},
        },
        path,
    )


def load_state(path: Path, device: torch.device) -> tuple[nn.Module, dict[str, Any]]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    architecture = checkpoint["architecture"]
    model = create_model(architecture, pretrained=False)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()
    return model, checkpoint


def train_variant(
    name: str,
    architecture: str,
    train_rows: list[ImageRow],
    validation_rows: list[ImageRow],
    device: torch.device,
    head_epochs: int,
    fine_tune_epochs: int,
) -> dict[str, Any]:
    seed_everything(SEED)
    started = time.perf_counter()
    LOGGER.info("variant=%s initializing pretrained architecture=%s", name, architecture)
    model = create_model(architecture, pretrained=True).to(device)
    train_loader = make_loader(train_rows, augment=True, shuffle=True)
    validation_loader = make_loader(validation_rows, augment=False, shuffle=False)
    criterion = nn.CrossEntropyLoss(
        weight=class_weights(train_rows, device),
        label_smoothing=0.1,
    )
    best_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    history: list[dict[str, Any]] = []

    for phase, epoch_limit in (("head", head_epochs), ("fine_tune", fine_tune_epochs)):
        if epoch_limit == 0:
            continue
        parameters = configure_trainable(
            model, architecture, fine_tune=phase == "fine_tune"
        )
        learning_rate = 1e-3 if phase == "head" else 2e-5
        optimizer = torch.optim.AdamW(
            parameters, lr=learning_rate, weight_decay=1e-4
        )
        stale_epochs = 0
        phase_best = best_loss
        phase_start_best = best_loss
        for epoch in range(1, epoch_limit + 1):
            epoch_started = time.perf_counter()
            train_loss, train_accuracy = run_epoch(
                model, train_loader, device, criterion, optimizer
            )
            validation_loss, validation_accuracy = run_epoch(
                model, validation_loader, device, criterion, None
            )
            record = {
                "phase": phase,
                "epoch": epoch,
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "validation_loss": validation_loss,
                "validation_accuracy": validation_accuracy,
                "seconds": time.perf_counter() - epoch_started,
                "learning_rate": learning_rate,
                "unfrozen_stage": (
                    "classifier"
                    if phase == "head"
                    else {
                        "mobilenet_v2": "features[-1] + classifier",
                        "efficientnet_b0": "features[-1] + classifier",
                        "resnet18": "layer4 + fc",
                    }[architecture]
                ),
            }
            history.append(record)
            LOGGER.info(
                "variant=%s phase=%s epoch=%d/%d train_loss=%.4f "
                "train_accuracy=%.4f val_loss=%.4f val_accuracy=%.4f "
                "seconds=%.1f lr=%.1e",
                name,
                phase,
                epoch,
                epoch_limit,
                train_loss,
                train_accuracy,
                validation_loss,
                validation_accuracy,
                record["seconds"],
                learning_rate,
            )
            if validation_loss < best_loss:
                best_loss = validation_loss
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                }
                stale_epochs = 0
            else:
                stale_epochs += 1
            if validation_loss < phase_best:
                phase_best = validation_loss
            if stale_epochs >= 2:
                LOGGER.info(
                    "variant=%s phase=%s early_stop epoch=%d; restoring best validation checkpoint",
                    name,
                    phase,
                    epoch,
                )
                break

        if phase == "fine_tune" and phase_best >= phase_start_best:
            LOGGER.info(
                "variant=%s fine-tuning did not improve validation loss; retaining "
                "the more-frozen head checkpoint",
                name,
            )
        if best_state is not None:
            model.load_state_dict(best_state)

    if best_state is None:
        raise RuntimeError(f"Variant {name} did not produce a validation checkpoint")
    model.load_state_dict(best_state)
    validation_logits = predict_logits(model, validation_rows, device)
    validation_parents, _ = parent_level(validation_rows, validation_logits)
    validation_metrics = binary_metrics(validation_parents)
    elapsed = time.perf_counter() - started
    checkpoint_path = CHECKPOINT_ROOT / f"{name}.pt"
    save_state(
        model,
        architecture,
        checkpoint_path,
        metadata={
            "name": name,
            "best_validation_loss": best_loss,
            "training_seconds": elapsed,
            "history": history,
        },
    )
    return {
        "name": name,
        "architecture": architecture,
        "validation": validation_metrics,
        "validation_logits": validation_logits,
        "training_seconds": elapsed,
        "best_validation_loss": best_loss,
        "history": history,
        "checkpoint_path": checkpoint_path,
        "synthetic": False,
    }


def evaluate_existing_mobilenet(
    validation_rows: list[ImageRow], device: torch.device
) -> dict[str, Any]:
    existing_path = MODEL_ROOT / "model.pt"
    if not existing_path.is_file():
        LOGGER.warning("No current model.pt found; skipping existing checkpoint baseline")
        return {}
    checkpoint = torch.load(existing_path, map_location="cpu", weights_only=True)
    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(LABELS))
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    logits = predict_logits(model, validation_rows, device)
    parents, _ = parent_level(validation_rows, logits)
    return {
        "name": "mobilenet_v2_current",
        "architecture": "mobilenet_v2",
        "validation": binary_metrics(parents),
        "validation_logits": logits,
        "training_seconds": None,
        "best_validation_loss": None,
        "history": [],
        "checkpoint_path": existing_path,
        "synthetic": False,
        "existing_checkpoint": True,
    }


def build_synthetic_training_rows(
    train_rows: list[ImageRow],
    output_root: Path,
    seed: int,
) -> tuple[list[ImageRow], list[np.ndarray], dict[str, Any]]:
    dusty_rows = [row for row in train_rows if row.label == LABEL_TO_INDEX["dusty"]]
    clean_rows = [row for row in train_rows if row.label == LABEL_TO_INDEX["clean"]]
    if not dusty_rows or not clean_rows:
        raise ValueError("Synthetic training requires clean and real dusty training patches")
    run_seed = seed
    rng = random.Random(run_seed)
    clean_sources = clean_rows.copy()
    rng.shuffle(clean_sources)
    selected_sources = [
        clean_sources[index % len(clean_sources)] for index in range(len(dusty_rows))
    ]
    generated_root = output_root / "synthetic_dusty"
    generated_rows: list[ImageRow] = []
    by_level: dict[str, list[Path]] = defaultdict(list)
    for index, source_row in enumerate(selected_sources):
        level = LEVELS[index % len(LEVELS)]
        synthetic_seed = seed + index
        destination = generated_root / level / f"synthetic_{index:05d}.jpg"
        destination.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(source_row.image_path) as opened:
            rgb = np.asarray(ImageOps.exif_transpose(opened).convert("RGB"))
        synthetic_image = synthesize_image(rgb, level, synthetic_seed)
        Image.fromarray(synthetic_image).save(destination, quality=95)
        by_level[level].append(destination)
        generated_rows.append(
            ImageRow(
                image_path=destination,
                relative_path=str(destination.relative_to(ROOT)),
                label=LABEL_TO_INDEX["dusty"],
                parent_path=f"synthetic:{source_row.parent_path}:{index}",
                group_id=f"synthetic:{source_row.group_id}:{index}",
                source="synthetic_dust",
                synthetic_level=level,
            )
        )

    sample_arrays: list[np.ndarray] = []
    for source_row in clean_sources[: min(16, len(clean_sources))]:
        with Image.open(source_row.image_path) as opened:
            sample_arrays.append(np.asarray(ImageOps.exif_transpose(opened).convert("RGB")))
    grid_paths = save_example_grids(sample_arrays, REPORT_ROOT, seed=seed)
    for level, paths in by_level.items():
        if not paths:
            raise RuntimeError(f"No synthetic examples were generated for level {level}")
    synthetic_rows = list(train_rows) + generated_rows
    manifest_path = output_root / "synthetic_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=(
                "synthetic_path",
                "level",
                "source_patch",
                "source_parent",
                "source_split",
                "seed",
            ),
        )
        writer.writeheader()
        for index, (source_row, synthetic_row) in enumerate(
            zip(selected_sources, generated_rows)
        ):
            writer.writerow(
                {
                    "synthetic_path": synthetic_row.relative_path,
                    "level": synthetic_row.synthetic_level,
                    "source_patch": source_row.relative_path,
                    "source_parent": source_row.parent_path,
                    "source_split": "train",
                    "seed": seed + index,
                }
            )
    info = {
        "real_dusty_training_patches": len(dusty_rows),
        "synthetic_dusty_training_patches": len(generated_rows),
        "dust_levels": dict(Counter(row.synthetic_level for row in generated_rows)),
        "clean_training_patches": len(clean_rows),
        "synthetic_data_root": str(output_root.relative_to(PROJECT_ROOT)),
        "synthetic_manifest": str(manifest_path.relative_to(PROJECT_ROOT)),
        "example_grids": {
            level: str(path.relative_to(PROJECT_ROOT))
            for level, path in grid_paths.items()
        },
        "validation_and_test_synthetic_count": 0,
    }
    return synthetic_rows, sample_arrays, info


def choose_candidate(results: list[dict[str, Any]]) -> dict[str, Any]:
    return max(
        results,
        key=lambda result: (
            result["validation"]["accuracy"],
            result["validation"]["dusty_recall"],
            -result["validation"]["false_dusty_rate_on_clean"],
            -float(result["best_validation_loss"] or math.inf),
        ),
    )


def parent_bootstrap(
    labels: np.ndarray, predictions: np.ndarray, seed: int = SEED, iterations: int = 2000
) -> dict[str, list[float]]:
    rng = np.random.default_rng(seed)
    indices = np.arange(len(labels))
    metrics: dict[str, list[float]] = {
        "accuracy": [],
        "dusty_recall": [],
        "false_dusty_rate_on_clean": [],
    }
    for _ in range(iterations):
        sample = rng.choice(indices, size=len(indices), replace=True)
        y = labels[sample]
        pred = predictions[sample]
        metrics["accuracy"].append(float(np.mean(y == pred)))
        dusty_mask = y == 1
        clean_mask = y == 0
        metrics["dusty_recall"].append(
            float(np.mean(pred[dusty_mask] == 1)) if np.any(dusty_mask) else float("nan")
        )
        metrics["false_dusty_rate_on_clean"].append(
            float(np.mean(pred[clean_mask] == 1)) if np.any(clean_mask) else float("nan")
        )
    return {
        metric: [
            float(np.nanpercentile(values, 2.5)),
            float(np.nanpercentile(values, 97.5)),
        ]
        for metric, values in metrics.items()
    }


def final_test_once(
    checkpoint_path: Path,
    temperature: float,
    device: torch.device,
) -> dict[str, Any]:
    test_rows = load_split("test")
    model, _checkpoint = load_state(checkpoint_path, device)
    logits = predict_logits(model, test_rows, device)
    parents, parent_logits = parent_level(test_rows, logits)
    labels = np.asarray([parent["label"] for parent in parents], dtype=np.int64)
    probabilities = softmax(parent_logits / temperature)
    predictions = (probabilities[:, 1] >= 0.5).astype(np.int64)
    matrix = confusion_matrix(labels, predictions, labels=[0, 1])
    dusty_total = int(np.sum(labels == 1))
    clean_total = int(np.sum(labels == 0))
    correct = int(np.sum(predictions == labels))
    output = {
        "protocol": "Single final evaluation on untouched test split after validation selection and temperature calibration.",
        "test_parents": len(parents),
        "test_patches": len(test_rows),
        "parent_accuracy": correct / len(parents),
        "dusty_recall": float(matrix[1, 1] / dusty_total) if dusty_total else 0.0,
        "false_dusty_rate_on_clean": (
            float(matrix[0, 1] / clean_total) if clean_total else 0.0
        ),
        "confusion_matrix_clean_dusty": matrix.tolist(),
        "bootstrap_parent_95": parent_bootstrap(labels, predictions),
        "temperature_fitted_on_validation": temperature,
        "clean_parents": clean_total,
        "dusty_parents": dusty_total,
    }
    (REPORT_ROOT / "model_v2_test.json").write_text(
        json.dumps(output, indent=2), encoding="utf-8"
    )
    return output


def write_model_card(
    selected: dict[str, Any],
    temperature: float,
    calibration: dict[str, Any],
    variants: list[dict[str, Any]],
    synthetic_info: dict[str, Any],
    final_test: dict[str, Any],
    split_sizes: dict[str, dict[str, int]],
    output_path: Path,
) -> None:
    versions = {
        package: importlib.metadata.version(package)
        for package in ("Pillow", "matplotlib", "numpy", "scikit-learn", "torch", "torchvision")
    }
    lines = [
        "# Solar-panel clean/dusty model v2",
        "",
        f"- Generated: {time.strftime('%Y-%m-%d')}",
        f"- Selected architecture: `{selected['architecture']}`.",
        "- Task: clean versus dusty, binary classification only.",
        f"- Input: {INPUT_SIZE}×{INPUT_SIZE} RGB; ImageNet mean/std normalization.",
        f"- Selected run training time: {selected['training_seconds']:.1f} seconds.",
        f"- Validation-selected temperature: {temperature:.6f}.",
        f"- Validation ECE before temperature scaling: {calibration['ece_before']:.4f}.",
        f"- Validation ECE after temperature scaling: {calibration['ece_after']:.4f}.",
        "- Selection used validation parent accuracy; ties were broken by dusty recall, "
        "then lower false-dusty rate. The held-out test split was not used for selection.",
        "",
        "## Data sources and split sizes",
        "",
        "- `solar-panel-dust-detection` ([Kaggle](https://www.kaggle.com/datasets/hemanthsai7/solar-panel-dust-detection)): CC0-1.0.",
        "- `pv-panel-defect-dataset` ([Kaggle](https://www.kaggle.com/datasets/alicjalena/pv-panel-defect-dataset)): CC-BY-NC-SA-4.0; non-commercial/share-alike terms apply.",
        "- Only clean and dusty classes are included; the cracked experiment was skipped after visual review.",
        "",
        "| Split | Patches | Parent images |",
        "|---|---:|---:|",
        f"| Train | {split_sizes['train']['patches']} | {split_sizes['train']['parents']} |",
        f"| Validation | {split_sizes['validation']['patches']} | {split_sizes['validation']['parents']} |",
        f"| Test | {final_test['test_patches']} | {final_test['test_parents']} |",
        "",
        "## Runtime versions",
        "",
        f"- Python: {platform.python_version()}",
    ]
    lines.extend(f"- {package}: {version}" for package, version in versions.items())
    lines.extend(
        [
            "",
        "## Public test (single final evaluation)",
        "",
        f"- Parent accuracy: {final_test['parent_accuracy']:.4f}; 95% parent-bootstrap CI "
        f"[{final_test['bootstrap_parent_95']['accuracy'][0]:.4f}, "
        f"{final_test['bootstrap_parent_95']['accuracy'][1]:.4f}].",
        f"- Dusty recall: {final_test['dusty_recall']:.4f}; 95% CI "
        f"[{final_test['bootstrap_parent_95']['dusty_recall'][0]:.4f}, "
        f"{final_test['bootstrap_parent_95']['dusty_recall'][1]:.4f}].",
        f"- False-dusty rate on clean: {final_test['false_dusty_rate_on_clean']:.4f}; "
        f"95% CI [{final_test['bootstrap_parent_95']['false_dusty_rate_on_clean'][0]:.4f}, "
        f"{final_test['bootstrap_parent_95']['false_dusty_rate_on_clean'][1]:.4f}].",
        f"- Confusion matrix (rows true clean/dusty, columns predicted clean/dusty): "
        f"`{final_test['confusion_matrix_clean_dusty']}`.",
        "",
        "## Validation comparison",
        "",
        "| Variant | Architecture | Parent accuracy | Dusty recall | False-dusty rate | Training seconds |",
        "|---|---|---:|---:|---:|---:|",
    ]
    )
    for result in variants:
        seconds = (
            "existing"
            if result["training_seconds"] is None
            else f'{result["training_seconds"]:.1f}'
        )
        metrics = result["validation"]
        lines.append(
            f"| {result['name']} | {result['architecture']} | {metrics['accuracy']:.4f} | "
            f"{metrics['dusty_recall']:.4f} | "
            f"{metrics['false_dusty_rate_on_clean']:.4f} | {seconds} |"
        )
    lines.extend(
        [
            "",
            "Synthetic dusty was included only when its real-validation dusty recall "
            "improved over the selected architecture's real-only variant.",
            f"- Real dusty training patches: {synthetic_info['real_dusty_training_patches']}.",
            f"- Synthetic dusty training patches: {synthetic_info['synthetic_dusty_training_patches']}.",
            "- Synthetic examples were created only from clean training patches. "
            "No synthetic data was added to validation or test.",
            "",
            "## Calibration",
            "",
            f"- Validation reliability plot: `{calibration['reliability_plot']}`.",
            "- Parent-level dusty precision and recall at thresholds:",
            "",
            "| Dusty threshold | Precision | Recall | Predicted dusty parents |",
            "|---:|---:|---:|---:|",
        ]
    )
    for threshold, metrics in calibration["threshold_metrics"].items():
        lines.append(
            f"| {threshold} | {metrics['precision']:.4f} | {metrics['recall']:.4f} | "
            f"{metrics['predicted_dusty']} |"
        )
    lines.extend(
        [
            "",
            "## Known limitations",
            "",
            "- Public images are predominantly broad panel/array imagery and do not match "
            "the intended close-up fixed webcam view.",
            "- Validation selection and temperature calibration use the same validation "
            "split, so the resulting validation metrics are selection/calibration estimates, "
            "not an independent final quality estimate.",
            "- Synthetic dust is an approximation; real dust grain, adhesion, glare, "
            "camera optics, and lighting on the actual panel may differ.",
            "- Test results describe public data only. Own-panel session-separated photos "
            "are still needed before deployment or setting a cleaning threshold.",
            "- The classifier has no cracked or shaded output.",
            "",
        ]
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def run_experiments(
    head_epochs: int,
    fine_tune_epochs: int,
    skip_synthetic: bool,
) -> None:
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_ROOT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(min(8, torch.get_num_threads()))
    torch.hub.set_dir(str(ROOT / "cache" / "torch"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    LOGGER.info("device=%s torch=%s torchvision=%s", device, torch.__version__, __import__("torchvision").__version__)

    train_rows = load_split("train")
    validation_rows = load_split("val")
    train_parents = {row.parent_path for row in train_rows}
    validation_parents = {row.parent_path for row in validation_rows}
    if train_parents & validation_parents:
        raise RuntimeError("Parent-image leakage between training and validation splits")
    train_groups = {row.group_id for row in train_rows}
    validation_groups = {row.group_id for row in validation_rows}
    if train_groups & validation_groups:
        raise RuntimeError("Duplicate-group leakage between training and validation splits")
    LOGGER.info(
        "train patches=%d parents=%d; validation patches=%d parents=%d",
        len(train_rows),
        len(train_parents),
        len(validation_rows),
        len(validation_parents),
    )

    variants: list[dict[str, Any]] = []
    existing = evaluate_existing_mobilenet(validation_rows, device)
    if existing:
        variants.append(existing)

    for architecture in ARCHITECTURES:
        variants.append(
            train_variant(
                name=f"{architecture}_regularized",
                architecture=architecture,
                train_rows=train_rows,
                validation_rows=validation_rows,
                device=device,
                head_epochs=head_epochs,
                fine_tune_epochs=fine_tune_epochs,
            )
        )

    real_candidates = [
        result
        for result in variants
        if not result.get("synthetic") and result.get("training_seconds") is not None
    ]
    base_winner = choose_candidate(real_candidates)
    selected = base_winner
    synthetic_info: dict[str, Any] = {
        "real_dusty_training_patches": 0,
        "synthetic_dusty_training_patches": 0,
        "dust_levels": {},
        "validation_and_test_synthetic_count": 0,
        "used": False,
    }
    if not skip_synthetic:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        synthetic_root = ROOT / "data" / "synthetic_dust" / f"compare_{timestamp}"
        combined_rows, _samples, synthetic_info = build_synthetic_training_rows(
            train_rows, synthetic_root, SEED
        )
        synthetic_variant = train_variant(
            name=f"{base_winner['architecture']}_synthetic",
            architecture=base_winner["architecture"],
            train_rows=combined_rows,
            validation_rows=validation_rows,
            device=device,
            head_epochs=head_epochs,
            fine_tune_epochs=fine_tune_epochs,
        )
        synthetic_variant["synthetic"] = True
        variants.append(synthetic_variant)
        recall_improved = (
            synthetic_variant["validation"]["dusty_recall"]
            > base_winner["validation"]["dusty_recall"]
        )
        synthetic_info["used"] = recall_improved
        synthetic_info["validation_dusty_recall_improved"] = recall_improved
        synthetic_info["base_variant"] = base_winner["name"]
        synthetic_info["synthetic_variant"] = synthetic_variant["name"]
        if recall_improved:
            selected = synthetic_variant
            LOGGER.info(
                "Keeping synthetic variant: real-validation dusty recall %.4f > %.4f",
                synthetic_variant["validation"]["dusty_recall"],
                base_winner["validation"]["dusty_recall"],
            )
        else:
            LOGGER.info(
                "Rejecting synthetic variant: real-validation dusty recall %.4f did not improve over %.4f",
                synthetic_variant["validation"]["dusty_recall"],
                base_winner["validation"]["dusty_recall"],
            )

    selected_checkpoint = selected["checkpoint_path"]
    selected_model, _ = load_state(selected_checkpoint, device)
    selected_logits = predict_logits(selected_model, validation_rows, device)
    validation_parents, parent_logits = parent_level(validation_rows, selected_logits)
    validation_labels = np.asarray(
        [parent["label"] for parent in validation_parents], dtype=np.int64
    )
    temperature = fit_temperature(parent_logits, validation_labels)
    calibrated_probabilities = softmax(parent_logits / temperature)
    ece_before, _ = expected_calibration_error(
        softmax(parent_logits), validation_labels
    )
    ece_after, reliability = expected_calibration_error(
        calibrated_probabilities, validation_labels
    )
    reliability_path = REPORT_ROOT / "model_v2_reliability.png"
    plot_reliability(reliability, reliability_path)
    calibration = {
        "temperature": temperature,
        "ece_before": ece_before,
        "ece_after": ece_after,
        "threshold_metrics": threshold_metrics(
            calibrated_probabilities, validation_labels, (0.6, 0.7, 0.8, 0.9)
        ),
        "reliability": reliability,
        "reliability_plot": str(reliability_path.relative_to(PROJECT_ROOT)),
        "validation_parent_metrics_calibrated": binary_metrics(
            validation_parents, calibrated_probabilities
        ),
    }

    final_path = MODEL_ROOT / "model_v2.pt"
    save_state(
        selected_model,
        selected["architecture"],
        final_path,
        temperature=temperature,
        metadata={
            "selected_variant": selected["name"],
            "validation_parent_metrics": selected["validation"],
            "calibration": calibration,
            "synthetic": synthetic_info,
        },
    )
    labels_path = MODEL_ROOT / "labels_v2.json"
    labels_path.write_text(
        json.dumps(
            {
                "labels": list(LABELS),
                "input_size": INPUT_SIZE,
                "normalization": {"mean": list(MEAN), "std": list(STD)},
                "architecture": selected["architecture"],
                "temperature": temperature,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    comparison = {
        "data_scope_for_model_selection": "training and validation only",
        "test_accessed_before_final_step": False,
        "selected_variant": selected["name"],
        "selected_architecture": selected["architecture"],
        "selected_validation_metrics": selected["validation"],
        "training_hyperparameters": {
            "label_smoothing": 0.1,
            "weight_decay": 1e-4,
            "head_learning_rate": 1e-3,
            "fine_tune_learning_rate": 2e-5,
            "fine_tune_unfrozen_stage": "last feature stage only (or ResNet layer4)",
            "early_stopping_patience": 2,
        },
        "variants": [
            {
                key: result[key]
                for key in (
                    "name",
                    "architecture",
                    "validation",
                    "training_seconds",
                    "best_validation_loss",
                    "synthetic",
                )
            }
            for result in variants
        ],
        "calibration": calibration,
        "synthetic": synthetic_info,
    }
    comparison_path = REPORT_ROOT / "model_v2_validation_comparison.json"
    comparison_path.write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    final_test = final_test_once(final_path, temperature, device)
    comparison["final_test_once"] = final_test
    comparison_path.write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    write_model_card(
        selected,
        temperature,
        calibration,
        variants,
        synthetic_info,
        final_test,
        {
            "train": {
                "patches": len(train_rows),
                "parents": len(train_parents),
            },
            "validation": {
                "patches": len(validation_rows),
                "parents": len(validation_parents),
            },
        },
        MODEL_ROOT / "model_card_v2.md",
    )
    LOGGER.info("Selected validation variant: %s", selected["name"])
    LOGGER.info("Validation calibration: %s", json.dumps(calibration, sort_keys=True))
    LOGGER.info("Final one-time test result: %s", json.dumps(final_test, sort_keys=True))
    LOGGER.info("Wrote model=%s labels=%s", final_path, labels_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--head-epochs", type=int, default=3)
    parser.add_argument("--fine-tune-epochs", type=int, default=4)
    parser.add_argument(
        "--skip-synthetic",
        action="store_true",
        help="Compare architectures without running the synthetic-dust variant.",
    )
    return parser.parse_args()


def main() -> None:
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(
                REPORT_ROOT / "model_v2_training_console.log",
                encoding="utf-8",
            ),
        ],
    )
    arguments = parse_args()
    if arguments.head_epochs < 1 or arguments.fine_tune_epochs < 0:
        raise ValueError("--head-epochs must be positive and --fine-tune-epochs nonnegative")
    run_experiments(
        arguments.head_epochs,
        arguments.fine_tune_epochs,
        arguments.skip_synthetic,
    )


if __name__ == "__main__":
    main()

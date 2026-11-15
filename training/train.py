"""Train the primary clean-versus-dusty MobileNetV2 classifier."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import random
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageOps
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import models, transforms
from torchvision.models import MobileNet_V2_Weights
from torchvision.transforms import InterpolationMode


ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
SPLIT_ROOT = ROOT / "splits" / "main"
LOG_ROOT = ROOT / "logs"
CHECKPOINT_ROOT = ROOT / "checkpoints"
MODEL_ROOT = PROJECT_ROOT / "model"
LABELS = ("clean", "dusty")
LABEL_TO_INDEX = {label: index for index, label in enumerate(LABELS)}
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
INPUT_SIZE = 224
SEED = 42
LOGGER = logging.getLogger("train")


@dataclass(frozen=True)
class ImageRow:
    patch_path: Path
    label: int
    parent_path: str
    group_id: str
    source: str


class PatchDataset(Dataset[tuple[torch.Tensor, int]]):
    def __init__(self, rows: list[ImageRow], augmentation: bool) -> None:
        self.rows = rows
        self.transform = build_transform(augmentation)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        row = self.rows[index]
        with Image.open(row.patch_path) as opened:
            image = ImageOps.exif_transpose(opened).convert("RGB")
        return self.transform(image), row.label


def build_transform(augmentation: bool) -> transforms.Compose:
    operations: list[Any] = []
    if augmentation:
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
                    brightness=0.25,
                    contrast=0.25,
                    saturation=0.20,
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
    operations.extend([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
    return transforms.Compose(operations)


def read_split(name: str) -> list[ImageRow]:
    path = SPLIT_ROOT / f"{name}.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Main-model split is missing: {path}")
    rows: list[ImageRow] = []
    with path.open(newline="", encoding="utf-8") as stream:
        for item in csv.DictReader(stream):
            if item["label"] not in LABEL_TO_INDEX:
                raise ValueError(f"Unexpected main-model label in {path}: {item['label']}")
            patch_path = PROJECT_ROOT / "training" / item["patch_path"]
            if not patch_path.is_file():
                raise FileNotFoundError(f"Split patch is missing: {patch_path}")
            rows.append(
                ImageRow(
                    patch_path=patch_path,
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


def make_model(pretrained: bool) -> nn.Module:
    weights = MobileNet_V2_Weights.DEFAULT if pretrained else None
    model = models.mobilenet_v2(weights=weights)
    input_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(input_features, len(LABELS))
    return model


def make_loader(
    rows: list[ImageRow],
    augmentation: bool,
    batch_size: int,
    shuffle: bool,
) -> DataLoader[tuple[torch.Tensor, int]]:
    dataset = PatchDataset(rows, augmentation)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=False,
    )


def class_weights(rows: list[ImageRow], device: torch.device) -> torch.Tensor:
    parent_label: dict[str, int] = {}
    for row in rows:
        parent_label[row.parent_path] = row.label
    counts = Counter(parent_label.values())
    weights = torch.tensor(
        [1.0 / max(counts[index], 1) for index in range(len(LABELS))],
        dtype=torch.float32,
        device=device,
    )
    return weights / weights.mean()


def configure_phase(model: nn.Module, phase: int) -> list[nn.Parameter]:
    for parameter in model.parameters():
        parameter.requires_grad = False
    if phase == 1:
        for parameter in model.classifier.parameters():
            parameter.requires_grad = True
    else:
        for layer in model.features[14:]:
            for parameter in layer.parameters():
                parameter.requires_grad = True
        for parameter in model.classifier.parameters():
            parameter.requires_grad = True
    return [parameter for parameter in model.parameters() if parameter.requires_grad]


def run_epoch(
    model: nn.Module,
    loader: DataLoader[tuple[torch.Tensor, int]],
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
) -> tuple[float, float]:
    training = optimizer is not None
    model.train(training)
    if training:
        for module in model.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm) and not any(
                parameter.requires_grad for parameter in module.parameters()
            ):
                module.eval()
    total_loss = 0.0
    total_correct = 0
    total_items = 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            logits = model(images)
            loss = criterion(logits, labels)
            if training:
                loss.backward()
                optimizer.step()
        batch_size = labels.size(0)
        total_loss += float(loss.item()) * batch_size
        total_correct += int((logits.argmax(dim=1) == labels).sum().item())
        total_items += batch_size
    if total_items == 0:
        raise ValueError("Cannot run an epoch with no samples")
    return total_loss / total_items, total_correct / total_items


def smoke_test(device: torch.device) -> float:
    all_rows = read_split("train")
    chosen: list[ImageRow] = []
    for label in range(len(LABELS)):
        label_rows = [row for row in all_rows if row.label == label]
        rng = random.Random(SEED + label)
        rng.shuffle(label_rows)
        chosen.extend(label_rows[:10])
    if len(chosen) != 20:
        raise ValueError(f"Smoke test requires 20 samples, found {len(chosen)}")

    model = make_model(pretrained=True).to(device)
    parameters = configure_phase(model, phase=1)
    optimizer = torch.optim.Adam(parameters, lr=1e-3)
    weights = class_weights(chosen, device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    loader = make_loader(chosen, augmentation=True, batch_size=4, shuffle=True)
    start = time.perf_counter()
    loss, accuracy = run_epoch(model, loader, criterion, device, optimizer)
    duration = time.perf_counter() - start
    LOGGER.info(
        "SMOKE epoch=1 samples=%d loss=%.4f accuracy=%.4f runtime_seconds=%.2f",
        len(chosen),
        loss,
        accuracy,
        duration,
    )
    return duration


def train_phase(
    model: nn.Module,
    phase: int,
    train_rows: list[ImageRow],
    val_rows: list[ImageRow],
    device: torch.device,
    max_epochs: int,
    patience: int,
    best_val_loss: float,
    best_state: dict[str, torch.Tensor] | None,
) -> tuple[float, dict[str, torch.Tensor], list[dict[str, float]]]:
    parameters = configure_phase(model, phase)
    learning_rate = 1e-3 if phase == 1 else 1e-4
    optimizer = torch.optim.Adam(parameters, lr=learning_rate)
    criterion = nn.CrossEntropyLoss(weight=class_weights(train_rows, device))
    train_loader = make_loader(train_rows, augmentation=True, batch_size=32, shuffle=True)
    val_loader = make_loader(val_rows, augmentation=False, batch_size=64, shuffle=False)
    stale_epochs = 0
    history: list[dict[str, float]] = []

    for epoch in range(1, max_epochs + 1):
        started = time.perf_counter()
        train_loss, train_accuracy = run_epoch(
            model, train_loader, criterion, device, optimizer
        )
        val_loss, val_accuracy = run_epoch(
            model, val_loader, criterion, device, optimizer=None
        )
        duration = time.perf_counter() - started
        record = {
            "phase": float(phase),
            "epoch": float(epoch),
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
            "epoch_seconds": duration,
        }
        history.append(record)
        LOGGER.info(
            "phase=%d epoch=%d/%d train_loss=%.4f train_accuracy=%.4f "
            "val_loss=%.4f val_accuracy=%.4f seconds=%.1f",
            phase,
            epoch,
            max_epochs,
            train_loss,
            train_accuracy,
            val_loss,
            val_accuracy,
            duration,
        )
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            stale_epochs = 0
        else:
            stale_epochs += 1
        if stale_epochs >= patience:
            LOGGER.info("phase=%d early_stop epoch=%d", phase, epoch)
            break
    if best_state is None:
        raise RuntimeError("Training did not produce a validation checkpoint")
    return best_val_loss, best_state, history


def configure_logging(log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    LOGGER.setLevel(logging.INFO)
    LOGGER.addHandler(file_handler)
    LOGGER.addHandler(stream_handler)


def train_main(smoke_only: bool, phase1_epochs: int, phase2_epochs: int) -> None:
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_path = LOG_ROOT / f"main_{'smoke' if smoke_only else 'training'}_{timestamp}.log"
    configure_logging(log_path)
    seed_everything(SEED)
    torch.set_num_threads(min(8, torch.get_num_threads()))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    LOGGER.info("device=%s torch=%s torchvision=%s", device, torch.__version__, __import__("torchvision").__version__)

    smoke_duration = smoke_test(device)
    if smoke_only:
        LOGGER.info("Smoke test completed successfully; full training was not started.")
        return

    train_rows = read_split("train")
    val_rows = read_split("val")
    LOGGER.info("train patches=%d val patches=%d", len(train_rows), len(val_rows))
    for split_name, rows in (("train", train_rows), ("val", val_rows)):
        parent_counts = Counter(row.label for row in {row.parent_path: row for row in rows}.values())
        LOGGER.info(
            "%s parent counts clean=%d dusty=%d",
            split_name,
            parent_counts[LABEL_TO_INDEX["clean"]],
            parent_counts[LABEL_TO_INDEX["dusty"]],
        )

    model = make_model(pretrained=True).to(device)
    best_val_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    full_started = time.perf_counter()
    history: list[dict[str, float]] = []
    for phase, epochs in ((1, phase1_epochs), (2, phase2_epochs)):
        best_val_loss, best_state, phase_history = train_phase(
            model,
            phase,
            train_rows,
            val_rows,
            device,
            epochs,
            patience=3,
            best_val_loss=best_val_loss,
            best_state=best_state,
        )
        history.extend(phase_history)
        model.load_state_dict(best_state)

    training_seconds = time.perf_counter() - full_started
    if best_state is None:
        raise RuntimeError("Training failed to save a best model state")
    CHECKPOINT_ROOT.mkdir(parents=True, exist_ok=True)
    checkpoint_path = CHECKPOINT_ROOT / "main_best.pt"
    checkpoint = {
        "state_dict": best_state,
        "labels": list(LABELS),
        "input_size": INPUT_SIZE,
        "mean": list(MEAN),
        "std": list(STD),
        "seed": SEED,
        "best_val_loss": best_val_loss,
        "history": history,
    }
    torch.save(checkpoint, checkpoint_path)
    MODEL_ROOT.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, MODEL_ROOT / "model.pt")
    (MODEL_ROOT / "labels.json").write_text(
        json.dumps(
            {
                "labels": list(LABELS),
                "input_size": INPUT_SIZE,
                "normalization": {"mean": list(MEAN), "std": list(STD)},
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    history_path = LOG_ROOT / "main_history.json"
    history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
    LOGGER.info("smoke_test_seconds=%.2f", smoke_duration)
    LOGGER.info("full_training_seconds=%.2f", training_seconds)
    LOGGER.info("best_val_loss=%.5f checkpoint=%s", best_val_loss, checkpoint_path)
    LOGGER.info("history=%s log=%s", history_path, log_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke-test", action="store_true", help="Run the 20-image smoke test only.")
    parser.add_argument("--phase1-epochs", type=int, default=5)
    parser.add_argument("--phase2-epochs", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.phase1_epochs < 1 or args.phase2_epochs < 1:
        raise ValueError("Epoch limits must be positive")
    train_main(args.smoke_test, args.phase1_epochs, args.phase2_epochs)


if __name__ == "__main__":
    main()

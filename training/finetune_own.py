"""Fine-tune a supported model using session-separated own photos and public replay."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import random
import tempfile
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageOps
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from torchvision.transforms import InterpolationMode


ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
OWN_ROOT = ROOT / "data" / "own"
PUBLIC_SPLIT = ROOT / "splits" / "main" / "train.csv"
MODEL_PATH = PROJECT_ROOT / "model" / "model.pt"
MODEL_LABELS_PATH = PROJECT_ROOT / "model" / "labels.json"
OUTPUT_MODEL_PATH = PROJECT_ROOT / "model" / "own_finetuned.pt"
OUTPUT_REPORT_PATH = ROOT / "reports" / "own_test.json"
IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
SEED = 42
LOGGER = logging.getLogger("finetune_own")


@dataclass(frozen=True)
class Photo:
    path: Path
    label: str
    session: str
    is_own: bool


class OwnPhotoDataset(Dataset[tuple[torch.Tensor, int, float]]):
    def __init__(
        self,
        photos: list[Photo],
        labels: list[str],
        transform: transforms.Compose,
        own_weight: float,
    ) -> None:
        self.photos = photos
        self.label_to_index = {label: index for index, label in enumerate(labels)}
        self.transform = transform
        self.own_weight = own_weight

    def __len__(self) -> int:
        return len(self.photos)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, float]:
        photo = self.photos[index]
        with Image.open(photo.path) as opened:
            image = ImageOps.exif_transpose(opened).convert("RGB")
        sample_weight = self.own_weight if photo.is_own else 1.0
        return (
            self.transform(image),
            self.label_to_index[photo.label],
            sample_weight,
        )


def build_transform(augmentation: bool) -> transforms.Compose:
    operations: list[Any]
    if augmentation:
        operations = [
            transforms.RandomResizedCrop(
                224,
                scale=(0.4, 1.0),
                ratio=(0.9, 1.1),
                interpolation=InterpolationMode.BILINEAR,
            ),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10, interpolation=InterpolationMode.BILINEAR),
            transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.2),
            transforms.RandomApply(
                [transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 1.5))],
                p=0.2,
            ),
        ]
    else:
        operations = [
            transforms.Resize(256, interpolation=InterpolationMode.BILINEAR),
            transforms.CenterCrop(224),
        ]
    operations.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )
    return transforms.Compose(operations)


def collect_own_photos(root: Path, supported_labels: set[str]) -> tuple[list[Photo], list[str]]:
    if not root.is_dir():
        raise FileNotFoundError(f"Own-photo folder is missing: {root}")
    photos: list[Photo] = []
    unsupported_labels: list[str] = []
    for label_directory in sorted(path for path in root.iterdir() if path.is_dir()):
        label = label_directory.name
        if label not in supported_labels:
            if any(path.is_dir() for path in label_directory.iterdir()):
                unsupported_labels.append(label)
            continue
        for session_directory in sorted(
            path for path in label_directory.iterdir() if path.is_dir()
        ):
            for image_path in sorted(session_directory.rglob("*")):
                if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS:
                    photos.append(
                        Photo(
                            path=image_path,
                            label=label,
                            session=session_directory.name,
                            is_own=True,
                        )
                    )
    return photos, unsupported_labels


def split_sessions(
    photos: list[Photo], seed: int = SEED
) -> tuple[list[Photo], list[Photo], dict[str, list[str]]]:
    sessions_by_label: dict[str, set[str]] = defaultdict(set)
    photos_by_label_session: dict[tuple[str, str], list[Photo]] = defaultdict(list)
    for photo in photos:
        sessions_by_label[photo.label].add(photo.session)
        photos_by_label_session[(photo.label, photo.session)].append(photo)
    rng = random.Random(seed)
    train: list[Photo] = []
    own_test: list[Photo] = []
    heldout: dict[str, list[str]] = {}
    for label, sessions_set in sorted(sessions_by_label.items()):
        sessions = sorted(sessions_set)
        if len(sessions) >= 3:
            shuffled = sessions.copy()
            rng.shuffle(shuffled)
            test_count = max(1, round(0.20 * len(shuffled)))
            test_count = min(test_count, len(shuffled) - 1)
            test_sessions = set(shuffled[:test_count])
        elif len(sessions) == 2:
            test_sessions = {sessions[-1]}
        else:
            test_sessions = set()
        heldout[label] = sorted(test_sessions)
        for session in sessions:
            destination = own_test if session in test_sessions else train
            destination.extend(photos_by_label_session[(label, session)])
    return train, own_test, heldout


def make_model(labels: list[str]) -> nn.Module:
    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(labels))
    return model


def load_public_photos(labels: set[str]) -> list[Photo]:
    if not PUBLIC_SPLIT.is_file():
        raise FileNotFoundError(f"Public training split is missing: {PUBLIC_SPLIT}")
    photos: list[Photo] = []
    with PUBLIC_SPLIT.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if row["label"] not in labels:
                continue
            path = PROJECT_ROOT / "training" / row["patch_path"]
            photos.append(
                Photo(path=path, label=row["label"], session=row["group_id"], is_own=False)
            )
    if not photos:
        raise ValueError("No compatible public replay patches are available")
    return photos


def evaluate_own_test(
    model: nn.Module,
    photos: list[Photo],
    labels: list[str],
    device: torch.device,
) -> dict[str, Any]:
    if not photos:
        return {"images": 0, "metrics": {}, "warning": "No own-test session was available."}
    dataset = OwnPhotoDataset(photos, labels, build_transform(False), own_weight=1.0)
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)
    truths: list[int] = []
    predictions: list[int] = []
    with torch.inference_mode():
        for images, target, _weight in loader:
            logits = model(images.to(device))
            truths.extend(target.tolist())
            predictions.extend(logits.argmax(dim=1).cpu().tolist())
    precision, recall, f1, support = precision_recall_fscore_support(
        truths,
        predictions,
        labels=list(range(len(labels))),
        zero_division=0,
    )
    per_class = {
        label: {
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(f1[index]),
            "support": int(support[index]),
        }
        for index, label in enumerate(labels)
    }
    return {
        "images": len(photos),
        "sessions": sorted({photo.session for photo in photos}),
        "accuracy": float(accuracy_score(truths, predictions)),
        "per_class": per_class,
    }


def fine_tune(
    model: nn.Module,
    own_train: list[Photo],
    public_photos: list[Photo],
    labels: list[str],
    own_weight: float,
    epochs: int,
    device: torch.device,
) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = False
    for block in model.features[14:]:
        for parameter in block.parameters():
            parameter.requires_grad = True
    for parameter in model.classifier.parameters():
        parameter.requires_grad = True
    samples = list(own_train)
    # Include public replay with a capped count so own photos remain emphasized.
    replay_rng = random.Random(SEED)
    replay_rng.shuffle(public_photos)
    replay_count = min(len(public_photos), max(1, len(own_train)))
    samples.extend(public_photos[:replay_count])
    dataset = OwnPhotoDataset(samples, labels, build_transform(True), own_weight)
    loader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=0)
    optimizer = torch.optim.Adam(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=1e-5,
    )
    criterion = nn.CrossEntropyLoss(reduction="none")
    model.to(device)
    model.train()
    for epoch in range(1, epochs + 1):
        started = time.perf_counter()
        weighted_loss = 0.0
        total_weight = 0.0
        for images, targets, weights in loader:
            images = images.to(device)
            targets = targets.to(device)
            weights = weights.to(device)
            optimizer.zero_grad(set_to_none=True)
            losses = criterion(model(images), targets)
            loss = (losses * weights).sum() / weights.sum()
            loss.backward()
            optimizer.step()
            weighted_loss += float((losses.detach() * weights).sum().item())
            total_weight += float(weights.sum().item())
        LOGGER.info(
            "epoch=%d/%d weighted_loss=%.4f seconds=%.1f",
            epoch,
            epochs,
            weighted_loss / max(total_weight, 1.0),
            time.perf_counter() - started,
        )


def self_test() -> None:
    with tempfile.TemporaryDirectory(prefix="own_photo_test_", dir=OWN_ROOT) as temporary:
        root = Path(temporary)
        for label in ("clean", "dusty", "cracked", "shaded"):
            for session in ("day_one", "day_two", "day_three"):
                directory = root / label / session
                directory.mkdir(parents=True, exist_ok=True)
                Image.new("RGB", (32, 32), (100, 110, 120)).save(
                    directory / "frame.jpg"
                )
        photos, unsupported = collect_own_photos(root, {"clean", "dusty"})
        train, own_test, heldout = split_sessions(photos)
        assert len(photos) == 6
        assert unsupported == ["cracked", "shaded"]
        assert {(photo.label, photo.session) for photo in train}.isdisjoint(
            (photo.label, photo.session) for photo in own_test
        )
        assert len(heldout["clean"]) == 1 and len(heldout["dusty"]) == 1
        LOGGER.info(
            "SELF_TEST passed photos=%d train=%d own_test=%d heldout_sessions=%s",
            len(photos),
            len(train),
            len(own_test),
            heldout,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--own-weight", type=float, default=3.0)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    arguments = parse_args()
    if arguments.own_weight <= 0 or arguments.epochs < 1:
        raise ValueError("--own-weight and --epochs must be positive")
    OWN_ROOT.mkdir(parents=True, exist_ok=True)
    if arguments.self_test:
        self_test()
        return
    if not MODEL_PATH.is_file() or not MODEL_LABELS_PATH.is_file():
        raise FileNotFoundError("Train and export model/model.pt and model/labels.json first")

    labels_payload = json.loads(MODEL_LABELS_PATH.read_text(encoding="utf-8"))
    labels = labels_payload["labels"]
    own_photos, unsupported = collect_own_photos(OWN_ROOT, set(labels))
    if unsupported:
        LOGGER.warning(
            "Skipping own-photo classes unsupported by this model: %s",
            ", ".join(sorted(unsupported)),
        )
    if not own_photos:
        raise ValueError("No own photos matching model labels were found")
    own_train, own_test, heldout_sessions = split_sessions(own_photos)
    session_counts = {
        label: len({photo.session for photo in own_photos if photo.label == label})
        for label in labels
    }
    if sum(session_counts.values()) < 3 or any(
        count < 3 for count in session_counts.values() if count
    ):
        LOGGER.warning(
            "Fewer than three own-photo sessions for at least one class; "
            "own-test/fine-tuning results are unreliable. Sessions: %s",
            session_counts,
        )
    if not own_train:
        raise ValueError("No own training sessions remain after session holdout")

    checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
    model = make_model(labels)
    model.load_state_dict(checkpoint["state_dict"])
    public_photos = load_public_photos(set(labels))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fine_tune(
        model,
        own_train,
        public_photos,
        labels,
        arguments.own_weight,
        arguments.epochs,
        device,
    )
    own_test_metrics = evaluate_own_test(model, own_test, labels, device)
    OUTPUT_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            **checkpoint,
            "state_dict": model.state_dict(),
            "own_weight": arguments.own_weight,
            "own_test_metrics": own_test_metrics,
            "own_test_sessions": heldout_sessions,
        },
        OUTPUT_MODEL_PATH,
    )
    OUTPUT_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_REPORT_PATH.write_text(
        json.dumps(
            {
                "own_weight": arguments.own_weight,
                "session_counts": session_counts,
                "heldout_sessions": heldout_sessions,
                "own_test": own_test_metrics,
                "note": "Own-test metrics are the only metrics that count for deployment.",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    LOGGER.info("Own-test metrics: %s", json.dumps(own_test_metrics, sort_keys=True))


if __name__ == "__main__":
    main()

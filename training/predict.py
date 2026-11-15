"""Predict clean or dusty for an image with an exported model."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import torch
from PIL import Image, ImageOps
from torch import nn
from torchvision import models, transforms
from torchvision.transforms import InterpolationMode


ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
MODEL_PATH = PROJECT_ROOT / "model" / "model_v2.pt"


def build_transform(input_size: int, mean: list[float], std: list[float]) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize(256, interpolation=InterpolationMode.BILINEAR),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )


def make_model(architecture: str, class_count: int) -> nn.Module:
    if architecture == "mobilenet_v2":
        model = models.mobilenet_v2(weights=None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, class_count)
    elif architecture == "efficientnet_b0":
        model = models.efficientnet_b0(weights=None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, class_count)
    elif architecture == "resnet18":
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, class_count)
    else:
        raise ValueError(f"Unsupported model architecture: {architecture}")
    return model


def labels_path_for(model_path: Path) -> Path:
    suffix = model_path.stem.removeprefix("model")
    return model_path.parent / f"labels{suffix}.json"


def load_image(image_path: Path, use_roi: bool) -> Image.Image:
    if not image_path.is_file():
        raise FileNotFoundError(f"Image file does not exist: {image_path}")
    if not use_roi:
        with Image.open(image_path) as opened:
            return ImageOps.exif_transpose(opened).convert("RGB")
    frame = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if frame is None:
        raise OSError(f"Could not decode image: {image_path}")
    sys.path.insert(0, str(PROJECT_ROOT))
    from app.panel_roi import warp_to_panel

    panel = warp_to_panel(frame)
    return Image.fromarray(cv2.cvtColor(panel, cv2.COLOR_BGR2RGB))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument(
        "--roi",
        action="store_true",
        help="Apply the calibrated fixed-panel crop before classification.",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=MODEL_PATH,
        help="Model checkpoint (default: model/model_v2.pt).",
    )
    arguments = parser.parse_args()
    model_path = arguments.model.resolve()
    labels_path = labels_path_for(model_path)
    if not model_path.is_file() or not labels_path.is_file():
        raise FileNotFoundError(
            f"Model or matching labels file is missing: {model_path}, {labels_path}"
        )
    metadata = json.loads(labels_path.read_text(encoding="utf-8"))
    labels = metadata["labels"]
    normalization = metadata["normalization"]
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
    architecture = checkpoint.get("architecture", metadata.get("architecture", "mobilenet_v2"))
    model = make_model(architecture, len(labels))
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    image = load_image(arguments.image, arguments.roi)
    tensor = build_transform(
        int(metadata["input_size"]),
        normalization["mean"],
        normalization["std"],
    )(image).unsqueeze(0)
    with torch.inference_mode():
        logits = model(tensor)
        temperature = float(checkpoint.get("temperature", metadata.get("temperature", 1.0)))
        probabilities = torch.softmax(logits / temperature, dim=1)[0]
    predicted_index = int(probabilities.argmax().item())
    print(f"{labels[predicted_index]} {probabilities[predicted_index].item():.6f}")


if __name__ == "__main__":
    main()

from pathlib import Path
from typing import Protocol


class InspectionModel(Protocol):
    def predict(self, image_path: str) -> dict: ...


class ModelManager:
    """Optional integration point; never represents the demo simulator as a trained model."""

    def __init__(self, model_path: str = "") -> None:
        self.model_path = model_path
        self.model: InspectionModel | None = None

    @property
    def available(self) -> bool:
        return self.model is not None and Path(self.model_path).exists()

    def load(self, loader) -> None:
        if not self.model_path or not Path(self.model_path).is_file():
            raise FileNotFoundError("Configure a valid trained-model file before loading.")
        self.model = loader(self.model_path)

    def inspect(self, image_path: str) -> dict:
        if not self.available:
            raise RuntimeError("No trained AI model is loaded. Use the clearly labelled Demo AI instead.")
        return self.model.predict(image_path)

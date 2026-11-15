from pathlib import Path
from datetime import datetime, timezone

from PIL import Image

from ai.model_manager import ModelManager
from simulation.ai_simulator import CONDITIONS
from simulation.ai_simulator import inspect as simulate_inspection


class InspectionService:
    def __init__(self, model_manager: ModelManager | None = None) -> None:
        self.model_manager = model_manager or ModelManager()

    def inspect_demo(self, image: Image.Image, condition: str, filename: str) -> dict:
        image.verify()
        result = simulate_inspection(condition, Path(filename).name)
        result["source"] = "Demo AI (simulated)"
        return result

    def inspect_real(self, image_path: str) -> dict:
        result = self.model_manager.inspect(image_path)
        condition = result.get("condition")
        try:
            confidence = float(result["confidence"])
            affected_area = float(result["affected_area"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("AI model result must include numeric confidence and affected_area.") from error
        if condition not in CONDITIONS or not 0 <= confidence <= 100 or not 0 <= affected_area <= 100:
            raise ValueError("AI model returned an invalid condition, confidence, or affected area.")
        return {
            "filename": Path(image_path).name,
            "condition": condition,
            "confidence": confidence,
            "affected_area": affected_area,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "source": "Real AI",
        }

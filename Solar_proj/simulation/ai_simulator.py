import random
from datetime import datetime, timezone


CONDITIONS = ("Normal", "Dust", "Crack", "Partial Shading")


def inspect(condition: str, filename: str = "demo-panel.jpg",
            rng: random.Random | None = None) -> dict:
    if condition not in CONDITIONS:
        raise ValueError(f"Unsupported demo condition: {condition}")
    generator = rng or random.Random()
    area = {
        "Normal": generator.uniform(0, 3),
        "Dust": generator.uniform(25, 55),
        "Crack": generator.uniform(2, 20),
        "Partial Shading": generator.uniform(8, 45),
    }[condition]
    return {
        "filename": filename,
        "condition": condition,
        "confidence": round(generator.uniform(78, 97), 1),
        "affected_area": round(area, 1),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "source": "Demo AI (simulated)",
    }

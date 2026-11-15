import math
import random
import time


class SensorSimulator:
    """Generates explicitly simulated readings within the small panel's rated range."""

    def __init__(self, seed: int | None = None, initial_energy_wh: float = 0.0) -> None:
        self.random = random.Random(seed)
        self.started_at = time.monotonic()
        self.previous_time: float | None = None
        self.previous_power = 0.0
        self.energy_wh = max(0.0, initial_energy_wh)
        self.condition = "Normal"
        self.low_generation = False

    def set_scenario(self, condition: str, low_generation: bool = False) -> None:
        self.condition = condition
        self.low_generation = low_generation

    def reading(self) -> dict[str, float]:
        elapsed_total = time.monotonic() - self.started_at
        daylight = 0.62 + 0.25 * math.sin(elapsed_total / 45)
        voltage = 5.2 + 0.55 * daylight + self.random.uniform(-0.15, 0.15)
        current_a = 0.167 * daylight + self.random.uniform(-0.012, 0.012)
        if self.condition == "Dust":
            current_a *= 0.72
        elif self.condition == "Partial Shading":
            current_a *= 0.48
        elif self.low_generation:
            voltage *= 0.65
            current_a *= 0.22
        voltage = min(6.0, max(0.5, voltage))
        current_a = min(0.167, max(0.005, current_a))
        power = voltage * current_a
        now = time.monotonic()
        if self.previous_time is not None:
            elapsed = max(0, now - self.previous_time)
            self.energy_wh += (self.previous_power + power) / 2 * elapsed / 3600
        self.previous_time = now
        self.previous_power = power
        return {"voltage": voltage, "current": current_a, "power": power,
                "energy_wh": self.energy_wh}

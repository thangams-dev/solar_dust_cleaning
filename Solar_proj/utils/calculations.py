from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class ElectricalReading:
    voltage: float
    current: float
    power: float
    energy_wh: float
    timestamp: datetime


def calculate_power(voltage: float, current_a: float) -> float:
    if voltage < 0 or current_a < 0:
        raise ValueError("Voltage and current must be non-negative.")
    return voltage * current_a


def accumulate_energy(previous_energy_wh: float, previous_power_w: float,
                      current_power_w: float, elapsed_seconds: float) -> float:
    if min(previous_energy_wh, previous_power_w, current_power_w, elapsed_seconds) < 0:
        raise ValueError("Energy, power, and elapsed time must be non-negative.")
    average_power = (previous_power_w + current_power_w) / 2
    return previous_energy_wh + average_power * elapsed_seconds / 3600

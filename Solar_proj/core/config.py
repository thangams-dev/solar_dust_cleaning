from dataclasses import dataclass
import os
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]


@dataclass
class AppConfig:
    database_path: str = os.getenv("SOLAR_DATABASE_PATH", str(ROOT_DIR / "data" / "solar_monitor.db"))
    image_directory: str = os.getenv("SOLAR_IMAGE_DIRECTORY", str(ROOT_DIR / "data" / "images"))
    operating_mode: str = "Demo Mode"
    serial_port: str = ""
    baud_rate: int = 115200
    camera_index: int = 0
    sampling_interval: int = 3
    ina219_address: str = "0x40"
    voltage_limit: float = 6.5
    current_limit: float = 0.25
    power_limit: float = 1.2
    dust_threshold: float = 20.0
    minimum_confidence: float = 60.0
    cleaning_timeout: int = 60
    servo_min_angle: int = 0
    servo_max_angle: int = 180
    auto_cleaning: bool = False


def ensure_directories(config: AppConfig) -> None:
    Path(config.database_path).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
    Path(config.image_directory).expanduser().resolve().mkdir(parents=True, exist_ok=True)

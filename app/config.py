"""Shared paths and image settings for the panel camera pipeline."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
APP_ROOT = Path(__file__).resolve().parent
CONFIG_DIR = APP_ROOT / "config"
ROI_CONFIG_PATH = CONFIG_DIR / "roi.json"
CLEAN_REFERENCE_PATH = CONFIG_DIR / "clean_reference.png"
HEAVY_DUST_REFERENCE_PATH = CONFIG_DIR / "heavy_dust_reference.png"
PANEL_WIDTH = 480
PANEL_HEIGHT = 400
DUSTY_THRESHOLD = 0.70
PANEL_CORNERS = (
    "top_left",
    "top_right",
    "bottom_right",
    "bottom_left",
)

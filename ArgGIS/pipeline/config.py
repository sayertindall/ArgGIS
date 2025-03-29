# pipeline/config.py

import os
import yaml
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# === PROJECT ROOT ===
ROOT_DIR = Path(__file__).resolve().parents[1]

# Load YAML configuration if available
CONFIG = {}
config_path = ROOT_DIR / "config.yaml"
if config_path.exists():
    with open(config_path, "r") as f:
        CONFIG = yaml.safe_load(f)

# === DATA DIRECTORIES ===
# Use values from YAML if available, otherwise use defaults
DATA_RAW = Path(CONFIG.get("paths", {}).get("data_raw", str(ROOT_DIR / "data" / "raw")))
DATA_INTERIM = Path(CONFIG.get("paths", {}).get("data_interim", str(ROOT_DIR / "data" / "interim")))
DATA_PROCESSED = Path(CONFIG.get("paths", {}).get("data_processed", str(ROOT_DIR / "data" / "processed")))
OUTPUTS_DIR = Path(CONFIG.get("paths", {}).get("outputs", str(ROOT_DIR / "outputs")))

# === METADATA ===
METADATA_DIR = Path(CONFIG.get("paths", {}).get("metadata", str(ROOT_DIR / "metadata")))

# === MAP OUTPUTS ===
MAPS_DIR = OUTPUTS_DIR / "maps"
INTERACTIVE_DIR = OUTPUTS_DIR / "interactive"
TABLES_DIR = OUTPUTS_DIR / "tables"

# === COMMON CRS DEFINITIONS ===
CRS_WGS84 = CONFIG.get("gis", {}).get("crs_wgs84", "EPSG:4326")
CRS_PROJECTED = CONFIG.get("gis", {}).get("crs_projected", "EPSG:3857")  # Web Mercator

# === ENVIRONMENT VARIABLES ===
MAPBOX_TOKEN = os.getenv("MAPBOX_TOKEN")
DB_URL = os.getenv("DB_URL")

# === TRANSFORM CONFIGS ===
TRANSFORM_CONFIG = CONFIG.get("transforms", {})
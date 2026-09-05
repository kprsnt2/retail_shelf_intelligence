"""System Configuration for On-Prem Retail Shelf Intelligence."""
from pathlib import Path
from pydantic import BaseModel
import os

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data"
SAMPLE_DIR = DATA_DIR / "sample_shelves"
CATALOG_PATH = DATA_DIR / "store_catalog.json"
PLANOGRAMS_PATH = DATA_DIR / "planograms.json"

class SystemConfig(BaseModel):
    app_name: str = "Deepwork Retail Intelligence"
    version: str = "0.1.0"
    on_prem_mode: bool = True
    
    # Vision pipeline parameters
    min_facing_confidence: float = 0.60
    min_oos_void_width_px: int = 40
    row_segmentation_bands: int = 4  # Top, Eye-Level, Reach, Bottom
    
    # Financial modeling defaults
    default_stockout_penalty_multiplier: float = 1.25  # lost customer goodwill factor
    eye_level_premium_multiplier: float = 1.35  # Eye-level displays generate ~35% higher lift
    
    # Optional Cloud VLM fallback (disabled by default for on-prem data sovereignty)
    enable_cloud_vlm_fallback: bool = False
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")

config = SystemConfig()

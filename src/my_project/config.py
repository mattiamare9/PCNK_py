# Configuration for my_project
from pathlib import Path

# Project root (two levels above this file)
ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "datasets"

# Output directory for figures (default)
OUTPUT_DIR = ROOT / "OUT_SPM"

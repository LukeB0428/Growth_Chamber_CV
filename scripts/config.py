"""
config.py — Central path configuration for Growth Chamber CV pipeline
EE496 | Luke Buckley | Maynooth University

All scripts import paths from here. Works on both Windows (laptop) and
Linux (Raspberry Pi) without any changes.

Usage in any script:
    from config import BASE_DIR, IMAGES_DIR, RESULTS_DIR, CALIB_DIR, SCRIPTS_DIR
"""

import sys
from pathlib import Path

# ── Project root ──────────────────────────────────────────────────────────────
# This file lives in scripts/, so parent = project root.
BASE_DIR    = Path(__file__).resolve().parent.parent

# ── Key directories ───────────────────────────────────────────────────────────
SCRIPTS_DIR = BASE_DIR / "scripts"
IMAGES_DIR  = BASE_DIR / "images"
RESULTS_DIR = BASE_DIR / "results"
CALIB_DIR   = BASE_DIR / "calibration"
PLOTS_DIR   = RESULTS_DIR / "plots"

# ── Results files ─────────────────────────────────────────────────────────────
METRICS_CSV       = RESULTS_DIR / "metrics.csv"
POT_METRICS_CSV   = RESULTS_DIR / "pot_metrics.csv"
GROUND_TRUTH_CSV  = RESULTS_DIR / "ground_truth.csv"
SCHEDULER_LOG     = RESULTS_DIR / "scheduler_log.txt"

# ── Visualisation output dirs ─────────────────────────────────────────────────
HEALTH_VIS_DIR   = RESULTS_DIR / "health_visualisations"
BOLTING_VIS_DIR  = RESULTS_DIR / "bolting_visualisations"
LEAF_VIS_DIR     = RESULTS_DIR / "leaf_visualisations"
LI600_ANNOT_DIR  = RESULTS_DIR / "li600_annotations"

# ── Model weights ─────────────────────────────────────────────────────────────
MODEL_PATH       = SCRIPTS_DIR / "best_model.pth"
SAM2_WEIGHTS_DIR = SCRIPTS_DIR / "sam2_weights"

# ── Training dataset ──────────────────────────────────────────────────────────
DATASET_PATH     = BASE_DIR / "A1"

# ── Python interpreter (venv) — platform-aware ───────────────────────────────
if sys.platform == "win32":
    PYTHON_BIN = SCRIPTS_DIR / ".venv" / "Scripts" / "python.exe"
else:
    PYTHON_BIN = SCRIPTS_DIR / ".venv" / "bin" / "python"

# ── Species configuration ─────────────────────────────────────────────────────
SPECIES_CONFIG_DIR = BASE_DIR / "config" / "species"

# ── Entry-point scripts (used by scheduler and dashboard) ────────────────────
CAPTURE_SCRIPT  = SCRIPTS_DIR / "capture_image.py"
ANALYSE_SCRIPT  = SCRIPTS_DIR / "analyse_chamber.py"

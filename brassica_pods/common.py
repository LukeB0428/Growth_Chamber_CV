"""
common.py — Shared infrastructure for the brassica_pods module.

This is the module's bridge to project-wide resources. It deliberately REUSES
the existing pipeline's utilities (SAM2 weights, project root) by *reference*
rather than refactoring scripts/ into a literal /common package — the scripts/
pipeline is thesis-critical and deployed on the Pi, so it stays untouched.

When the brassica work matures, the path helpers and PixelScale here are the
natural seed for a promoted top-level /common library.

Provides:
  - Module path constants (data dirs, weights, dataset config).
  - PixelScale: the px -> mm conversion every capture session must record,
    or all size metrics are meaningless (see plan §7).
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

# ── Locate the project root and reuse existing pipeline paths ────────────────
MODULE_DIR = Path(__file__).resolve().parent           # .../brassica_pods
PROJECT_ROOT = MODULE_DIR.parent                       # repo root
SCRIPTS_DIR = PROJECT_ROOT / "scripts"

# Reuse the SAM2 checkpoint the Arabidopsis pipeline already downloads, so we
# do not pull a second ~150 MB copy. (scripts/leaf_count.py manages it.)
SAM2_WEIGHTS_DIR = SCRIPTS_DIR / "sam2_weights"

# ── Module-local paths ───────────────────────────────────────────────────────
DATA_DIR = MODULE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"                 # raw rig captures
LABELS_DIR = DATA_DIR / "labels"           # hand-labelled masks (SAM2-assisted)
POD_CUTOUTS_DIR = DATA_DIR / "pod_cutouts"  # synth: individual pod crops
POD_MASKS_DIR = DATA_DIR / "pod_masks"      # synth: matching <name>_mask.png
BACKGROUNDS_DIR = DATA_DIR / "backgrounds"  # synth: empty-rig backgrounds
SYNTH_DIR = DATA_DIR / "synthetic"          # synth output (images/ + labels/)
DATASET_DIR = DATA_DIR / "dataset"          # final YOLO-seg train/val split

WEIGHTS_DIR = MODULE_DIR / "weights"        # trained YOLO-seg checkpoints
DATASET_YAML = MODULE_DIR / "train" / "pods.yaml"

# ── External reference dataset (deepcanola, B. napus, CC-BY-4.0) ──────────────
# Kept OUTSIDE the OneDrive-synced repo so ~28 GB does not sync to the cloud and
# is never git-committed. Override with the BRASSICA_DATA_DIR env var.
#   Source: Atkins et al., Zenodo DOI 10.5281/zenodo.13903900 (CC-BY-4.0).
EXTERNAL_DATA_DIR = Path(
    os.environ.get("BRASSICA_DATA_DIR", Path.home() / "brassica_data"))
DEEPCANOLA_DIR = EXTERNAL_DATA_DIR / "deepcanola"

# Single foreground class for the segmenter.
CLASS_NAMES = ["pod"]


# ── px -> mm scale + scripts path ────────────────────────────────────────────
# Promoted to the shared `phenotyping` package (the README flagged common.py as
# its seed). Re-exported here so existing
# `from brassica_pods.common import PixelScale` / `ensure_scripts_importable`
# imports keep working unchanged.
from phenotyping.scale import PixelScale                 # noqa: E402,F401
from phenotyping.paths import ensure_scripts_importable  # noqa: E402,F401

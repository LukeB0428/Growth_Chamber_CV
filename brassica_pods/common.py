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


# ── px -> mm scale ───────────────────────────────────────────────────────────
@dataclass
class PixelScale:
    """
    Pixels-per-millimetre conversion for one capture session.

    Plan §7: "px->mm scale must be fixed and captured every session or size
    metrics are meaningless." Derive `px_per_mm` from a fiducial/ruler of known
    length in the frame, persist it next to the image, and load it at analysis
    time. Never hard-code a scale — working distance drift invalidates it.
    """

    px_per_mm: float
    source: str = ""          # e.g. "ruler 50mm = 412px" or "ArUco 30mm tag"

    @property
    def mm_per_px(self) -> float:
        return 1.0 / self.px_per_mm

    def length_mm(self, length_px: float) -> float:
        return length_px * self.mm_per_px

    def area_mm2(self, area_px: float) -> float:
        return area_px * (self.mm_per_px ** 2)

    @classmethod
    def from_fiducial(cls, fiducial_px: float, fiducial_mm: float,
                      source: str = "") -> "PixelScale":
        """Build a scale from a measured fiducial: known length in px and mm."""
        if fiducial_px <= 0 or fiducial_mm <= 0:
            raise ValueError("fiducial_px and fiducial_mm must be positive")
        return cls(px_per_mm=fiducial_px / fiducial_mm,
                   source=source or f"{fiducial_mm}mm = {fiducial_px}px")

    def to_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(
            {"px_per_mm": self.px_per_mm, "source": self.source}, indent=2))

    @classmethod
    def from_json(cls, path: str | Path) -> "PixelScale":
        d = json.loads(Path(path).read_text())
        return cls(px_per_mm=float(d["px_per_mm"]), source=d.get("source", ""))


def ensure_scripts_importable() -> None:
    """Add scripts/ to sys.path so we can reuse its helpers without copying."""
    p = str(SCRIPTS_DIR)
    if p not in sys.path:
        sys.path.insert(0, p)

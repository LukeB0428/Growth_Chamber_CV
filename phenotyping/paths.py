"""Project paths + a helper to reuse scripts/ utilities without copying them."""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent     # repo root
SCRIPTS_DIR = PROJECT_ROOT / "scripts"


def ensure_scripts_importable() -> None:
    """Add scripts/ to sys.path so shared code can reuse its helpers (e.g.
    greenness_metrics, species_config) without modifying the thesis pipeline."""
    p = str(SCRIPTS_DIR)
    if p not in sys.path:
        sys.path.insert(0, p)

"""
species_config.py — Per-species parameter configuration for Growth Chamber CV
Luke Buckley | Maynooth University

Loads species-specific thresholds (segmentation, bolting, health, greenness)
from JSON files in config/species/. All scripts that have species-specific
constants expose a configure(cfg) function that replaces their module-level
defaults with values from the loaded config.

Supported species (add a new JSON to config/species/ to extend):
    arabidopsis  — Arabidopsis thaliana (default, calibrated on EE496 trial)
    barley       — Hordeum vulgare (planned Q4 2026)
    brassica     — Brassica napus / rapa (planned Q1 2027)

Usage (from analyse_chamber.py or any orchestrator):
    from species_config import load_and_apply

    load_and_apply('arabidopsis')   # loads JSON and configures all sub-modules
"""

import json
from config import SPECIES_CONFIG_DIR

DEFAULT_SPECIES = "arabidopsis"


def load_species_config(species_name=DEFAULT_SPECIES):
    """
    Load and return the species config dict from config/species/{species}.json.
    Raises FileNotFoundError with a helpful message if the species is unknown.
    """
    path = SPECIES_CONFIG_DIR / f"{species_name}.json"
    if not path.is_file():
        available = [p.stem for p in SPECIES_CONFIG_DIR.glob("*.json")]
        raise FileNotFoundError(
            f"No config found for species '{species_name}'. "
            f"Available: {available}. "
            f"Add {path} to support a new species."
        )
    with open(path) as f:
        cfg = json.load(f)
    return cfg


def load_and_apply(species_name=DEFAULT_SPECIES):
    """
    Load the species config and call configure(cfg) on every sub-module that
    has species-specific constants. Safe to call multiple times.

    Args:
        species_name : str — must match a filename in config/species/

    Returns:
        cfg : dict — the loaded species config (for inspection/logging)
    """
    cfg = load_species_config(species_name)

    import analyse_image
    import health_score
    import bolting_detection
    import greenness_metrics
    import inflorescence_mask

    analyse_image.configure(cfg)
    health_score.configure(cfg)
    bolting_detection.configure(cfg)
    greenness_metrics.configure(cfg)
    inflorescence_mask.configure(cfg)

    print(f"  [species] Configured for {cfg.get('display_name', species_name)}")
    return cfg


def list_available_species():
    """Return list of species names with available config files."""
    return [p.stem for p in SPECIES_CONFIG_DIR.glob("*.json")]

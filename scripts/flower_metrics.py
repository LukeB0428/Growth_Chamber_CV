"""
flower_metrics.py — Brassica yellow-flower detection + flowering metrics.

Brassica (canola/rapeseed) flowers bright yellow; flowering TIME and INTENSITY
are key live-plant phenotypes. This wires the flower config in
config/species/brassica.json (yellow HSV band + NDYI) into a working detector.

Two-signal detection (precise):
  - HSV yellow band   (hue ~40-70°, high S/V)  — primary discriminator vs green
  - NDYI = (G-B)/(G+B) above threshold         — confirms "yellowness"
combined (AND), morphologically cleaned, small components removed.

Metrics per image:
  flower_area_pct      — yellow flower pixels as % of the image
  flowering_fraction   — flower pixels / (plant canopy + flowers)  (needs plant_mask)
  flower_clusters      — connected flower components (proxy for # inflorescences)
  is_flowering         — flower_area_pct >= onset threshold

Onset over a time series = first day is_flowering becomes True (see flowering_onset).

configure(cfg) loads bounds from a species config; defaults are brassica.json's.

Usage:
    from flower_metrics import configure, detect_flowers
    configure(load_brassica_cfg())             # or pass the dict
    out = detect_flowers(bgr, plant_mask=green_mask)
    print(out["flower_area_pct"], out["is_flowering"])
"""

from __future__ import annotations

import cv2
import numpy as np

# ── Defaults (brassica.json: inflorescence.flower + developmental_stage) ─────
H_MIN, H_MAX = 20, 35          # OpenCV hue (0-180) -> ~40-70 degrees = yellow
S_MIN, S_MAX = 100, 255
V_MIN, V_MAX = 80, 255
NDYI_THRESHOLD = 0.05
MIN_COMPONENT_AREA = 50
ONSET_PCT = 2.0                # flowering_yellow_pixel_pct (developmental_stage)
MORPH_K = 5


def configure(cfg: dict) -> None:
    """Configure from a species config dict. Accepts either the full species
    config (uses cfg['inflorescence']['flower'] + developmental_stage thresholds)
    or just the flower sub-dict."""
    global H_MIN, H_MAX, S_MIN, S_MAX, V_MIN, V_MAX
    global NDYI_THRESHOLD, MIN_COMPONENT_AREA, ONSET_PCT
    flower = cfg.get("inflorescence", {}).get("flower", cfg.get("flower", cfg))
    H_MIN = flower.get("h_min", H_MIN);  H_MAX = flower.get("h_max", H_MAX)
    S_MIN = flower.get("s_min", S_MIN);  S_MAX = flower.get("s_max", S_MAX)
    V_MIN = flower.get("v_min", V_MIN);  V_MAX = flower.get("v_max", V_MAX)
    NDYI_THRESHOLD = flower.get("ndyi_threshold", NDYI_THRESHOLD)
    MIN_COMPONENT_AREA = flower.get("min_component_area", MIN_COMPONENT_AREA)
    th = cfg.get("developmental_stage", {}).get("thresholds", {})
    ONSET_PCT = th.get("flowering_yellow_pixel_pct", ONSET_PCT)


def flower_mask(bgr: np.ndarray) -> np.ndarray:
    """Binary yellow-flower mask (uint8 0/255): HSV-yellow AND high-NDYI."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    yellow = cv2.inRange(hsv, np.array([H_MIN, S_MIN, V_MIN], np.uint8),
                         np.array([H_MAX, S_MAX, V_MAX], np.uint8))

    b, g, r = (bgr[:, :, i].astype(np.float32) for i in range(3))
    ndyi = (g - b) / (g + b + 1e-6)
    ndyi_mask = (ndyi > NDYI_THRESHOLD).astype(np.uint8) * 255

    m = cv2.bitwise_and(yellow, ndyi_mask)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (MORPH_K, MORPH_K))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k)

    # Drop tiny noise components.
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    clean = np.zeros_like(m)
    clusters = 0
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= MIN_COMPONENT_AREA:
            clean[lab == i] = 255
            clusters += 1
    return clean, clusters


def detect_flowers(bgr: np.ndarray, plant_mask: np.ndarray | None = None) -> dict:
    """Detect yellow flowers and return flowering metrics + the mask."""
    fmask, clusters = flower_mask(bgr)
    flower_px = int((fmask > 0).sum())
    total_px = fmask.shape[0] * fmask.shape[1]
    area_pct = 100.0 * flower_px / total_px

    flowering_fraction = None
    if plant_mask is not None:
        plant = (plant_mask > 0) | (fmask > 0)      # canopy excludes yellow, add it back
        pp = int(plant.sum())
        flowering_fraction = round(100.0 * flower_px / pp, 2) if pp else 0.0

    return {
        "flower_mask": fmask,
        "flower_pixels": flower_px,
        "flower_area_pct": round(area_pct, 3),
        "flowering_fraction": flowering_fraction,
        "flower_clusters": clusters,
        "is_flowering": area_pct >= ONSET_PCT,
    }


def flowering_onset(series: list[tuple[str, bool]]) -> str | None:
    """First date where is_flowering is True, given [(date, is_flowering), ...]
    sorted by date. Returns the onset date string or None."""
    for date, flowering in series:
        if flowering:
            return date
    return None


def overlay(bgr: np.ndarray, fmask: np.ndarray) -> np.ndarray:
    """Magenta overlay of detected flowers for visual inspection."""
    vis = bgr.copy()
    vis[fmask > 0] = (255, 0, 255)
    return cv2.addWeighted(bgr, 0.5, vis, 0.5, 0)

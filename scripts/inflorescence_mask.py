"""
inflorescence_mask.py — Bolt stem / inflorescence detection for Growth Chamber CV
Luke Buckley | Maynooth University

When Arabidopsis plants bolt, the central inflorescence stem appears pale
green to cream in overhead RGB images. Its low chlorophyll concentration
means low HSV saturation — below the S ≥ 40 threshold of the standard
green mask — so it is excluded from canopy coverage calculations.

This causes the apparent canopy DROP seen in Stage 3 data: enriched plants
bolt earlier, their inflorescence tissue is missed, and their canopy cover
is understated precisely when the treatment effect should be largest.

This module recovers those pixels by:
  1. Restricting the search to the inner 45% radius zone around the pot
     centroid (where the bolt emerges — peripheral pale areas are soil/rim)
  2. Detecting pale plant tissue: green-yellow hue, LOW saturation (5–40),
     medium-high brightness (90–255)
  3. Excluding pixels already captured by the standard green mask
  4. Filtering small disconnected components (noise)

The resulting mask is ORed with the standard green mask before all
downstream metric calculations (canopy cover, vegetation indices, etc.)

Usage (as module, called from analyse_chamber.py):
    from inflorescence_mask import get_inflorescence_mask
    inflo_mask = get_inflorescence_mask(pot_image, pot_mask, green_mask, pot)
    combined   = cv2.bitwise_or(green_mask, inflo_mask)
"""

import cv2
import numpy as np

def configure(cfg):
    """Apply species config to this module's inflorescence detection thresholds."""
    global INFLO_H_MIN, INFLO_H_MAX, INFLO_S_MIN, INFLO_S_MAX, INFLO_V_MIN, INFLO_V_MAX
    global CENTRAL_ZONE_FRACTION, MIN_COMPONENT_AREA
    i = cfg.get('inflorescence', {})
    INFLO_H_MIN           = i.get('h_min',                  INFLO_H_MIN)
    INFLO_H_MAX           = i.get('h_max',                  INFLO_H_MAX)
    INFLO_S_MIN           = i.get('s_min',                  INFLO_S_MIN)
    INFLO_S_MAX           = i.get('s_max',                  INFLO_S_MAX)
    INFLO_V_MIN           = i.get('v_min',                  INFLO_V_MIN)
    INFLO_V_MAX           = i.get('v_max',                  INFLO_V_MAX)
    CENTRAL_ZONE_FRACTION = i.get('central_zone_fraction',  CENTRAL_ZONE_FRACTION)
    MIN_COMPONENT_AREA    = i.get('min_component_area',     MIN_COMPONENT_AREA)


# ── Pale tissue HSV thresholds ────────────────────────────────────────────────
# Green mask captures:    H 25-90,  S 40-255, V 40-255
# Inflorescence captures: H 20-130, S  5-40,  V 90-255
# The S range is deliberately BELOW the green mask floor — that is the gap
# this module fills. Hue is broader to catch cream/yellow flower buds.
INFLO_H_MIN = 20
INFLO_H_MAX = 130
INFLO_S_MIN = 5
INFLO_S_MAX = 40
INFLO_V_MIN = 90
INFLO_V_MAX = 255

# Search zone: fraction of pot radius from centre
# Bolt emerges from the rosette centre — peripheral pale regions are soil/rim
CENTRAL_ZONE_FRACTION = 0.45

# Minimum connected component area (pixels) to include
# Prevents isolated noise pixels contributing to canopy metrics
MIN_COMPONENT_AREA = 50

# Morphological kernel for cleanup
_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))


def get_inflorescence_mask(pot_image, pot_mask, existing_green_mask, pot):
    """
    Detect pale bolt stem / inflorescence pixels not captured by the green mask.

    Args:
        pot_image           : BGR image masked to pot circle (background zeroed)
        pot_mask            : uint8 circular binary mask for this pot (255 inside)
        existing_green_mask : uint8 standard HSV green mask — excluded from output
                              to avoid double-counting canopy pixels
        pot                 : dict with keys x, y, r (pot centre and radius in px)

    Returns:
        inflorescence_mask  : uint8 binary mask of additional bolt/inflorescence
                              pixels (0 or 255). Add to existing_green_mask with
                              cv2.bitwise_or() before computing canopy metrics.
    """
    h, w = pot_image.shape[:2]

    # ── 1. Restrict search to central zone ───────────────────────────────────
    central_mask = np.zeros((h, w), dtype=np.uint8)
    central_r    = int(pot['r'] * CENTRAL_ZONE_FRACTION)
    cv2.circle(central_mask, (pot['x'], pot['y']), central_r, 255, -1)
    search_zone = cv2.bitwise_and(pot_mask, central_mask)

    if np.sum(search_zone > 0) == 0:
        return np.zeros((h, w), dtype=np.uint8)

    # ── 2. Detect pale tissue in HSV ─────────────────────────────────────────
    # Avoid converting a fully-zeroed image to HSV (produces artefacts on edges)
    hsv        = cv2.cvtColor(pot_image, cv2.COLOR_BGR2HSV)
    pale_mask  = cv2.inRange(
        hsv,
        np.array([INFLO_H_MIN, INFLO_S_MIN, INFLO_V_MIN], dtype=np.uint8),
        np.array([INFLO_H_MAX, INFLO_S_MAX, INFLO_V_MAX], dtype=np.uint8),
    )

    # ── 3. Restrict to search zone, exclude existing green pixels ────────────
    pale_in_zone = cv2.bitwise_and(pale_mask, search_zone)
    new_pixels   = cv2.bitwise_and(
        pale_in_zone,
        cv2.bitwise_not(existing_green_mask)
    )

    if np.sum(new_pixels > 0) == 0:
        return np.zeros((h, w), dtype=np.uint8)

    # ── 4. Morphological cleanup ──────────────────────────────────────────────
    # Open: remove isolated single-pixel noise
    # Close: fill small gaps in detected stem region
    new_pixels = cv2.morphologyEx(new_pixels, cv2.MORPH_OPEN,  _KERNEL)
    new_pixels = cv2.morphologyEx(new_pixels, cv2.MORPH_CLOSE, _KERNEL)

    # ── 5. Filter by minimum component area ──────────────────────────────────
    n_labels, label_map, stats, _ = cv2.connectedComponentsWithStats(
        new_pixels, connectivity=8
    )
    filtered = np.zeros((h, w), dtype=np.uint8)
    for i in range(1, n_labels):
        if stats[i, cv2.CC_STAT_AREA] >= MIN_COMPONENT_AREA:
            filtered[label_map == i] = 255

    return filtered

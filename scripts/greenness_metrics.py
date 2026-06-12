"""
greenness_metrics.py — Absolute Greenness Metrics for Growth Chamber CV
EE496 | Luke Buckley | Maynooth University

Computes 12 colour metrics over canopy pixels (green mask only), covering
raw channel values, GCC, CIE Lab, a composite greenness score, and a named
shade label. Called by analyse_image.py and analyse_chamber.py after masking.

Usage (as module, called from analyse_image.py / analyse_chamber.py):
    from greenness_metrics import compute_greenness_metrics
    metrics = compute_greenness_metrics(green_mask, bgr_image)

Computes 10 new colour metrics over canopy pixels:
  Raw channels:   mean_hue, mean_saturation, mean_value, mean_r, mean_g, mean_b
  GCC:            gcc (Green Chromatic Coordinate)
  CIE Lab:        lab_L, lab_a, lab_b
  Greenness score: greenness_score (0-100)
  Named shade:    green_shade (string label)

All metrics are computed over the green mask (canopy pixels only).

Thresholds derived from empirical analysis of 9 Arabidopsis images:
  Dark green wildtype:   L* ~91-93, a* ~-7.1 to -7.6
  Mutant (light green):  L* ~100,   a* ~-6.6
  Young/seedling:        L* ~108-114
  Mid green:             L* ~108,   a* ~-8.9

Named shade categories (using L* + a*):
  "deep-green"    L* < 94  AND  a* < -7.0   mature dark wildtype
  "mid-green"     L* < 104 AND  a* < -6.5   healthy mid-stage
  "light-green"   L* < 115                  young tissue / mutant
  "yellow-green"  otherwise                 stressed / very young

Greenness score (0-100):
  Higher = more green. Weighted combination:
    50% from a* (most direct green axis, normalised)
    30% from NGRDI
    20% from GCC
  Score of 100 = maximally green, 0 = least green in expected range.
"""

import cv2
import numpy as np


# ── Thresholds (derived from empirical Lab analysis of real Arabidopsis images) ──

# Named shade: (L*_max, a*_max, label) — checked in order, first match wins
SHADE_RULES = [
    (94,  -7.0,  "deep-green"),    # mature dark wildtype
    (104, -6.5,  "mid-green"),     # healthy mid-stage or labelled plants
    (115,  None, "light-green"),   # young tissue, mutant, seedlings
]
SHADE_DEFAULT = "yellow-green"    # stressed, etiolated, or very young

# Greenness score normalisation bounds (recalibrated on actual trial data, May 2026)
# Overhead growth chamber imagery differs from field: GCC > 0.40, a* < -10 are common.
A_STAR_MIN  = -16.0   # most green — actual range in trial: -15 to -7
A_STAR_MAX  =  -4.0   # least green
NGRDI_MIN   =  0.03   # actual minimum observed in trial
NGRDI_MAX   =  0.20   # actual maximum observed
GCC_MIN     =  0.34
GCC_MAX     =  0.44   # overhead camera GCC consistently > 0.40


def configure(cfg):
    """Apply species config to this module's greenness normalisation bounds."""
    global A_STAR_MIN, A_STAR_MAX, NGRDI_MIN, NGRDI_MAX, GCC_MIN, GCC_MAX
    global SHADE_RULES, SHADE_DEFAULT
    g = cfg.get('greenness', {})
    A_STAR_MIN    = g.get('a_star_min',    A_STAR_MIN)
    A_STAR_MAX    = g.get('a_star_max',    A_STAR_MAX)
    NGRDI_MIN     = g.get('ngrdi_min',     NGRDI_MIN)
    NGRDI_MAX     = g.get('ngrdi_max',     NGRDI_MAX)
    GCC_MIN       = g.get('gcc_min',       GCC_MIN)
    GCC_MAX       = g.get('gcc_max',       GCC_MAX)
    SHADE_DEFAULT = g.get('shade_default', SHADE_DEFAULT)
    if 'shade_rules' in g:
        SHADE_RULES = [(r['l_max'], r['a_max'], r['label']) for r in g['shade_rules']]


def compute_greenness_metrics(green_mask, bgr_image):
    """
    Compute 10 absolute greenness metrics over canopy pixels.

    Args:
        green_mask:  uint8 mask (255 = plant pixel) at same resolution as bgr_image
        bgr_image:   full BGR image as numpy array (uint8)

    Returns:
        dict with keys:
            mean_hue, mean_saturation, mean_value,
            mean_r, mean_g, mean_b,
            gcc,
            lab_L, lab_a, lab_b,
            greenness_score,
            green_shade
        All float except green_shade (str). Returns None values if no plant pixels.
    """
    empty = {
        'mean_hue': None, 'mean_saturation': None, 'mean_value': None,
        'mean_r': None, 'mean_g': None, 'mean_b': None,
        'gcc': None,
        'lab_L': None, 'lab_a': None, 'lab_b': None,
        'greenness_score': None, 'green_shade': None,
    }

    plant = green_mask > 0
    if plant.sum() < 50:
        return empty

    # Resize mask to image size if needed (depth pipeline may have different res)
    if green_mask.shape != bgr_image.shape[:2]:
        green_mask = cv2.resize(green_mask, (bgr_image.shape[1], bgr_image.shape[0]),
                                interpolation=cv2.INTER_NEAREST)
        plant = green_mask > 0

    # ── Raw BGR channels ──────────────────────────────────────────────────────
    b_ch = bgr_image[:, :, 0].astype(float)
    g_ch = bgr_image[:, :, 1].astype(float)
    r_ch = bgr_image[:, :, 2].astype(float)

    R = float(r_ch[plant].mean())
    G = float(g_ch[plant].mean())
    B = float(b_ch[plant].mean())

    # ── HSV channels ──────────────────────────────────────────────────────────
    hsv = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)
    # Convert OpenCV hue (0-180) to standard degrees (0-360)
    mean_hue = float(hsv[:, :, 0][plant].astype(float).mean()) * 2.0
    mean_sat = float(hsv[:, :, 1][plant].astype(float).mean())
    mean_val = float(hsv[:, :, 2][plant].astype(float).mean())

    # ── GCC ───────────────────────────────────────────────────────────────────
    gcc = float(G / (R + G + B + 1e-6))

    # ── CIE L*a*b* ───────────────────────────────────────────────────────────
    # OpenCV stores Lab with L in [0,255], a and b offset by 128
    lab = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2Lab)
    lab_L = float(lab[:, :, 0][plant].astype(float).mean())
    lab_a = float(lab[:, :, 1][plant].astype(float).mean()) - 128.0  # true a*
    lab_b = float(lab[:, :, 2][plant].astype(float).mean()) - 128.0  # true b*

    # ── NGRDI (needed for greenness score) ───────────────────────────────────
    ngrdi = float((G - R) / (G + R + 1e-6))

    # ── Greenness score (0-100) ───────────────────────────────────────────────
    # Normalise each component to [0,1] — higher = more green
    # a* is negative for green, so we invert: more negative = higher score
    a_norm     = np.clip((A_STAR_MAX - lab_a)  / (A_STAR_MAX - A_STAR_MIN), 0, 1)
    ngrdi_norm = np.clip((ngrdi - NGRDI_MIN)   / (NGRDI_MAX  - NGRDI_MIN),  0, 1)
    gcc_norm   = np.clip((gcc   - GCC_MIN)     / (GCC_MAX    - GCC_MIN),    0, 1)

    greenness_score = round(float(0.50 * a_norm + 0.30 * ngrdi_norm + 0.20 * gcc_norm) * 100, 1)

    # ── Named shade ───────────────────────────────────────────────────────────
    green_shade = SHADE_DEFAULT
    for L_max, a_max, label in SHADE_RULES:
        l_ok = lab_L < L_max
        a_ok = (a_max is None) or (lab_a < a_max)
        if l_ok and a_ok:
            green_shade = label
            break

    return {
        'mean_hue':        round(mean_hue, 2),
        'mean_saturation': round(mean_sat, 2),
        'mean_value':      round(mean_val, 2),
        'mean_r':          round(R, 2),
        'mean_g':          round(G, 2),
        'mean_b':          round(B, 2),
        'gcc':             round(gcc, 4),
        'lab_L':           round(lab_L, 2),
        'lab_a':           round(lab_a, 2),
        'lab_b':           round(lab_b, 2),
        'greenness_score': greenness_score,
        'green_shade':     green_shade,
    }

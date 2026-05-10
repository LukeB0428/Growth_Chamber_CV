"""
health_score.py -- Composite plant health score for Growth Chamber CV pipeline
EE496 | Luke Buckley | Maynooth University

Computes a single 0-100 health score and a named health label from six
plant metrics. Higher score = healthier plant.

Weightings (biologically motivated):
    Necrosis %       25%  — most severe, irreversible tissue death
    Chlorosis %      20%  — early stress indicator, reversible
    NGRDI            20%  — photosynthetic capacity proxy
    Curl score       15%  — physical stress response
    Symmetry score   10%  — structural health
    Canopy cover %   10%  — overall plant vigour

Normalisation bounds (derived from Arabidopsis literature + empirical data):
    Each metric is normalised to [0,1] where 1 = healthiest.
    Stress metrics (chlorosis, necrosis, curl) are inverted so 0% = score 1.

Health labels:
    80-100  healthy
    60-79   mild-stress
    40-59   moderate-stress
    0-39    severe-stress
"""

import numpy as np

# ── Normalisation bounds ──────────────────────────────────────────────────────
# Stress metrics: lower is healthier → inverted normalisation
CHLOROSIS_MAX  = 30.0   # % — at this level plant is severely chlorotic
NECROSIS_MAX   = 20.0   # % — at this level plant has severe necrosis
CURL_MAX       = 1.0    # score — maximum expected curl (0=flat, 1=severe)

# Health metrics: higher is healthier → normal normalisation
NGRDI_MIN      = 0.05   # below this = very poor photosynthetic activity
NGRDI_MAX      = 0.20   # above this = peak healthy greenness
SYMMETRY_MIN   = 0.3    # below this = severely asymmetric
SYMMETRY_MAX   = 1.0    # perfect symmetry
COVER_MIN      = 1.0    # % — below this = barely germinated / dying
COVER_MAX      = 60.0   # % — above this = full vigorous canopy

# ── Weightings ────────────────────────────────────────────────────────────────
WEIGHTS = {
    'necrosis':   0.25,
    'chlorosis':  0.20,
    'ngrdi':      0.20,
    'curl':       0.15,
    'symmetry':   0.10,
    'cover':      0.10,
}

# ── Health label thresholds ───────────────────────────────────────────────────
HEALTH_LABELS = [
    (80, "healthy"),
    (60, "mild-stress"),
    (40, "moderate-stress"),
    ( 0, "severe-stress"),
]


def _norm_inverted(value, max_val):
    """Normalise a stress metric (lower = healthier). Returns 0-1."""
    if value is None:
        return 0.5  # neutral if missing
    return float(np.clip(1.0 - (value / max_val), 0.0, 1.0))


def _norm(value, min_val, max_val):
    """Normalise a health metric (higher = healthier). Returns 0-1."""
    if value is None:
        return 0.5  # neutral if missing
    return float(np.clip((value - min_val) / (max_val - min_val + 1e-6), 0.0, 1.0))


def compute_health_score(chlorosis_pct, necrosis_pct, curl_score,
                          symmetry_score, ngrdi_mean, canopy_cover_pct):
    """
    Compute a composite health score and label from six plant metrics.

    Args:
        chlorosis_pct     : float, % of canopy showing yellowing (0-100)
        necrosis_pct      : float, % of canopy showing dead tissue (0-100)
        curl_score        : float, leaf curl score (0=flat, 1=severe curl)
        symmetry_score    : float, rosette symmetry (0=asymmetric, 1=perfect)
        ngrdi_mean        : float, mean NGRDI over canopy pixels
        canopy_cover_pct  : float, % of image/pot occupied by plant

    Returns:
        dict with keys:
            health_score  : float, 0-100 (higher = healthier)
            health_label  : str, one of healthy / mild-stress /
                            moderate-stress / severe-stress
    """
    # Normalise each metric to [0,1] where 1 = healthiest
    n_necrosis  = _norm_inverted(necrosis_pct,    NECROSIS_MAX)
    n_chlorosis = _norm_inverted(chlorosis_pct,   CHLOROSIS_MAX)
    n_curl      = _norm_inverted(curl_score,      CURL_MAX)
    n_ngrdi     = _norm(ngrdi_mean,               NGRDI_MIN,    NGRDI_MAX)
    n_symmetry  = _norm(symmetry_score,           SYMMETRY_MIN, SYMMETRY_MAX)
    n_cover     = _norm(canopy_cover_pct,         COVER_MIN,    COVER_MAX)

    # Weighted sum → 0-100
    raw_score = (
        WEIGHTS['necrosis']  * n_necrosis  +
        WEIGHTS['chlorosis'] * n_chlorosis +
        WEIGHTS['ngrdi']     * n_ngrdi     +
        WEIGHTS['curl']      * n_curl      +
        WEIGHTS['symmetry']  * n_symmetry  +
        WEIGHTS['cover']     * n_cover
    )
    health_score = round(float(raw_score * 100), 1)

    # Named label
    health_label = "severe-stress"
    for threshold, label in HEALTH_LABELS:
        if health_score >= threshold:
            health_label = label
            break

    return {
        'health_score': health_score,
        'health_label': health_label,
    }

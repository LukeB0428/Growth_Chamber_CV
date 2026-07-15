"""
Tests for CV metric math. Currently locks in the per-pot LAI denominator fix:
compute_lai must divide canopy pixels by the ANALYSIS-REGION area, not the whole
frame, or per-pot LAI is understated ~60x.

Runs standalone or under pytest (see test_fault_floor.py header).
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from health_metrics import compute_lai

FRAME_H, FRAME_W = 1080, 1920
POT_R = 104                       # ~34k-px pot circle, matches the rig
POT_AREA = int(np.pi * POT_R ** 2)


def _pot_mask_with_green(green_fraction_of_pot):
    """Full-frame mask with a pot circle, `green_fraction_of_pot` of it set."""
    mask = np.zeros((FRAME_H, FRAME_W), dtype=np.uint8)
    n_green = int(green_fraction_of_pot * POT_AREA)
    # Fill n_green pixels inside a central band (positions don't matter to LAI).
    ys = np.repeat(np.arange(FRAME_H // 2, FRAME_H // 2 + 200), FRAME_W)
    xs = np.tile(np.arange(FRAME_W), 200)
    idx = np.arange(n_green)
    mask[ys[idx], xs[idx]] = 255
    return mask


def test_lai_denominator_fix_is_large():
    # 7% of the pot is canopy. Whole-frame denominator understates LAI ~60x.
    mask = _pot_mask_with_green(0.07)
    lai_frame = compute_lai(mask, depth_map=None)                  # buggy basis
    lai_pot   = compute_lai(mask, depth_map=None, roi_area_px=POT_AREA)  # fixed
    ratio = lai_pot / lai_frame
    assert ratio > 50, f"expected ~60x correction, got {ratio:.1f}x"
    # Sanity: 7% canopy → Beer-Lambert LAI ≈ -ln(0.93)/0.5 ≈ 0.145
    assert abs(lai_pot - 0.1451) < 0.02, lai_pot


def test_lai_monotonic_in_canopy():
    a = compute_lai(_pot_mask_with_green(0.05), roi_area_px=POT_AREA)
    b = compute_lai(_pot_mask_with_green(0.20), roi_area_px=POT_AREA)
    assert b > a


def test_lai_fraction_clamped():
    # A fully-green ROI must not blow up (canopy_fraction clamped to 0.999).
    mask = _pot_mask_with_green(1.0)
    lai = compute_lai(mask, roi_area_px=POT_AREA)
    assert np.isfinite(lai) and lai > 0


def test_lai_zero_canopy_is_zero():
    mask = np.zeros((FRAME_H, FRAME_W), dtype=np.uint8)
    assert compute_lai(mask, roi_area_px=POT_AREA) == 0.0


def _run_standalone():
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS {t.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"  FAIL {t.__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return failed


if __name__ == "__main__":
    sys.exit(1 if _run_standalone() else 0)

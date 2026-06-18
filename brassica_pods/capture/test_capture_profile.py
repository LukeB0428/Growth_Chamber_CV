"""
test_capture_profile.py — tests for the capture compliance gate.

Runnable without pytest:  python -m brassica_pods.capture.test_capture_profile
(also works under pytest). Uses a REAL ArUco marker for genuine detection +
scale; QR and ColorChecker detectors are monkeypatched because they cannot be
synthesised in this environment (no QR encoder; cv2.mcc needs opencv-contrib).
Sharpness/exposure/aruco/depth logic are exercised for real.

Covers the contract's required tests:
  - compliant frame            -> passed is True
  - one frame missing each element -> the exact matching failure code
  - station_multiview, depth=None  -> NO_DEPTH_IN_INSITU
"""

from __future__ import annotations

import contextlib
import numpy as np
import cv2

from brassica_pods.capture import capture_profile as cp

VALID_ID = "EXP1-001-PLANT-000"
MANIFEST = {"EXP1-001": {"plant_id": "EXP1-001", "variety": "Aviron"}}


# ── frame builders ───────────────────────────────────────────────────────────
def _textured(h=800, w=1000, base=120, noise=25):
    """Mid-grey textured canvas: high Laplacian, no exposure clipping."""
    rng = np.random.RandomState(0)
    img = np.full((h, w, 3), base, np.uint8)
    n = rng.randint(-noise, noise, (h, w, 3)).astype(np.int16)
    return np.clip(img.astype(np.int16) + n, 0, 255).astype(np.uint8)


def _with_marker(marker_id=23, side=200):
    """Textured canvas with a real DICT_5X5_100 marker pasted in."""
    img = _textured()
    d = cv2.aruco.getPredefinedDictionary(cp.ARUCO_DICT_ID)
    m = cv2.aruco.generateImageMarker(d, marker_id, side)
    m3 = cv2.cvtColor(m, cv2.COLOR_GRAY2BGR)
    img[50:50 + side, 50:50 + side] = m3
    return img


@contextlib.contextmanager
def _patch(**funcs):
    """Temporarily replace module-level detectors (no pytest needed)."""
    saved = {k: getattr(cp, k) for k in funcs}
    try:
        for k, v in funcs.items():
            setattr(cp, k, v)
        yield
    finally:
        for k, v in saved.items():
            setattr(cp, k, v)


# All-present stubs for the non-synthesisable detectors.
_QR_OK = lambda img: VALID_ID
_CC_OK = lambda img: 24


# ── tests ────────────────────────────────────────────────────────────────────
def test_compliant_frame_passes():
    img = _with_marker()
    with _patch(decode_qr=_QR_OK, detect_colorchecker=_CC_OK):
        r = cp.validate_frame(img, mode="spread")
    assert r.passed is True, r.reasons
    assert r.reasons == []
    assert r.scale is not None and r.scale.px_per_mm > 0
    # sidecar populates every contract block
    sc = cp.build_sidecar(r, {"image": "t.jpg"})
    for k in ("aruco", "pixel_scale", "qr", "colorchecker", "quality",
              "calibration", "compliance"):
        assert k in sc, k
    assert sc["pixel_scale"]["type"] == "scalar"


def test_missing_aruco():
    img = _textured()                     # no marker
    with _patch(decode_qr=_QR_OK, detect_colorchecker=_CC_OK):
        r = cp.validate_frame(img, mode="spread")
    assert cp.MISSING_ARUCO in r.reasons and not r.passed


def test_missing_qr():
    img = _with_marker()
    with _patch(decode_qr=lambda i: "", detect_colorchecker=_CC_OK):
        r = cp.validate_frame(img, mode="spread")
    assert cp.MISSING_QR in r.reasons and not r.passed


def test_invalid_qr_format():
    img = _with_marker()
    with _patch(decode_qr=lambda i: "not-an-id", detect_colorchecker=_CC_OK):
        r = cp.validate_frame(img, mode="spread")
    assert cp.INVALID_QR_FORMAT in r.reasons and not r.passed


def test_missing_colorchecker():
    img = _with_marker()
    with _patch(decode_qr=_QR_OK, detect_colorchecker=lambda i: 0):
        r = cp.validate_frame(img, mode="spread")
    assert cp.MISSING_COLORCHECKER in r.reasons and not r.passed


def test_blurry():
    img = np.full((800, 1000, 3), 120, np.uint8)   # uniform -> ~0 Laplacian
    with _patch(detect_aruco=lambda i, *a, **k: {"marker_id": 1, "side_px": 200.0,
                "px_per_mm": 6.67, "corners": np.zeros((4, 2))},
                decode_qr=_QR_OK, detect_colorchecker=_CC_OK):
        r = cp.validate_frame(img, mode="spread")
    assert cp.BLURRY in r.reasons and not r.passed


def test_bad_exposure():
    rng = np.random.RandomState(1)
    img = np.clip(250 + rng.randint(-8, 8, (800, 1000, 3)), 0, 255).astype(np.uint8)
    with _patch(detect_aruco=lambda i, *a, **k: {"marker_id": 1, "side_px": 200.0,
                "px_per_mm": 6.67, "corners": np.zeros((4, 2))},
                decode_qr=_QR_OK, detect_colorchecker=_CC_OK):
        r = cp.validate_frame(img, mode="spread")
    assert cp.BAD_EXPOSURE in r.reasons and not r.passed


def test_no_depth_in_insitu():
    img = _with_marker()
    with _patch(decode_qr=_QR_OK, detect_colorchecker=_CC_OK):
        r = cp.validate_frame(img, mode="station_multiview", depth=None)
    assert cp.NO_DEPTH_IN_INSITU in r.reasons and not r.passed


def test_insitu_with_depth_builds_depth_scale():
    img = _with_marker()
    depth = np.full((800, 1000), 500.0, np.float32)
    with _patch(decode_qr=_QR_OK, detect_colorchecker=_CC_OK):
        r = cp.validate_frame(img, mode="station_multiview", depth=depth, fx_px=800.0)
    assert cp.NO_DEPTH_IN_INSITU not in r.reasons
    assert isinstance(r.scale, cp.DepthPixelScale)
    assert abs(r.scale.mm_per_px(10, 10) - 500.0 / 800.0) < 1e-6


def test_qr_parsed_fields_in_sidecar():
    img = _with_marker()
    with _patch(decode_qr=_QR_OK, detect_colorchecker=_CC_OK):
        r = cp.validate_frame(img, mode="spread")
    qr = r.details["qr"]
    assert qr["valid"] and qr["batch"] == "EXP1" and qr["plant"] == "001"
    assert qr["organ"] == "PLANT" and qr["organ_number"] == "000"
    assert qr["plant_id"] == "EXP1-001"


def test_lowercase_id_canonicalised():
    img = _with_marker()
    with _patch(decode_qr=lambda i: "exp1-001-pod-002", detect_colorchecker=_CC_OK):
        r = cp.validate_frame(img, mode="spread")
    assert cp.INVALID_QR_FORMAT not in r.reasons
    assert r.details["qr"]["canonical"] == "EXP1-001-POD-002"


def test_semantic_plant_must_be_000():
    img = _with_marker()
    with _patch(decode_qr=lambda i: "EXP1-001-PLANT-001", detect_colorchecker=_CC_OK):
        r = cp.validate_frame(img, mode="spread")
    assert cp.INVALID_QR_FORMAT in r.reasons and not r.passed


def test_semantic_organ_must_be_ge_001():
    img = _with_marker()
    with _patch(decode_qr=lambda i: "EXP1-001-POD-000", detect_colorchecker=_CC_OK):
        r = cp.validate_frame(img, mode="spread")
    assert cp.INVALID_QR_FORMAT in r.reasons and not r.passed


def test_manifest_strict_unknown_rejects():
    img = _with_marker()
    with _patch(decode_qr=lambda i: "EXP1-999-PLANT-000", detect_colorchecker=_CC_OK):
        r = cp.validate_frame(img, mode="spread", manifest=MANIFEST,
                              manifest_mode="strict")
    assert cp.UNKNOWN_SAMPLE in r.reasons and not r.passed


def test_manifest_warn_unknown_passes():
    img = _with_marker()
    with _patch(decode_qr=lambda i: "EXP1-999-PLANT-000", detect_colorchecker=_CC_OK):
        r = cp.validate_frame(img, mode="spread", manifest=MANIFEST,
                              manifest_mode="warn")
    assert cp.UNKNOWN_SAMPLE not in r.reasons and r.passed
    assert r.details["qr"]["manifest_known"] is False


def test_manifest_known_passes_strict():
    img = _with_marker()
    with _patch(decode_qr=_QR_OK, detect_colorchecker=_CC_OK):
        r = cp.validate_frame(img, mode="spread", manifest=MANIFEST,
                              manifest_mode="strict")
    assert r.passed and r.details["qr"]["manifest_known"] is True


def test_marker_size_guard_blocks_unconfirmed():
    assert cp.MARKER_SIZE_CONFIRMED is False          # default ships unconfirmed
    raised = False
    try:
        cp.start_capture_session()
    except cp.SessionNotReady:
        raised = True
    assert raised, "session must refuse to start while unconfirmed"
    with _patch(MARKER_SIZE_CONFIRMED=True):
        assert cp.start_capture_session() is True


def test_per_mode_marker_size_selected():
    img = _with_marker()
    depth = np.full((800, 1000), 500.0, np.float32)
    with _patch(decode_qr=_QR_OK, detect_colorchecker=_CC_OK):
        r_spread = cp.validate_frame(img, mode="spread")
        r_insitu = cp.validate_frame(img, mode="station_multiview",
                                     depth=depth, fx_px=800.0)
    assert r_spread.details["aruco"]["marker_size_mm"] == 30.0
    assert r_insitu.details["aruco"]["marker_size_mm"] == 70.0


def test_warn_mode_is_loud_and_recorded():
    img = _with_marker()
    with _patch(decode_qr=lambda i: "EXP1-999-PLANT-000", detect_colorchecker=_CC_OK):
        r = cp.validate_frame(img, mode="spread", manifest=MANIFEST,
                              manifest_mode="warn")
    assert r.passed and r.warnings                       # passes but warns
    assert "WARNING" in cp.format_compliance(r)          # surfaced in operator view
    assert r.details["qr"]["manifest_mode"] == "warn"    # recorded in sidecar
    assert r.details["compliance"]["manifest_mode"] == "warn"


def test_charuco_board_detects():
    board = cp.charuco_board()
    img = board.generateImage((1000, 1400))              # (w, h), 5x7 -> A4 ratio
    det = cv2.aruco.CharucoDetector(board)
    ch_corners, ch_ids, _, _ = det.detectBoard(img)
    assert ch_ids is not None and len(ch_ids) > 0


def _main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL  {t.__name__}: {e}")
        except Exception as e:
            failed += 1
            print(f"ERROR {t.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return failed


if __name__ == "__main__":
    import sys
    sys.exit(1 if _main() else 0)

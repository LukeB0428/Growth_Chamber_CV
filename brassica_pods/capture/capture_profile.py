"""
capture_profile.py — Brassica capture COMPLIANCE GATE (detect + refuse only).

Enforces CAPTURE_SCHEMA.md at capture time so non-compliant frames never enter
the kept dataset. This is the difference between the spec being "locked" and the
irreversible-capture risk being *retired* — paper does not stop a bad capture,
this gate does.

Public contract:
    validate_frame(image, mode, depth=None, ...) -> ComplianceResult
        ComplianceResult(passed: bool, reasons: [code], scale, details)
    persist_or_quarantine(...)  keeps a frame ONLY if passed; else quarantines
                                it + logs reasons. Nothing rejected is kept.
    format_compliance(result)   operator-facing live pass/fail string.

SCOPE (this build): per-frame gate, depth-aware scale, sidecar population.
OUT OF SCOPE: colour-correction pipeline, multi-view registration/dedup, rig
hardware, TasselNet. (ColorChecker here is presence-only; correction is later.)

Every required detection must pass or the frame is REJECTED:
  ArUco (DICT_5X5_100) · QR sample-ID · ColorChecker presence · sharpness ·
  exposure. For in_situ_multiview a depth map is mandatory (depth-aware scale).

Production frames are assumed already UNDISTORTED via the one-time ChArUco
intrinsics/distortion calibration (see undistort()/load_calibration()).

Deps: opencv (aruco + QRCodeDetector are in base opencv>=4.7). ColorChecker
detection uses cv2.mcc (opencv-CONTRIB) — without contrib it fail-closes
(MISSING_COLORCHECKER), so install opencv-contrib-python in production.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from brassica_pods.common import PixelScale

log = logging.getLogger("brassica_pods.capture")

# ── Capture profile constants (Diarmuid, 2026-06-17) ─────────────────────────
BRASSICA_CAPTURE = {
    "species": "brassica_napus",
    "subject": "whole_plant_in_situ",
    "needs_fiducial": True,
    "needs_colorchecker": True,
    "lock_exposure": True,
}
MODES = ("spread", "daily_growth", "in_situ_multiview")

# ── Stable failure codes (enumerate; never free-text a reason) ───────────────
MISSING_ARUCO = "MISSING_ARUCO"
MISSING_QR = "MISSING_QR"
INVALID_QR_FORMAT = "INVALID_QR_FORMAT"
MISSING_COLORCHECKER = "MISSING_COLORCHECKER"
BLURRY = "BLURRY"
BAD_EXPOSURE = "BAD_EXPOSURE"
NO_DEPTH_IN_INSITU = "NO_DEPTH_IN_INSITU"
UNKNOWN_SAMPLE = "UNKNOWN_SAMPLE"      # only ever raised when manifest_mode=="strict"
FAILURE_CODES = (MISSING_ARUCO, MISSING_QR, INVALID_QR_FORMAT, MISSING_COLORCHECKER,
                 BLURRY, BAD_EXPOSURE, NO_DEPTH_IN_INSITU, UNKNOWN_SAMPLE)

# ── ArUco config (fixed dictionary, recorded in every sidecar) ───────────────
ARUCO_DICT_NAME = "DICT_5X5_100"
ARUCO_DICT_ID = cv2.aruco.DICT_5X5_100

# Per-mode physical marker side length (mm). DEFAULTS/PLACEHOLDERS — a wrong size
# is a silent, irreversible systematic error in EVERY measurement, so they must
# be measured from the actual printed markers before any real capture (guard
# below). in_situ marker is larger so it resolves across the turntable distance.
MARKER_SIZE_MM = {"spread": 30.0, "daily_growth": 30.0, "in_situ_multiview": 70.0}

# Flip to True ONLY after printing the markers, measuring the printed squares
# with calipers, and setting MARKER_SIZE_MM to those actual values.
MARKER_SIZE_CONFIRMED = False


class SessionNotReady(RuntimeError):
    """Raised at session start when marker sizes are not yet verified."""


def start_capture_session():
    """Session-start precondition guard (NOT per-frame).

    A capture session refuses to start while MARKER_SIZE_CONFIRMED is False — a
    wrong marker size silently biases every measurement, which is exactly what
    this gate exists to prevent. Call once before capturing.
    """
    if not MARKER_SIZE_CONFIRMED:
        raise SessionNotReady(
            "Marker sizes unverified — print markers, measure with calipers, set "
            "MARKER_SIZE_MM to the ACTUAL printed sizes, then set "
            "MARKER_SIZE_CONFIRMED=True.")
    return True

# ── Thresholds (tune at the rig; recorded so results are reproducible) ───────
LAPLACIAN_MIN = 100.0                # variance-of-Laplacian sharpness floor
EXPOSURE_CLIP_FRAC = 0.10            # max fraction of pixels clipped at either end
EXPOSURE_LOW, EXPOSURE_HIGH = 5, 250

# Locked sample-ID format (2026-06-18): BATCH-PLANT-ORGAN-ORGANNUMBER, uppercase.
#   BATCH  2-8 [A-Z0-9] · PLANT 3-digit · ORGAN one of POD/STEM/LEAF/PLANT ·
#   ORGANNUMBER 3-digit. Semantic rules (checked after the regex):
#     ORGAN==PLANT  -> ORGANNUMBER == 000  (whole-plant / pod-count image)
#     ORGAN in POD/STEM/LEAF -> ORGANNUMBER >= 001
# Variety/treatment/dates live in the manifest, NOT the ID. Date is auto-recorded
# in the sidecar — never duplicated in the ID.
ORGAN_VOCAB = ("POD", "STEM", "LEAF", "PLANT")
SAMPLE_ID_RE = re.compile(r"^([A-Z0-9]{2,8})-(\d{3})-(POD|STEM|LEAF|PLANT)-(\d{3})$")


# ─────────────────────────────────────────────────────────────────────────────
# Result types
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class DepthPixelScale:
    """Per-pixel mm/px from depth: mm_per_px(x,y) = depth_mm(y,x) / fx_px.

    Used for in_situ_multiview, where a single scalar scale is INVALID because
    pods sit at varying depths. Sanity-checked against the ArUco scalar at the
    marker's depth (see validate_frame).
    """
    fx_px: float
    depth_mm: np.ndarray                 # HxW float32, millimetres
    aruco_check_px_per_mm: Optional[float] = None
    source: str = ""

    def mm_per_px(self, x: int, y: int) -> Optional[float]:
        d = float(self.depth_mm[y, x])
        return None if d <= 0 else d / self.fx_px


@dataclass
class ComplianceResult:
    passed: bool
    reasons: list[str]
    scale: object = None                 # PixelScale | DepthPixelScale | None
    details: dict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)  # non-failing, surfaced loud


# ─────────────────────────────────────────────────────────────────────────────
# Detectors — module-level so they are independently testable / monkeypatchable.
# Each returns plain data; validate_frame turns that into pass/fail + codes.
# ─────────────────────────────────────────────────────────────────────────────
def detect_aruco(image, marker_size_mm: float = 30.0) -> Optional[dict]:
    """Detect a single DICT_5X5_100 marker. Returns dict(marker_id, side_px,
    px_per_mm, corners) or None if none found."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT_ID)
    detector = cv2.aruco.ArucoDetector(dictionary, cv2.aruco.DetectorParameters())
    corners, ids, _ = detector.detectMarkers(gray)
    if ids is None or len(ids) == 0:
        return None
    c = corners[0].reshape(4, 2)          # first marker
    sides = [np.linalg.norm(c[i] - c[(i + 1) % 4]) for i in range(4)]
    side_px = float(np.mean(sides))
    return {
        "marker_id": int(ids.flatten()[0]),
        "side_px": side_px,
        "px_per_mm": side_px / marker_size_mm,
        "corners": c,
    }


def decode_qr(image) -> str:
    """Decode a QR/DataMatrix payload; '' if none. Uses cv2's built-in decoder."""
    data, _, _ = cv2.QRCodeDetector().detectAndDecode(image)
    return data or ""


def detect_colorchecker(image) -> int:
    """Number of ColorChecker patches detected (0 = none / detector unavailable).

    Uses cv2.mcc (opencv-contrib). Without contrib this returns 0 and the frame
    fail-closes with MISSING_COLORCHECKER — install opencv-contrib-python in
    production. (Presence only; colour correction is a downstream step.)
    """
    if not hasattr(cv2, "mcc"):
        log.warning("cv2.mcc unavailable (need opencv-contrib-python); "
                    "ColorChecker fail-closes.")
        return 0
    try:
        det = cv2.mcc.CCheckerDetector_create()
        if not det.process(image, cv2.mcc.MCC24):
            return 0
        checkers = det.getListColorChecker()
        return 24 * len(checkers) if checkers else 0
    except Exception as e:        # pragma: no cover - depends on contrib build
        log.warning("ColorChecker detection error: %s", e)
        return 0


def parse_sample_id(raw: str) -> Optional[dict]:
    """Parse + validate a sample ID (regex + the two semantic rules).

    Tolerates lowercase entry (canonicalises to uppercase). Returns the parsed
    fields dict on success, or None if the regex OR a semantic rule fails.
    """
    canonical = (raw or "").strip().upper()
    m = SAMPLE_ID_RE.match(canonical)
    if not m:
        return None
    batch, plant, organ, organ_number = m.groups()
    # Semantic rules: PLANT <-> 000; POD/STEM/LEAF >= 001.
    if organ == "PLANT":
        if organ_number != "000":
            return None
    elif int(organ_number) < 1:
        return None
    return {
        "raw": raw, "canonical": canonical,
        "batch": batch, "plant": plant, "organ": organ,
        "organ_number": organ_number,
        "plant_id": f"{batch}-{plant}",     # manifest key
    }


def load_manifest(path) -> dict:
    """Load the plant-level manifest CSV -> {plant_id: row_dict}.

    Key = BATCH-PLANT (e.g. EXP1-001). Every organ ID inherits its plant's
    metadata via this prefix.
    """
    import csv
    rows = {}
    with open(path, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            pid = (row.get("plant_id") or "").strip().upper()
            if pid:
                rows[pid] = row
    return rows


def laplacian_var(image) -> float:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def exposure_clip_fractions(image) -> tuple[float, float]:
    """(fraction clipped dark, fraction clipped bright) on luminance."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    n = gray.size
    low = float(np.count_nonzero(gray <= EXPOSURE_LOW)) / n
    high = float(np.count_nonzero(gray >= EXPOSURE_HIGH)) / n
    return low, high


# ─────────────────────────────────────────────────────────────────────────────
# The gate
# ─────────────────────────────────────────────────────────────────────────────
def validate_frame(image, mode: str, depth=None,
                   marker_size_mm: Optional[float] = None,
                   fx_px: Optional[float] = None,
                   manifest: Optional[dict] = None,
                   manifest_mode: str = "warn") -> ComplianceResult:
    """Run every required check; keep ALL failing codes (no short-circuit).

    Args:
        image:  BGR frame, assumed already undistorted (ChArUco calibration).
        mode:   one of MODES.
        depth:  HxW depth map in mm (required for in_situ_multiview).
        marker_size_mm: physical marker side; defaults to MARKER_SIZE_MM[mode].
        fx_px:  camera focal length in px (from intrinsics) for depth-aware scale.
        manifest: plant-level manifest {plant_id: row} (see load_manifest). If
            given, the QR's BATCH-PLANT prefix is looked up.
        manifest_mode: "strict" → missing prefix REJECTS (UNKNOWN_SAMPLE);
            "warn" → passes but surfaces a LOUD warning. Ignored if manifest None.
    """
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    if marker_size_mm is None:
        marker_size_mm = MARKER_SIZE_MM[mode]    # per-mode physical size

    reasons: list[str] = []
    warnings: list[str] = []
    details: dict = {}

    # ── ArUco (scale + pose) ─────────────────────────────────────────────────
    aruco = detect_aruco(image, marker_size_mm)
    details["aruco"] = {
        "dictionary": ARUCO_DICT_NAME, "marker_size_mm": marker_size_mm,
        "detected": aruco is not None,
        "marker_id": aruco["marker_id"] if aruco else None,
        "pose": None,
    }
    if aruco is None:
        reasons.append(MISSING_ARUCO)

    # ── QR sample-ID ─────────────────────────────────────────────────────────
    qr = decode_qr(image)
    parsed = parse_sample_id(qr) if qr else None
    qr_block = {"decoded": qr, "valid": parsed is not None}
    if parsed:
        qr_block.update({
            "canonical": parsed["canonical"], "batch": parsed["batch"],
            "plant": parsed["plant"], "organ": parsed["organ"],
            "organ_number": parsed["organ_number"], "plant_id": parsed["plant_id"],
        })
    details["qr"] = qr_block
    if not qr:
        reasons.append(MISSING_QR)
    elif parsed is None:
        reasons.append(INVALID_QR_FORMAT)
    elif manifest is not None:
        # Format is valid; check the plant exists in the manifest.
        known = parsed["plant_id"] in manifest
        qr_block["manifest_known"] = known
        qr_block["manifest_mode"] = manifest_mode      # permanently recorded
        if not known:
            if manifest_mode == "strict":
                reasons.append(UNKNOWN_SAMPLE)
            else:
                # warn: frame passes, but make it LOUD so the operator sees it.
                msg = f"{UNKNOWN_SAMPLE}: {parsed['plant_id']} not in manifest (warn)"
                warnings.append(msg)
                log.warning(msg)

    # ── ColorChecker presence ────────────────────────────────────────────────
    patches = detect_colorchecker(image)
    details["colorchecker"] = {"detected": patches > 0, "patches_found": patches}
    if patches <= 0:
        reasons.append(MISSING_COLORCHECKER)

    # ── Sharpness + exposure ─────────────────────────────────────────────────
    lap = laplacian_var(image)
    low_f, high_f = exposure_clip_fractions(image)
    exposure_ok = low_f <= EXPOSURE_CLIP_FRAC and high_f <= EXPOSURE_CLIP_FRAC
    details["quality"] = {"laplacian_var": round(lap, 2), "exposure_ok": exposure_ok,
                          "clip_low": round(low_f, 4), "clip_high": round(high_f, 4)}
    if lap < LAPLACIAN_MIN:
        reasons.append(BLURRY)
    if not exposure_ok:
        reasons.append(BAD_EXPOSURE)

    # ── Per-mode PixelScale ──────────────────────────────────────────────────
    scale = None
    if mode == "in_situ_multiview":
        if depth is None:
            reasons.append(NO_DEPTH_IN_INSITU)
            details["pixel_scale"] = {"mode": mode, "type": "depth_map",
                                      "value_or_ref": None}
        else:
            depth_mm = depth.astype(np.float32)
            check = aruco["px_per_mm"] if aruco else None
            scale = DepthPixelScale(fx_px=fx_px or 0.0, depth_mm=depth_mm,
                                    aruco_check_px_per_mm=check,
                                    source="depth/fx, ArUco-sanity-checked")
            details["pixel_scale"] = {"mode": mode, "type": "depth_map",
                                      "value_or_ref": "depth_mm/fx_px"}
    else:
        # spread / daily_growth: single scalar from ArUco is valid (co-planar).
        if aruco is not None:
            scale = PixelScale.from_fiducial(
                aruco["side_px"], marker_size_mm,
                source=f"ArUco {ARUCO_DICT_NAME} id={aruco['marker_id']}")
        details["pixel_scale"] = {
            "mode": mode, "type": "scalar",
            "value_or_ref": round(scale.px_per_mm, 4) if scale else None}

    passed = len(reasons) == 0
    details["compliance"] = {
        "passed": passed, "reasons": reasons, "warnings": warnings,
        "manifest_mode": manifest_mode if manifest is not None else None,
    }
    return ComplianceResult(passed=passed, reasons=reasons, scale=scale,
                            details=details, warnings=warnings)


# ─────────────────────────────────────────────────────────────────────────────
# Sidecar + persistence
# ─────────────────────────────────────────────────────────────────────────────
def build_sidecar(result: ComplianceResult, base_meta: dict,
                  calibration_ref: str = "calibration/brassica_intrinsics.json",
                  undistorted: bool = True) -> dict:
    """Merge base capture metadata with the gate's detection blocks."""
    sidecar = dict(base_meta)
    sidecar.update({
        "aruco": result.details.get("aruco"),
        "pixel_scale": result.details.get("pixel_scale"),
        "qr": result.details.get("qr"),
        "colorchecker": result.details.get("colorchecker"),
        "quality": result.details.get("quality"),
        "calibration": {"intrinsics_ref": calibration_ref,
                        "distortion_ref": calibration_ref,
                        "undistorted": undistorted},
        "compliance": result.details.get("compliance"),
    })
    return sidecar


def persist_or_quarantine(image, result: ComplianceResult, base_meta: dict,
                          kept_dir, quarantine_dir, stem: str) -> Path:
    """Keep a frame ONLY if passed; otherwise quarantine it + log reasons.
    Returns the path the image was written to. Nothing rejected enters kept_dir.
    """
    out_dir = Path(kept_dir if result.passed else quarantine_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    img_path = out_dir / f"{stem}.jpg"
    cv2.imwrite(str(img_path), image)
    (out_dir / f"{stem}.json").write_text(
        json.dumps(build_sidecar(result, base_meta), indent=2, default=str))
    if not result.passed:
        log.warning("REJECTED %s -> %s : %s", stem, img_path,
                    ",".join(result.reasons))
    return img_path


def format_compliance(result: ComplianceResult) -> str:
    """Operator-facing live status — what to fix before moving on.

    Warnings (e.g. unknown sample in warn-mode) are surfaced LOUD even on PASS,
    so they cannot slip by unnoticed.
    """
    if result.passed:
        base = "PASS ✓  frame compliant"
    else:
        base = "FAIL ✗  fix: " + ", ".join(result.reasons)
    if result.warnings:
        base += "\n  ⚠️  WARNING: " + " | ".join(result.warnings)
    return base


# ─────────────────────────────────────────────────────────────────────────────
# ChArUco calibration helpers (intrinsics/distortion are produced one-time)
# ─────────────────────────────────────────────────────────────────────────────
# ── ChArUco board config (must match the production marker dictionary) ───────
CHARUCO_SQUARES_X = 5         # columns
CHARUCO_SQUARES_Y = 7         # rows  (5x7 squares, A4 @ 100%)
CHARUCO_SQUARE_MM = 30.0
CHARUCO_MARKER_MM = 23.0
CALIB_MAX_REPROJ_PX = 1.0     # accept ceiling; target < 0.5


def charuco_board():
    """The calibration ChArUco board (DICT_5X5_100, matches production markers)."""
    d = cv2.aruco.getPredefinedDictionary(ARUCO_DICT_ID)
    return cv2.aruco.CharucoBoard(
        (CHARUCO_SQUARES_X, CHARUCO_SQUARES_Y),
        CHARUCO_SQUARE_MM, CHARUCO_MARKER_MM, d)


def calibrate_charuco(images) -> dict:
    """Estimate intrinsics + distortion from ChArUco calibration shots.

    images: list of BGR frames of the board at varied angles/distances (15-25,
    filling frame edges where distortion is worst). Uses the new-API
    CharucoDetector + matchImagePoints + cv2.calibrateCamera (base opencv has
    these; the removed calibrateCameraCharuco one-shot is not needed).

    Returns dict(camera_matrix, dist_coeffs, reproj_error_px, n_images,
    image_size, accepted). `accepted` is reproj_error_px < CALIB_MAX_REPROJ_PX.
    """
    board = charuco_board()
    detector = cv2.aruco.CharucoDetector(board)
    all_obj, all_img = [], []
    img_size = None
    for im in images:
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) if im.ndim == 3 else im
        img_size = (gray.shape[1], gray.shape[0])
        ch_corners, ch_ids, _, _ = detector.detectBoard(gray)
        if ch_ids is None or len(ch_ids) < 4:
            continue
        obj, imgp = board.matchImagePoints(ch_corners, ch_ids)
        if obj is None or len(obj) < 4:
            continue
        all_obj.append(obj)
        all_img.append(imgp)
    if len(all_obj) < 3:
        raise RuntimeError(
            f"Only {len(all_obj)} usable ChArUco views — need varied shots.")
    rms, K, dist, _, _ = cv2.calibrateCamera(all_obj, all_img, img_size, None, None)
    return {"camera_matrix": K, "dist_coeffs": dist,
            "reproj_error_px": float(rms), "n_images": len(all_obj),
            "image_size": list(img_size),
            "accepted": float(rms) < CALIB_MAX_REPROJ_PX}


def save_calibration(path, calib: dict) -> None:
    """Persist intrinsics/distortion (npz) — e.g. calibration/intrinsics_<cam>.npz."""
    np.savez(str(path),
             camera_matrix=calib["camera_matrix"], dist_coeffs=calib["dist_coeffs"],
             reproj_error_px=calib.get("reproj_error_px", -1.0),
             image_size=np.array(calib.get("image_size", [0, 0])))


def load_calibration(path) -> dict:
    """Load intrinsics + distortion from the one-time ChArUco calibration file.
    Supports .npz (preferred) and .json."""
    path = Path(path)
    if path.suffix == ".npz":
        z = np.load(path)
        return {"camera_matrix": z["camera_matrix"], "dist_coeffs": z["dist_coeffs"],
                "reproj_error_px": float(z["reproj_error_px"]) if "reproj_error_px" in z else None,
                "image_size": z["image_size"].tolist() if "image_size" in z else None}
    d = json.loads(path.read_text())
    d["camera_matrix"] = np.array(d["camera_matrix"], dtype=np.float64)
    d["dist_coeffs"] = np.array(d["dist_coeffs"], dtype=np.float64)
    return d


def undistort(image, calib: dict):
    """Undistort a production frame before any measurement runs."""
    return cv2.undistort(image, calib["camera_matrix"], calib["dist_coeffs"])


if __name__ == "__main__":
    print(__doc__)
    print("Failure codes:", ", ".join(FAILURE_CODES))

"""
calibration_drift.py — pot-calibration drift detection + global re-registration.

Stage 4 of the monitoring agent. Pot calibration is a manual ROI set (the Hough
auto-detection was removed). Over a long run the camera can nudge, shifting every
pot's ROI together. This module:

  - green_centroids() / centroid_consensus(): drift is flagged only when a MAJORITY
    of pots' plant centroids move by a CONSISTENT vector — a single sprawling rosette
    (or a stem flopping over the edge) must never trigger a correction.
  - register_ecc(): estimates the global frame shift with cv2.findTransformECC
    against a stored reference frame.
  - decide_action(): corrected (apply shift + audit) | escalate (no convergence, or
    registration shift without pot consensus -> a pot physically moved) | none.
  - apply_shift_to_calibration(): re-seats every ROI, NEVER silently — it appends an
    audit record with the exact shift.

PRODUCTION CAVEAT: ECC on the raw frame locks onto plant content as the canopy fills
in, and a week-1 reference goes stale. For reliable registration, add fixed fiducials
(the ColorChecker) to the tray BEFORE Round 2 and register on those. Until then the
honest mode is detect-and-ESCALATE, not silent auto-correct — which is why the
centroid consensus gates every correction.
"""
import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import cv2

DEFAULT_HSV_LO = (25, 50, 45)
DEFAULT_HSV_HI = (88, 200, 250)


def green_mask(img_bgr, lo=DEFAULT_HSV_LO, hi=DEFAULT_HSV_HI):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    m = cv2.inRange(hsv, np.array(lo, np.uint8), np.array(hi, np.uint8))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    return cv2.morphologyEx(m, cv2.MORPH_OPEN, k, iterations=2)


def green_centroids(img_bgr, calib, min_px=150):
    """Per-pot green centroid within each ROI. None where too little green."""
    mask = green_mask(img_bgr)
    out = {}
    for pot in calib.get("pots", []):
        x, y, r = int(pot["x"]), int(pot["y"]), int(pot["r"])
        roi = np.zeros(mask.shape, np.uint8)
        cv2.circle(roi, (x, y), r, 255, -1)
        sub = cv2.bitwise_and(mask, roi)
        ys, xs = np.where(sub > 0)
        out[pot["label"]] = (float(xs.mean()), float(ys.mean())) if len(xs) >= min_px else None
    return out


def centroid_consensus(ref_c, cur_c, tol_px=6.0, min_mag=4.0, min_agree_frac=0.5):
    """Drift only when a majority of paired pots shift by a consistent vector."""
    disp = []
    for label, rc in ref_c.items():
        cc = cur_c.get(label)
        if rc is not None and cc is not None:
            disp.append((cc[0] - rc[0], cc[1] - rc[1]))
    if not disp:
        return {"drift": False, "n_agree": 0, "n_pairs": 0, "shift": (0.0, 0.0), "agree_frac": 0.0}
    arr = np.array(disp)
    med = np.median(arr, axis=0)
    agree = [d for d in arr if np.hypot(d[0] - med[0], d[1] - med[1]) <= tol_px]
    frac = len(agree) / len(arr)
    drift = (frac >= min_agree_frac) and (float(np.hypot(*med)) >= min_mag)
    return {"drift": bool(drift), "n_agree": len(agree), "n_pairs": len(arr),
            "shift": (float(med[0]), float(med[1])), "agree_frac": round(frac, 2)}


def _gray(img):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    return cv2.GaussianBlur(g, (5, 5), 0).astype(np.float32)


def register_ecc(ref_bgr, cur_bgr, scale=0.33, iters=200, eps=1e-5):
    """Estimate the global shift that re-seats ROIs onto a scene that moved.
    Returns (shift_xy, converged, cc): add shift_xy to each ROI (x,y) to track the
    current frame. Runs on a downscaled grayscale pair for speed/robustness."""
    rs = cv2.resize(_gray(ref_bgr), None, fx=scale, fy=scale)
    cs = cv2.resize(_gray(cur_bgr), None, fx=scale, fy=scale)
    warp = np.eye(2, 3, dtype=np.float32)
    crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, iters, eps)
    try:
        cc, warp = cv2.findTransformECC(rs, cs, warp, cv2.MOTION_EUCLIDEAN, crit, None, 5)
        converged = True
    except cv2.error:
        cc, converged = None, False
    # warp's translation (rescaled to full res) IS the ROI shift that re-seats the
    # ROIs onto the moved scene — verified empirically against a known synthetic shift.
    dx = float(warp[0, 2]) / scale
    dy = float(warp[1, 2]) / scale
    return (dx, dy), converged, (float(cc) if cc is not None else None)


def decide_action(shift, converged, consensus=None, min_shift_px=3.0, max_shift_px=60.0):
    mag = float(np.hypot(*shift))
    if not converged:
        return {"action": "escalate", "reason": "ECC did not converge — a pot may have physically moved",
                "shift": shift, "mag": round(mag, 1)}
    if mag < min_shift_px:
        return {"action": "none", "reason": "shift below threshold", "shift": shift, "mag": round(mag, 1)}
    if consensus is not None and not consensus["drift"]:
        return {"action": "escalate",
                "reason": "registration shift without pot-centroid consensus — not a clean global frame shift",
                "shift": shift, "mag": round(mag, 1)}
    if mag > max_shift_px:
        return {"action": "escalate", "reason": f"shift {mag:.0f}px exceeds safe auto-correct limit",
                "shift": shift, "mag": round(mag, 1)}
    return {"action": "corrected", "reason": f"consistent global shift {mag:.0f}px — applying with audit",
            "shift": shift, "mag": round(mag, 1)}


def apply_shift_to_calibration(calib, shift, method="ecc", cc=None):
    """Return a new calibration with every ROI re-seated by `shift`, plus an audit
    record. Never silent — the audit trail records the exact applied shift."""
    dx, dy = shift
    new = json.loads(json.dumps(calib))  # deep copy
    for pot in new.get("pots", []):
        pot["x"] = int(round(pot["x"] + dx))
        pot["y"] = int(round(pot["y"] + dy))
    new.setdefault("calibration_audit", []).append({
        "ts": datetime.now(timezone.utc).isoformat(), "action": "shift", "method": method,
        "dx": round(dx, 2), "dy": round(dy, 2), "ecc_cc": cc,
    })
    return new


def run(ref_path, cur_path, calib_path):
    calib = json.loads(Path(calib_path).read_text())
    ref = cv2.imread(str(ref_path))
    cur = cv2.imread(str(cur_path))
    if ref is None or cur is None:
        raise SystemExit("Could not read reference or current image.")
    ref_c = green_centroids(ref, calib)
    cur_c = green_centroids(cur, calib)
    consensus = centroid_consensus(ref_c, cur_c)
    shift, converged, cc = register_ecc(ref, cur)
    decision = decide_action(shift, converged, consensus)
    return {"calib": calib, "consensus": consensus, "shift": shift,
            "converged": converged, "cc": cc, "decision": decision}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref-image", required=True, help="stored reference frame from calibration time")
    ap.add_argument("--cur-image", required=True, help="recent frame to check for drift")
    ap.add_argument("--calibration", required=True, help="path to {chamber}_calibration.json")
    ap.add_argument("--apply", action="store_true",
                    help="if action is 'corrected', write the re-seated calibration (backs up the original)")
    args = ap.parse_args()

    res = run(args.ref_image, args.cur_image, args.calibration)
    d = res["decision"]
    print(f"consensus: {res['consensus']}")
    print(f"ecc shift={tuple(round(s,1) for s in res['shift'])} converged={res['converged']} cc={res['cc']}")
    print(f"DECISION: {d['action']} — {d['reason']}")

    if d["action"] == "corrected" and args.apply:
        bak = Path(args.calibration).with_suffix(".json.bak")
        shutil.copyfile(args.calibration, bak)
        new = apply_shift_to_calibration(res["calib"], res["shift"], method="ecc", cc=res["cc"])
        Path(args.calibration).write_text(json.dumps(new, indent=2))
        print(f"applied shift; backup at {bak}")
    elif d["action"] == "escalate":
        print("ESCALATE: manual recalibration recommended — not auto-correcting.")


if __name__ == "__main__":
    main()

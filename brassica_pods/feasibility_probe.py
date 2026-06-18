"""
feasibility_probe.py — turn occlusion-probe data into a go/no-go (see PROBE.md).

Collect per plant: a non-destructive VISIBLE pod count and the destructive TRUE
count (optionally a photo-visible count that better mimics a fixed rig). This
computes the visible fraction f = visible/true, its consistency CV(f), and the
PRE-COMMITTED verdict.

Key fact (PROBE.md): with a perfect detector + single global calibration, mean
count %error ≈ CV(f). So CV(f) is an OPTIMISTIC FLOOR on achievable error — a real
detector only makes it worse. The thresholds bake in headroom for that.

CSV columns: plant_id, true_count, visible_count[, photo_visible_count, strip_minutes]
Decision uses photo_visible_count when present (closer to a real rig), else visible_count.

CLI:
    python -m brassica_pods.feasibility_probe --csv probe_data.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

# Pre-committed thresholds (do NOT move after seeing the data).
GO_CV_MAX = 0.10
GO_FMEAN_MIN = 0.50
NOGO_CV_MIN = 0.20      # CV strictly above this -> NO-GO
NOGO_FMEAN_MIN = 0.40   # f_mean strictly below this -> NO-GO


def verdict(cv_f: float, f_mean: float) -> str:
    """Apply the locked decision rule to CV(f) and mean visible fraction."""
    if cv_f > NOGO_CV_MIN or f_mean < NOGO_FMEAN_MIN:
        return "NO-GO"
    if cv_f <= GO_CV_MAX and f_mean >= GO_FMEAN_MIN:
        return "GO"
    return "MARGINAL"


def analyze_probe(records: list[dict]) -> dict:
    """records: [{plant_id, visible, true}] -> stats + verdict.

    `visible` is the chosen visible count (photo if available, else human).
    """
    rows = [r for r in records if r.get("true", 0) > 0]
    if len(rows) < 2:
        raise ValueError("Need ≥2 plants with true_count>0 to estimate CV(f).")
    f = np.array([r["visible"] / r["true"] for r in rows], dtype=float)
    f_mean = float(f.mean())
    f_std = float(f.std(ddof=1))            # sample std — probe is a sample
    cv = f_std / f_mean if f_mean > 0 else float("inf")
    return {
        "n_plants": len(rows),
        "f_mean": round(f_mean, 4),
        "f_std": round(f_std, 4),
        "cv_f": round(cv, 4),
        "optimistic_mape_pct": round(cv * 100, 2),   # ≈ best-possible count %error
        "f_min": round(float(f.min()), 4),
        "f_max": round(float(f.max()), 4),
        "verdict": verdict(cv, f_mean),
    }


def load_probe_csv(path) -> list[dict]:
    out = []
    with open(path, newline="", encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh):
            true_c = float(r["true_count"])
            photo = r.get("photo_visible_count", "").strip()
            visible = float(photo) if photo else float(r["visible_count"])
            out.append({"plant_id": r.get("plant_id", ""),
                        "visible": visible, "true": true_c,
                        "used_photo": bool(photo)})
    return out


def _report(stats: dict, records: list[dict]) -> str:
    # ASCII only — printed live on the (Windows) console; non-ASCII crashes it.
    used_photo = any(r.get("used_photo") for r in records)
    src = "photo-visible" if used_photo else "human-visible"
    lines = [
        f"Feasibility probe -- {stats['n_plants']} plants ({src} counts)",
        f"  visible fraction f: mean={stats['f_mean']:.2f}  "
        f"range=[{stats['f_min']:.2f}, {stats['f_max']:.2f}]  CV={stats['cv_f']:.2f}",
        f"  optimistic count %error floor (~CV): {stats['optimistic_mape_pct']:.1f}%  "
        f"(a real detector does WORSE)",
        "",
        f"  VERDICT: {stats['verdict']}",
    ]
    if stats["verdict"] == "GO":
        lines.append("  -> +/-10-15% plausible with a good detector. Build the station.")
    elif stats["verdict"] == "MARGINAL":
        lines.append("  -> +/-15% unlikely; ~+/-20-30% estimate at best. Decide if useful.")
    else:
        lines.append("  -> Non-destructive absolute count NOT achievable. Fall back to "
                     "spread-and-count (semi-automated) or estimate-only.")
    return "\n".join(lines)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Occlusion feasibility probe analysis")
    ap.add_argument("--csv", required=True, help="probe data CSV (see PROBE.md)")
    a = ap.parse_args()
    recs = load_probe_csv(a.csv)
    print(_report(analyze_probe(recs), recs))

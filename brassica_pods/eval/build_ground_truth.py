"""
build_ground_truth.py — per-image count + size ground truth for the real
deepcanola "novel" validation images.

The real images (validation_data.zip / novel_data/<set>/) have NO segmentation
masks (their COCO json lists images but zero annotations). They DO have manual
pod measurements in 'POD SCAN DATA.csv', keyed by Barcode, and each image is
named '<prefix>-<barcode> 001.jpg'. So:

    ground-truth pod count for an image = number of CSV rows for its barcode
    per-pod length (mm)                 = Length_valve_mm + Length_beak_mm

(decimal comma in the CSV, e.g. "53,586" -> 53.586). This is the eval target:
train on synthetic, validate count/size against these real measurements.

CAVEAT: row count = number of *measured* pods. If deepcanola did not measure
every pod on every scan, this is a lower bound on the true count — treat as the
manual-count reference, exactly as the source study used it.

CLI:
    python -m brassica_pods.eval.build_ground_truth \
        --novel "C:/Users/LukeB/brassica_data/deepcanola/extracted/novel_data/br017" \
        --out brassica_pods/data/eval_ground_truth.csv
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
from pathlib import Path

_BARCODE_RE = re.compile(r"-(\d{4,})\s")   # '...-113122 001.jpg' -> 113122


def _num(s: str):
    s = (s or "").strip().replace(",", ".")
    try:
        return float(s)
    except ValueError:
        return None


def barcode_from_filename(name: str) -> str | None:
    m = _BARCODE_RE.search(name)
    return m.group(1) if m else None


def build(novel_dir: str | Path) -> list[dict]:
    """Return [{image, barcode, gt_count, mean_pod_length_mm}] for one set."""
    novel_dir = Path(novel_dir)
    csvs = glob.glob(str(novel_dir / "*POD SCAN DATA*.csv"))
    if not csvs:
        raise FileNotFoundError(f"No 'POD SCAN DATA' CSV in {novel_dir}")
    rows = list(csv.DictReader(open(csvs[0], encoding="utf-8-sig")))

    # Group measurements by barcode.
    by_bc: dict[str, list[float]] = {}
    for r in rows:
        bc = (r.get("Barcode") or "").strip()
        if not bc:
            continue
        valve = _num(r.get("Length_valve _mm") or r.get("Length_valve_mm"))
        beak = _num(r.get("Length_beak_mm"))
        total = (valve or 0.0) + (beak or 0.0) if (valve or beak) else None
        by_bc.setdefault(bc, []).append(total) if total else by_bc.setdefault(bc, [])

    images = sorted(glob.glob(str(novel_dir / "images" / "*.jpg")))
    out = []
    for img in images:
        name = os.path.basename(img)
        bc = barcode_from_filename(name)
        lengths = [x for x in by_bc.get(bc, []) if x is not None]
        if bc is None or bc not in by_bc:
            continue
        count = len(by_bc[bc])
        out.append({
            "image": name,
            "barcode": bc,
            "gt_count": count,
            "mean_pod_length_mm": round(sum(lengths) / len(lengths), 2) if lengths else None,
        })
    return out


def write_csv(records: list[dict], out_path: str | Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["image", "barcode", "gt_count",
                                          "mean_pod_length_mm"])
        w.writeheader()
        w.writerows(records)
    counts = [r["gt_count"] for r in records]
    print(f"Wrote {len(records)} rows -> {out_path}")
    if counts:
        print(f"  pod count: min={min(counts)} max={max(counts)} "
              f"mean={sum(counts)/len(counts):.1f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--novel", required=True, help="novel_data/<set> dir")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    write_csv(build(a.novel), a.out)

"""
analyze.py — brassica silique inference (count + size + greenness).

Thin species wrapper over the shared engine `phenotyping.object_count`: brassica
uses YOLO-seg with a single "pod" class. The generic detect/dedup/size-filter/
greenness logic now lives in `phenotyping/` (promoted per the README); this file
keeps the brassica-specific defaults and the historical return contract.

    analyze(image, scale=None, weights=DEFAULT_WEIGHTS, conf=0.25,
            greenness=False) -> {
        "count":   int,
        "masks":   list[H×W bool],
        "per_pod": [ {length_mm,width_mm,area_mm2,length_px,width_px,area_px,
                      confidence}, ... ],
        "method":  "yolo-seg",
        "greenness": {...},     # only if greenness=True
    }

Count is the primary deliverable (Diarmuid); size comes free off the masks.
`pod_size` / `pod_greenness` remain as aliases for backward compatibility.
"""
from __future__ import annotations

from brassica_pods.common import WEIGHTS_DIR
from phenotyping.object_count import (
    detect_and_count,
    instance_size as pod_size,          # backward-compat alias
    instance_greenness,
)

DEFAULT_WEIGHTS = WEIGHTS_DIR / "pods_best.pt"

# Drop detections smaller than this fraction of the median pod area (beak/tip
# slivers). 0.30 was near-perfect on uniform ~20-pod scans but undercounts dense,
# size-varied scans; 0.20 is the robustness-hedged default. Re-tune on rig data.
MIN_AREA_FRAC = 0.20


def pod_greenness(masks, img, species: str = "brassica") -> dict:
    """Greenness metrics over the union of pod masks (brassica-configured)."""
    return instance_greenness(masks, img, species=species)


def analyze(image, scale=None, weights=DEFAULT_WEIGHTS, conf: float = 0.25,
            greenness: bool = False, imgsz: int = 1536, dedup: bool = True,
            min_area_frac: float = MIN_AREA_FRAC) -> dict:
    """Segment, count and size every silique in one image (see module docstring).

    imgsz defaults to 1536 to match the training resolution — siliques are thin
    and undercount at YOLO's 640 default.
    """
    out = detect_and_count(
        image, weights=weights, conf=conf, imgsz=imgsz, dedup=dedup,
        min_area_frac=min_area_frac, scale=scale, greenness=greenness,
        species="brassica",
    )
    out["per_pod"] = out.pop("instances")   # preserve the brassica contract
    return out

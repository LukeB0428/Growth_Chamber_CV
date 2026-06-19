"""
analyze.py — inference entry point for the brassica_pods module.

Exposes the single clean function the plan (§4) mandates:

    analyze(image, scale=None, weights=None, conf=0.25, greenness=False) -> {
        "count":   int,
        "masks":   list[np.ndarray],          # H×W bool, one per pod
        "per_pod": [ {"length_mm","width_mm","area_mm2","length_px","width_px",
                      "area_px","confidence"} , ... ],
        "method":  str,                        # "yolo-seg" | ...
        "greenness": {...},                    # only if greenness=True (pod union)
    }

Count is the primary deliverable (Diarmuid). Size metrics (length/width/area)
and greenness are kept because they come cheaply off the same masks.

The model is Ultralytics YOLO-seg, run on CPU locally (training happens on
Colab/GPU; see train/). Until trained weights exist, analyze() raises a clear
error pointing at the training step — the *phenotype extraction* below is fully
implemented and unit-testable from masks alone, so it is the stable contract
downstream code (logging, dashboard, commercial wrapper) can build on now.

Size metrics (per pod, plan §5):
  - length / width : sides of the min-area rotated rectangle around the mask.
  - area           : mask pixel count.
  All converted to mm via a PixelScale (px->mm). Without a scale, only *_px
  fields are populated and mm fields are None.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from brassica_pods.common import PixelScale, WEIGHTS_DIR, ensure_scripts_importable

DEFAULT_WEIGHTS = WEIGHTS_DIR / "pods_best.pt"

# Mask-based dedup: drop a detection if this fraction of ITS mask is already
# covered by a higher-confidence detection. Catches duplicate boxes on one
# diagonal pod that axis-aligned NMS misses (their boxes barely overlap but
# their masks nearly coincide).
DEDUP_CONTAINMENT = 0.45


def _dedup_masks(masks, confs):
    """Greedy mask-NMS: keep highest-confidence masks, drop ones mostly covered
    by an already-kept mask. Returns the indices to KEEP (in original order)."""
    order = sorted(range(len(masks)), key=lambda i: (confs[i] if confs is not None
                                                      else 0.0), reverse=True)
    kept_union_idx: list[int] = []
    keep = []
    for i in order:
        m = masks[i]
        area = int(m.sum())
        if area == 0:
            continue
        covered = max((int(np.logical_and(m, masks[j]).sum()) / area
                       for j in keep), default=0.0)
        if covered <= DEDUP_CONTAINMENT:
            keep.append(i)
    return sorted(keep)


# ─────────────────────────────────────────────────────────────────────────────
# Greenness — reuse the Arabidopsis pipeline's metric module (no rewrite)
# ─────────────────────────────────────────────────────────────────────────────
def pod_greenness(masks: list[np.ndarray], img: np.ndarray,
                  species: str = "brassica") -> dict:
    """Greenness metrics (NGRDI/GCC/Lab/score/shade) over the union of pod masks.

    Reuses scripts/greenness_metrics.py by reference and configures it from
    config/species/brassica.json — scripts/ is not modified. Pod greenness
    tracks maturation (green siliques -> yellowing at ripening). For whole-plant
    greenness, pass a canopy mask list instead.
    """
    if not masks:
        return {}
    ensure_scripts_importable()
    import greenness_metrics as gm
    try:
        from species_config import load_and_apply
        load_and_apply(species)              # loads brassica greenness bounds
    except Exception:
        pass                                  # falls back to module defaults
    union = np.zeros(img.shape[:2], dtype=np.uint8)
    for m in masks:
        union[m > 0] = 255
    return gm.compute_greenness_metrics(union, img)


# ─────────────────────────────────────────────────────────────────────────────
# Phenotype extraction — pure function of a binary mask (no model needed)
# ─────────────────────────────────────────────────────────────────────────────
def pod_size(mask: np.ndarray, scale: Optional[PixelScale] = None) -> dict:
    """Length, width and area for one pod mask.

    length/width come from the minimum-area rotated rectangle (robust to pod
    orientation); area is the mask pixel count. mm fields are filled only when
    a PixelScale is supplied.
    """
    m = (mask > 0).astype(np.uint8)
    area_px = int(m.sum())
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts or area_px == 0:
        length_px = width_px = 0.0
    else:
        c = max(cnts, key=cv2.contourArea)
        (_, _), (w, h), _ = cv2.minAreaRect(c)
        length_px, width_px = (max(w, h), min(w, h))

    out = {
        "length_px": float(length_px),
        "width_px": float(width_px),
        "area_px": area_px,
        "length_mm": None,
        "width_mm": None,
        "area_mm2": None,
    }
    if scale is not None:
        out["length_mm"] = round(scale.length_mm(length_px), 3)
        out["width_mm"] = round(scale.length_mm(width_px), 3)
        out["area_mm2"] = round(scale.area_mm2(area_px), 3)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Model loading (lazy — Ultralytics import is heavy)
# ─────────────────────────────────────────────────────────────────────────────
_MODEL_CACHE: dict[str, object] = {}


def _load_model(weights: str | Path):
    weights = str(weights)
    if weights in _MODEL_CACHE:
        return _MODEL_CACHE[weights]
    if not Path(weights).is_file():
        raise FileNotFoundError(
            f"No YOLO-seg weights at {weights}. Train a model first "
            f"(see brassica_pods/train/), then place the checkpoint here.")
    from ultralytics import YOLO  # heavy import, deferred
    model = YOLO(weights)
    _MODEL_CACHE[weights] = model
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────
# Drop detections smaller than this fraction of the median pod area — removes
# the small "beak/tip" slivers the model detects as separate pods.
# 0.30 was near-perfect on 12 uniform ~20-pod scans (MAE 0.3) but UNDERcounts
# denser/size-varied scans (a real 40-pod scan dropped to 20). 0.20 is the
# robustness-hedged default: slightly higher error on uniform scans (MAE ~1.5)
# but preserves real smaller pods. Re-tune on real-rig data, not deepcanola.
MIN_AREA_FRAC = 0.20


def analyze(image, scale: Optional[PixelScale] = None,
            weights: str | Path = DEFAULT_WEIGHTS, conf: float = 0.25,
            greenness: bool = False, imgsz: int = 1536, dedup: bool = True,
            min_area_frac: float = MIN_AREA_FRAC) -> dict:
    """Segment, count and size every pod in one RGB/BGR image.

    Args:
        image:     BGR ndarray (cv2) or a path to an image file.
        scale:     PixelScale for px->mm; if None, only *_px metrics are returned.
        weights:   path to a trained YOLO-seg checkpoint.
        conf:      detection confidence threshold.
        greenness: if True, also return greenness metrics over the pod union
                   (reuses scripts/greenness_metrics.py).
        imgsz:     inference resolution; default 1536 to MATCH the training imgsz
                   (siliques are thin — predicting at YOLO's 640 default shrinks
                   them below training scale and undercounts).

    Returns the dict described in this module's docstring (plus "greenness"
    when requested).
    """
    if isinstance(image, (str, Path)):
        img = cv2.imread(str(image))
        if img is None:
            raise FileNotFoundError(f"Could not read image: {image}")
    else:
        img = image
    H, W = img.shape[:2]

    model = _load_model(weights)
    results = model.predict(img, conf=conf, imgsz=imgsz, verbose=False)[0]

    masks: list[np.ndarray] = []
    per_pod: list[dict] = []

    if results.masks is not None:
        confs = (results.boxes.conf.cpu().numpy()
                 if results.boxes is not None else None)
        # Collect raw masks first.
        raw_masks, raw_confs = [], []
        for i, mdata in enumerate(results.masks.data):
            m = mdata.cpu().numpy().astype(np.uint8)
            if m.shape[:2] != (H, W):                 # YOLO masks come at net res
                m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
            m_bool = m > 0
            if not m_bool.any():
                continue
            raw_masks.append(m_bool)
            raw_confs.append(float(confs[i]) if confs is not None else None)

        # Mask-based dedup — removes duplicate boxes on one diagonal pod that
        # axis-aligned NMS leaves behind.
        keep = (_dedup_masks(raw_masks, raw_confs) if dedup
                else list(range(len(raw_masks))))
        # Size filter — drop beak/tip slivers far smaller than a typical pod.
        if keep and min_area_frac > 0:
            areas = {i: int(raw_masks[i].sum()) for i in keep}
            med = float(np.median(list(areas.values())))
            keep = [i for i in keep if areas[i] >= min_area_frac * med]
        for i in keep:
            masks.append(raw_masks[i])
            info = pod_size(raw_masks[i], scale)
            info["confidence"] = raw_confs[i]
            per_pod.append(info)

    out = {
        "count": len(masks),
        "masks": masks,
        "per_pod": per_pod,
        "method": "yolo-seg",
    }
    if greenness:
        out["greenness"] = pod_greenness(masks, img)
    return out

"""
object_count.py — generic detect-and-count engine for repeated plant structures.

One Ultralytics YOLO model (segmentation OR detection) → per-instance masks, a
total count, and per-instance size. Species modules parameterise it:
  - brassica_pods : YOLO-seg, class "pod"    (siliques; masks → precise size)
  - barley        : YOLO-det, class "spike"  (GWHD-pretrained cereal heads; boxes)

`detect_and_count()` returns:
    {
      "count":     int,
      "masks":     list[np.ndarray],     # H×W bool, one per instance
      "instances": [ {length_mm,width_mm,area_mm2,length_px,width_px,area_px,
                      confidence}, ... ],
      "method":    "yolo-seg" | "yolo-det" | "none",
      "greenness": {...},                # only if greenness=True
    }

Generalised from brassica_pods.infer.analyze (which now delegates here). The
seg→masks path is identical to the original; the det→boxes path is new, so a
detection-only model (e.g. GWHD spike detector) works through the same pipeline
by synthesising a rectangular mask per box.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from phenotyping.paths import ensure_scripts_importable

# Mask-based dedup: drop a detection if this fraction of ITS mask is already
# covered by a higher-confidence one — catches duplicate boxes on one diagonal
# instance that axis-aligned NMS misses.
DEDUP_CONTAINMENT = 0.45

_MODEL_CACHE: dict[str, object] = {}


def _load_model(weights):
    weights = str(weights)
    if weights in _MODEL_CACHE:
        return _MODEL_CACHE[weights]
    if not Path(weights).is_file():
        raise FileNotFoundError(
            f"No YOLO weights at {weights}. Train or download a model first.")
    from ultralytics import YOLO  # heavy import, deferred
    model = YOLO(weights)
    _MODEL_CACHE[weights] = model
    return model


def _dedup_masks(masks, confs):
    """Greedy mask-NMS: keep highest-confidence masks, drop ones mostly covered
    by an already-kept mask. Returns the indices to KEEP (in original order)."""
    order = sorted(range(len(masks)),
                   key=lambda i: (confs[i] if confs is not None else 0.0),
                   reverse=True)
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


def instance_size(mask: np.ndarray, scale=None) -> dict:
    """Length, width and area for one instance mask. length/width come from the
    minimum-area rotated rectangle (robust to orientation); area is pixel count.
    mm fields are filled only when a PixelScale is supplied."""
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
        "length_px": float(length_px), "width_px": float(width_px),
        "area_px": area_px, "length_mm": None, "width_mm": None, "area_mm2": None,
    }
    if scale is not None:
        out["length_mm"] = round(scale.length_mm(length_px), 3)
        out["width_mm"] = round(scale.length_mm(width_px), 3)
        out["area_mm2"] = round(scale.area_mm2(area_px), 3)
    return out


def instance_greenness(masks: list[np.ndarray], img: np.ndarray,
                       species: Optional[str] = None) -> dict:
    """Greenness metrics over the union of instance masks. Reuses
    scripts/greenness_metrics.py by reference; if `species` is given, configures
    it from config/species/{species}.json. scripts/ is never modified."""
    if not masks:
        return {}
    ensure_scripts_importable()
    import greenness_metrics as gm
    if species:
        try:
            from species_config import load_and_apply
            load_and_apply(species)
        except Exception:
            pass  # fall back to module defaults
    union = np.zeros(img.shape[:2], dtype=np.uint8)
    for m in masks:
        union[m > 0] = 255
    return gm.compute_greenness_metrics(union, img)


def _instances_from_results(results, H, W):
    """Extract (raw_masks, raw_confs, method) from a YOLO result — segmentation
    masks if present, otherwise a rectangular mask per detection box."""
    confs = (results.boxes.conf.cpu().numpy()
             if results.boxes is not None else None)
    raw_masks, raw_confs = [], []

    if results.masks is not None:                       # segmentation model
        method = "yolo-seg"
        for i, mdata in enumerate(results.masks.data):
            m = mdata.cpu().numpy().astype(np.uint8)
            if m.shape[:2] != (H, W):                   # YOLO masks at net res
                m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
            mb = m > 0
            if not mb.any():
                continue
            raw_masks.append(mb)
            raw_confs.append(float(confs[i]) if confs is not None else None)
    elif results.boxes is not None:                     # detection-only model
        method = "yolo-det"
        xyxy = results.boxes.xyxy.cpu().numpy()
        for i, (x1, y1, x2, y2) in enumerate(xyxy):
            x1, y1 = max(0, int(x1)), max(0, int(y1))
            x2, y2 = min(W, int(x2)), min(H, int(y2))
            if x2 <= x1 or y2 <= y1:
                continue
            mb = np.zeros((H, W), dtype=bool)
            mb[y1:y2, x1:x2] = True
            raw_masks.append(mb)
            raw_confs.append(float(confs[i]) if confs is not None else None)
    else:
        method = "none"
    return raw_masks, raw_confs, method


def detect_and_count(image, weights, conf: float = 0.25, imgsz: int = 1536,
                     dedup: bool = True, min_area_frac: float = 0.20,
                     scale=None, greenness: bool = False,
                     species: Optional[str] = None) -> dict:
    """Detect, dedup, size-filter and count every instance in one image.

    Args:
        image:    BGR ndarray (cv2) or path.
        weights:  trained YOLO checkpoint (seg or det).
        imgsz:    inference resolution (match training; thin structures undercount
                  at YOLO's 640 default).
        min_area_frac: drop instances smaller than this fraction of the median
                  area (removes sliver false-positives); set 0 to disable.
        greenness/species: optional greenness over the instance union.
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
    raw_masks, raw_confs, method = _instances_from_results(results, H, W)

    keep = (_dedup_masks(raw_masks, raw_confs)
            if (dedup and raw_masks) else list(range(len(raw_masks))))
    if keep and min_area_frac > 0:
        areas = {i: int(raw_masks[i].sum()) for i in keep}
        med = float(np.median(list(areas.values())))
        keep = [i for i in keep if areas[i] >= min_area_frac * med]

    masks, instances = [], []
    for i in keep:
        masks.append(raw_masks[i])
        info = instance_size(raw_masks[i], scale)
        info["confidence"] = raw_confs[i]
        instances.append(info)

    out = {"count": len(masks), "masks": masks,
           "instances": instances, "method": method}
    if greenness:
        out["greenness"] = instance_greenness(masks, img, species)
    return out

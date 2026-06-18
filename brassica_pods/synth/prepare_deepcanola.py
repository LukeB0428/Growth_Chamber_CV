"""
prepare_deepcanola.py — stage deepcanola "data pools" into our synth inputs.

The deepcanola ground_truth_data.zip ships the raw material our synthetic
generator needs, but in deepcanola's naming:

    pod_pools/<set>/images/<scan>_pod<ID>.jpg        # RGB pod cut-out
    pod_pools/<set>/masks/<scan>_mask_pod<ID>.jpg    # matching mask
    background_pool/*.jpg                             # backgrounds

This adapter pairs each cut-out with its mask by (scan, pod ID), composites them
into RGBA PNGs (alpha = mask), and copies backgrounds — producing exactly the
inputs brassica_pods/synth/generate_dataset.py expects. RGBA output means the
generator's --masks arg is then unnecessary.

Reads from brassica_pods.common.DEEPCANOLA_DIR (override via BRASSICA_DATA_DIR).
Source data: Atkins et al., Zenodo 10.5281/zenodo.13903900 (CC-BY-4.0).

CLI:
    python -m brassica_pods.synth.prepare_deepcanola --limit 400
"""

from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path

import cv2
import numpy as np

from brassica_pods.common import (DEEPCANOLA_DIR, POD_CUTOUTS_DIR,
                                   BACKGROUNDS_DIR)

GT_ROOT = DEEPCANOLA_DIR / "extracted" / "ground_truth_data"
_POD_RE = re.compile(r"^(?P<scan>.+?)_(?:mask_)?pod(?P<id>\d+)$", re.IGNORECASE)


def _key(stem: str):
    """(scan, pod_id) key shared by an image and its mask, or None."""
    m = _POD_RE.match(stem)
    return (m.group("scan"), m.group("id")) if m else None


def _downscale(img, factor: float, interp):
    if factor == 1.0:
        return img
    h, w = img.shape[:2]
    return cv2.resize(img, (max(1, int(w * factor)), max(1, int(h * factor))),
                      interpolation=interp)


def stage(limit: int | None = None, gt_root: Path = GT_ROOT,
          downscale: float = 1.0) -> dict:
    """Build RGBA cut-outs in POD_CUTOUTS_DIR and copy backgrounds.

    downscale (0<f<=1) shrinks BOTH cut-outs and backgrounds by the same factor
    so their relative scale is preserved. The deepcanola scans are ~3500 px tall
    with ~1000 px pods; downscaling keeps the synthetic set near training
    resolution (imgsz) and the on-disk size manageable.
    """
    pod_pools = gt_root / "pod_pools"
    bg_pool = gt_root / "background_pool"
    if not pod_pools.is_dir():
        raise FileNotFoundError(
            f"{pod_pools} not found — extract ground_truth_data.zip first.")

    POD_CUTOUTS_DIR.mkdir(parents=True, exist_ok=True)
    BACKGROUNDS_DIR.mkdir(parents=True, exist_ok=True)

    made = 0
    for set_dir in sorted(p for p in pod_pools.iterdir() if p.is_dir()):
        img_dir, mask_dir = set_dir / "images", set_dir / "masks"
        if not (img_dir.is_dir() and mask_dir.is_dir()):
            continue
        # Index masks by (scan, id).
        masks = {}
        for mp in mask_dir.glob("*.jpg"):
            k = _key(mp.stem)
            if k:
                masks[k] = mp
        for ip in sorted(img_dir.glob("*.jpg")):
            if limit and made >= limit:
                break
            k = _key(ip.stem)
            if not k or k not in masks:
                continue
            bgr = cv2.imread(str(ip))
            alpha = cv2.imread(str(masks[k]), cv2.IMREAD_GRAYSCALE)
            if bgr is None or alpha is None:
                continue
            if alpha.shape[:2] != bgr.shape[:2]:
                alpha = cv2.resize(alpha, (bgr.shape[1], bgr.shape[0]),
                                   interpolation=cv2.INTER_NEAREST)
            bgr = _downscale(bgr, downscale, cv2.INTER_AREA)
            alpha = _downscale(alpha, downscale, cv2.INTER_NEAREST)
            alpha = (alpha > 127).astype(np.uint8) * 255
            if alpha.sum() == 0:
                continue
            rgba = np.dstack([bgr, alpha])
            out = POD_CUTOUTS_DIR / f"{set_dir.name}_{k[0]}_pod{k[1]}.png"
            cv2.imwrite(str(out), rgba)
            made += 1

    # Copy backgrounds (downscaled to match cut-outs).
    bg_copied = 0
    if bg_pool.is_dir():
        for bp in bg_pool.glob("*"):
            if bp.suffix.lower() in (".jpg", ".jpeg", ".png"):
                bg = cv2.imread(str(bp))
                if bg is None:
                    continue
                cv2.imwrite(str(BACKGROUNDS_DIR / bp.name),
                            _downscale(bg, downscale, cv2.INTER_AREA))
                bg_copied += 1

    print(f"Staged {made} RGBA pod cut-outs -> {POD_CUTOUTS_DIR}")
    print(f"Copied {bg_copied} backgrounds   -> {BACKGROUNDS_DIR}")
    return {"cutouts": made, "backgrounds": bg_copied}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Stage deepcanola data pools for synth")
    ap.add_argument("--limit", type=int, default=None,
                    help="max cut-outs to stage (default: all)")
    ap.add_argument("--downscale", type=float, default=1.0,
                    help="shrink cut-outs+backgrounds by this factor (e.g. 0.44)")
    ap.add_argument("--gt-root", default=str(GT_ROOT))
    a = ap.parse_args()
    stage(limit=a.limit, gt_root=Path(a.gt_root), downscale=a.downscale)

"""
generate_dataset.py — deepcanola-style semi-synthetic dataset generator.

Reimplements the *methodology* of deepcanola (Atkins et al., GPL-3.0) — paste
cut-out pods with their masks onto empty-rig backgrounds to amplify a tiny
hand-labelled seed set — WITHOUT vendoring its GPL code (plan §7). Output is
standard Ultralytics YOLO-seg format so it trains with stock tooling.

Data contract (mirrors deepcanola's utils/generate_dataset.sh inputs):
    pod_cutouts/   <name>.png           RGBA or BGR pod crops (one pod each)
    pod_masks/     <name>_mask.png      binary mask, same size as the cut-out
    backgrounds/   *.jpg|*.png          empty-rig / black-cloth backgrounds

For an RGBA cut-out the alpha channel is used as the mask and pod_masks/ is
optional. For a BGR cut-out a matching <name>_mask.png is required.

Output (YOLO-seg layout under --out):
    images/<split>/<stem>.jpg
    labels/<split>/<stem>.txt     one line per pod:  0 x1 y1 x2 y2 ... (normalised)

This generator is intentionally deterministic given (rng_seed, index) so runs
are reproducible — Math.random-style nondeterminism would make the train/val
split impossible to audit.

CLI:
    python -m brassica_pods.synth.generate_dataset \
        --cutouts brassica_pods/data/pod_cutouts \
        --masks   brassica_pods/data/pod_masks \
        --backgrounds brassica_pods/data/backgrounds \
        --out     brassica_pods/data/synthetic \
        --num-images 500 --min-pods 8 --max-pods 60 --val-frac 0.2 --seed 42
"""

from __future__ import annotations

import argparse
import glob
import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

# Minimum mask area (px) for a pasted pod to be worth a label after transform.
MIN_PASTED_AREA_PX = 80
# Polygon simplification tolerance as a fraction of contour perimeter.
POLY_EPS_FRAC = 0.004
# A pasted pod must remain at least this fraction visible (after later pods
# occlude it) to be emitted as a label — otherwise its mask is a partial/clipped
# fragment, and training on fragments teaches the model to over-split pods.
MIN_VISIBLE_FRAC = 0.80
# The largest contour must dominate the mask this much, else it is fragmented
# (split into pieces by an occluder) and is dropped rather than mislabelled.
DOMINANT_CONTOUR_FRAC = 0.85


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class SynthConfig:
    """Tunable knobs for one generation run. Defaults span realistic density
    and overlap so the eval set's hard case (dense clusters) is represented."""
    num_images: int = 500
    min_pods: int = 8
    max_pods: int = 30
    # Per-pod augmentation ranges.
    scale_range: tuple[float, float] = (0.6, 1.4)
    rotate_deg: tuple[float, float] = (0.0, 360.0)
    # Max fraction of a new pod that may overlap existing pods. Keep LOW so pods
    # stay separated like real laid-out scans — high overlap fragments masks and
    # teaches the model to over-split single pods into several detections.
    max_overlap: float = 0.15
    val_frac: float = 0.2
    seed: int = 42
    jitter_brightness: float = 0.15   # ± fractional brightness per pod
    out_size: Optional[tuple[int, int]] = None  # (w, h); None = keep background size


@dataclass
class _PodSprite:
    bgr: np.ndarray      # H×W×3
    alpha: np.ndarray    # H×W uint8 (0/255)
    name: str = ""


# ─────────────────────────────────────────────────────────────────────────────
# Loading
# ─────────────────────────────────────────────────────────────────────────────
def load_sprites(cutouts_dir: str | Path,
                 masks_dir: str | Path | None) -> list[_PodSprite]:
    """Load pod cut-outs and their masks into RGBA-style sprites.

    RGBA cut-outs use their own alpha. BGR cut-outs require a sibling
    <name>_mask.png in masks_dir.
    """
    cutouts_dir = Path(cutouts_dir)
    sprites: list[_PodSprite] = []
    paths = sorted(p for p in cutouts_dir.iterdir()
                   if p.suffix.lower() in (".png", ".jpg", ".jpeg")
                   and not p.stem.endswith("_mask"))

    for p in paths:
        img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f"  skip (unreadable): {p.name}")
            continue

        if img.ndim == 3 and img.shape[2] == 4:           # RGBA cut-out
            bgr = img[:, :, :3]
            alpha = img[:, :, 3]
        else:                                              # BGR + external mask
            bgr = img if img.ndim == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            if masks_dir is None:
                print(f"  skip (no alpha, no masks dir): {p.name}")
                continue
            mask_path = Path(masks_dir) / f"{p.stem}_mask.png"
            if not mask_path.is_file():
                print(f"  skip (missing mask): {p.name}")
                continue
            alpha = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if alpha is None or alpha.shape[:2] != bgr.shape[:2]:
                print(f"  skip (mask size mismatch): {p.name}")
                continue

        alpha = (alpha > 127).astype(np.uint8) * 255
        if alpha.sum() == 0:
            continue
        sprites.append(_PodSprite(bgr=bgr, alpha=alpha, name=p.stem))

    return sprites


def load_backgrounds(backgrounds_dir: str | Path) -> list[np.ndarray]:
    paths = sorted(glob.glob(os.path.join(str(backgrounds_dir), "*")))
    bgs = [cv2.imread(p) for p in paths]
    return [b for b in bgs if b is not None]


# ─────────────────────────────────────────────────────────────────────────────
# Geometry helpers
# ─────────────────────────────────────────────────────────────────────────────
def _transform_sprite(sprite: _PodSprite, scale: float, angle: float,
                      rng: random.Random) -> _PodSprite:
    """Scale + rotate a sprite, keeping bgr and alpha in lockstep."""
    h, w = sprite.alpha.shape[:2]
    new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
    bgr = cv2.resize(sprite.bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
    alpha = cv2.resize(sprite.alpha, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

    # Rotate about centre with an expanded canvas so corners are not clipped.
    M = cv2.getRotationMatrix2D((new_w / 2, new_h / 2), angle, 1.0)
    cos, sin = abs(M[0, 0]), abs(M[0, 1])
    bound_w = int(new_h * sin + new_w * cos)
    bound_h = int(new_h * cos + new_w * sin)
    M[0, 2] += (bound_w - new_w) / 2
    M[1, 2] += (bound_h - new_h) / 2
    bgr = cv2.warpAffine(bgr, M, (bound_w, bound_h), flags=cv2.INTER_LINEAR,
                         borderValue=(0, 0, 0))
    alpha = cv2.warpAffine(alpha, M, (bound_w, bound_h), flags=cv2.INTER_NEAREST,
                           borderValue=0)
    return _PodSprite(bgr=bgr, alpha=(alpha > 127).astype(np.uint8) * 255,
                      name=sprite.name)


def _mask_to_polygon(mask: np.ndarray) -> Optional[np.ndarray]:
    """Largest external contour of a binary mask -> simplified Nx2 polygon (px).
    Returns None if the mask is fragmented (largest piece not dominant)."""
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    areas = [cv2.contourArea(c) for c in cnts]
    c = cnts[int(np.argmax(areas))]
    total = sum(areas)
    if cv2.contourArea(c) < MIN_PASTED_AREA_PX:
        return None
    if total > 0 and cv2.contourArea(c) / total < DOMINANT_CONTOUR_FRAC:
        return None        # split into pieces by an occluder -> drop, don't mislabel
    eps = POLY_EPS_FRAC * cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, eps, True).reshape(-1, 2)
    return approx if len(approx) >= 3 else None


# ─────────────────────────────────────────────────────────────────────────────
# Compositing
# ─────────────────────────────────────────────────────────────────────────────
def compose_one(background: np.ndarray, sprites: list[_PodSprite],
                cfg: SynthConfig, rng: random.Random
                ) -> tuple[np.ndarray, list[np.ndarray]]:
    """Paste a random crowd of pods onto a background copy.

    Returns (composite_bgr, [polygon_px, ...]) where each polygon is the visible
    outline of one pasted pod after occlusion by later-pasted pods.
    """
    canvas = background.copy()
    if cfg.out_size is not None:
        canvas = cv2.resize(canvas, cfg.out_size, interpolation=cv2.INTER_AREA)
    H, W = canvas.shape[:2]

    occupancy = np.zeros((H, W), dtype=np.uint8)   # union of placed pod masks
    placed_masks: list[np.ndarray] = []            # full-frame masks, paste order
    placed_areas: list[int] = []                   # each pod's area at placement

    n_pods = rng.randint(cfg.min_pods, cfg.max_pods)
    for _ in range(n_pods):
        sp = rng.choice(sprites)
        scale = rng.uniform(*cfg.scale_range)
        angle = rng.uniform(*cfg.rotate_deg)
        t = _transform_sprite(sp, scale, angle, rng)
        ph, pw = t.alpha.shape[:2]
        if ph >= H or pw >= W:
            continue

        x = rng.randint(0, W - pw)
        y = rng.randint(0, H - ph)
        frame_mask = np.zeros((H, W), dtype=np.uint8)
        frame_mask[y:y + ph, x:x + pw] = t.alpha

        area = int((frame_mask > 0).sum())
        if area == 0:
            continue
        # Reject placements whose overlap with existing pods exceeds the cap.
        overlap = int(np.logical_and(frame_mask > 0, occupancy > 0).sum())
        if overlap / area > cfg.max_overlap:
            continue

        # Optional brightness jitter for lighting variety.
        bgr = t.bgr.astype(np.float32)
        if cfg.jitter_brightness > 0:
            f = 1.0 + rng.uniform(-cfg.jitter_brightness, cfg.jitter_brightness)
            bgr = np.clip(bgr * f, 0, 255)
        bgr = bgr.astype(np.uint8)

        # Alpha-blend the pod onto the canvas.
        a3 = (t.alpha[:, :, None] / 255.0)
        roi = canvas[y:y + ph, x:x + pw].astype(np.float32)
        canvas[y:y + ph, x:x + pw] = (
            a3 * bgr + (1 - a3) * roi).astype(np.uint8)

        # Later pods occlude earlier ones: subtract new mask from prior masks.
        for pm in placed_masks:
            pm[frame_mask > 0] = 0
        placed_masks.append(frame_mask)
        placed_areas.append(area)
        occupancy[frame_mask > 0] = 255

    # Emit a label only for pods still mostly visible AND not fragmented — a
    # heavily occluded or split mask becomes a partial example that teaches
    # over-splitting.
    polygons = []
    for m, orig_area in zip(placed_masks, placed_areas):
        if orig_area == 0 or int((m > 0).sum()) / orig_area < MIN_VISIBLE_FRAC:
            continue
        poly = _mask_to_polygon(m)
        if poly is not None:
            polygons.append(poly)
    return canvas, polygons


def write_yolo_label(label_path: Path, polygons: list[np.ndarray],
                     img_w: int, img_h: int, class_id: int = 0) -> None:
    """Write one YOLO-seg label file: '<cls> x1 y1 ... xn yn' normalised [0,1]."""
    lines = []
    for poly in polygons:
        norm = poly.astype(np.float32).copy()
        norm[:, 0] /= img_w
        norm[:, 1] /= img_h
        norm = np.clip(norm, 0.0, 1.0)
        coords = " ".join(f"{v:.6f}" for v in norm.reshape(-1))
        lines.append(f"{class_id} {coords}")
    label_path.write_text("\n".join(lines))


# ─────────────────────────────────────────────────────────────────────────────
# Driver
# ─────────────────────────────────────────────────────────────────────────────
def generate(cutouts_dir, masks_dir, backgrounds_dir, out_dir,
             cfg: SynthConfig) -> dict:
    """Generate a full synthetic YOLO-seg dataset. Returns a summary dict."""
    sprites = load_sprites(cutouts_dir, masks_dir)
    backgrounds = load_backgrounds(backgrounds_dir)
    if not sprites:
        raise RuntimeError(f"No pod sprites loaded from {cutouts_dir}")
    if not backgrounds:
        raise RuntimeError(f"No backgrounds loaded from {backgrounds_dir}")
    print(f"Loaded {len(sprites)} pod sprites, {len(backgrounds)} backgrounds.")

    out_dir = Path(out_dir)
    rng = random.Random(cfg.seed)
    n_val = int(round(cfg.num_images * cfg.val_frac))
    counts = {"train": 0, "val": 0, "pods_total": 0}

    for i in range(cfg.num_images):
        split = "val" if i < n_val else "train"
        bg = rng.choice(backgrounds)
        composite, polygons = compose_one(bg, sprites, cfg, rng)
        h, w = composite.shape[:2]

        stem = f"synth_{i:05d}"
        img_dir = out_dir / "images" / split
        lbl_dir = out_dir / "labels" / split
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        cv2.imwrite(str(img_dir / f"{stem}.jpg"), composite)
        write_yolo_label(lbl_dir / f"{stem}.txt", polygons, w, h)
        counts[split] += 1
        counts["pods_total"] += len(polygons)
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{cfg.num_images} images")

    print(f"Done. train={counts['train']} val={counts['val']} "
          f"pods={counts['pods_total']}")
    return counts


def _parse_args():
    ap = argparse.ArgumentParser(description="deepcanola-style synthetic pod dataset generator")
    ap.add_argument("--cutouts", required=True)
    ap.add_argument("--masks", default=None, help="optional if cut-outs are RGBA")
    ap.add_argument("--backgrounds", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--num-images", type=int, default=500)
    ap.add_argument("--min-pods", type=int, default=8)
    ap.add_argument("--max-pods", type=int, default=30)
    ap.add_argument("--max-overlap", type=float, default=0.15)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()


if __name__ == "__main__":
    a = _parse_args()
    cfg = SynthConfig(
        num_images=a.num_images, min_pods=a.min_pods, max_pods=a.max_pods,
        max_overlap=a.max_overlap, val_frac=a.val_frac, seed=a.seed,
    )
    generate(a.cutouts, a.masks, a.backgrounds, a.out, cfg)

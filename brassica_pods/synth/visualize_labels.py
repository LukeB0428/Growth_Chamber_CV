"""
visualize_labels.py — overlay YOLO-seg polygon labels on an image.

Sanity-check that generated (or hand-labelled) YOLO-seg masks line up with the
pods in the image. Draws each polygon outline + a per-pod index and the count.

CLI:
    python -m brassica_pods.synth.visualize_labels \
        --image brassica_pods/data/synthetic/images/train/synth_00010.jpg \
        --out   brassica_pods/data/synthetic/_preview.jpg
The label file is inferred by swapping images/->labels/ and .jpg->.txt.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def label_path_for(image_path: Path) -> Path:
    parts = list(image_path.parts)
    parts = ["labels" if p == "images" else p for p in parts]
    return Path(*parts).with_suffix(".txt")


def draw(image_path: str | Path, out_path: str | Path | None = None) -> int:
    image_path = Path(image_path)
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(image_path)
    h, w = img.shape[:2]
    lbl = label_path_for(image_path)
    polys = []
    if lbl.is_file():
        for line in lbl.read_text().splitlines():
            vals = line.split()
            if len(vals) < 7:
                continue
            xy = np.array(vals[1:], dtype=float).reshape(-1, 2)
            xy[:, 0] *= w
            xy[:, 1] *= h
            polys.append(xy.astype(np.int32))

    overlay = img.copy()
    for i, p in enumerate(polys):
        col = ((37 * i) % 255, (91 * i + 60) % 255, (140 * i + 30) % 255)
        cv2.fillPoly(overlay, [p], col)
        cv2.polylines(img, [p], True, (0, 255, 0), 1)
    img = cv2.addWeighted(overlay, 0.4, img, 0.6, 0)
    cv2.putText(img, f"pods: {len(polys)}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4)
    cv2.putText(img, f"pods: {len(polys)}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

    if out_path:
        cv2.imwrite(str(out_path), img)
        print(f"Wrote {out_path}  ({len(polys)} pods)")
    return len(polys)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    draw(a.image, a.out)

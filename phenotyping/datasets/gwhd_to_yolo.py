"""
gwhd_to_yolo.py — convert Global Wheat Head Detection (GWHD 2021) annotations to
Ultralytics YOLO-detection format.

Trains a cereal-spike detector that the shared engine (phenotyping.object_count,
detection path) then runs — the same weights are the warm-start for barley spikes.

GWHD 2021 ships a CSV with columns: image_name, BoxesString, domain
  BoxesString = ";"-joined "x1 y1 x2 y2" (pixels), or "no_box". Images are 1024².
(If your GWHD download uses different column names, --image-col / --box-col override.)

Output: <out>/{images,labels}/{train,val}/ + <out>/gwhd.yaml (single class "spike").

Run (Colab):
  python -m phenotyping.datasets.gwhd_to_yolo \
    --csv competition_train.csv --images images/ --out gwhd_yolo --val-frac 0.15
  yolo detect train data=gwhd_yolo/gwhd.yaml model=yolo11s.pt imgsz=1024 epochs=50
"""
import argparse
import csv
import random
import shutil
from pathlib import Path

import cv2

CLASS_NAME = "spike"   # cereal head/spike — same class barley reuses


def parse_boxes(s):
    """GWHD BoxesString -> list of (x1,y1,x2,y2) pixel boxes."""
    s = (s or "").strip()
    if not s or s.lower() in ("no_box", "nan"):
        return []
    out = []
    for chunk in s.split(";"):
        p = chunk.split()
        if len(p) == 4:
            try:
                out.append(tuple(float(v) for v in p))
            except ValueError:
                pass
    return out


def xyxy_to_yolo(b, W, H):
    x1, y1, x2, y2 = b
    return ((x1 + x2) / 2 / W, (y1 + y2) / 2 / H, abs(x2 - x1) / W, abs(y2 - y1) / H)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--images", required=True, help="dir with GWHD images")
    ap.add_argument("--out", required=True)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--image-col", default="image_name")
    ap.add_argument("--box-col", default="BoxesString")
    ap.add_argument("--copy", action="store_true", help="copy images (default: symlink)")
    args = ap.parse_args()

    with open(args.csv, newline="") as f:
        rows = list(csv.DictReader(f))
    random.seed(args.seed)
    random.shuffle(rows)
    n_val = int(len(rows) * args.val_frac)
    splits = {"val": rows[:n_val], "train": rows[n_val:]}

    out = Path(args.out)
    images_dir = Path(args.images)
    n_img = n_box = 0
    for split, srows in splits.items():
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)
        for r in srows:
            name = r.get(args.image_col)
            src = images_dir / name if name else None
            if not src or not src.is_file():
                continue
            img = cv2.imread(str(src))
            if img is None:
                continue
            H, W = img.shape[:2]
            boxes = parse_boxes(r.get(args.box_col))
            with open(out / "labels" / split / (Path(name).stem + ".txt"), "w") as lf:
                for b in boxes:
                    cx, cy, w, h = xyxy_to_yolo(b, W, H)
                    lf.write(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
                    n_box += 1
            dst = out / "images" / split / name
            if args.copy:
                shutil.copyfile(src, dst)
            else:
                try:
                    if not dst.exists():
                        dst.symlink_to(src.resolve())
                except OSError:
                    shutil.copyfile(src, dst)
            n_img += 1

    (out / "gwhd.yaml").write_text(
        f"path: {out.resolve()}\ntrain: images/train\nval: images/val\n"
        f"names:\n  0: {CLASS_NAME}\n")
    print(f"wrote {n_img} images, {n_box} boxes -> {out}")
    print(f"train: yolo detect train data={out / 'gwhd.yaml'} "
          f"model=yolo11s.pt imgsz=1024 epochs=50")


if __name__ == "__main__":
    main()

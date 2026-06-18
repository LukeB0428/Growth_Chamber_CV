"""
count_tool.py — click-to-count helper for EXHAUSTIVE manual pod counts.

deepcanola's CSV counts are a capped measurement-subset (~20), so they can't
validate true count accuracy. This tool makes a trustworthy ground truth: you
click every pod in a scan and it tallies + writes the CSV the eval consumes.

Controls (per image):
  LEFT-CLICK            drop a dot on a pod (tally goes up)
  u  /  RIGHT-CLICK     undo last dot
  r                     reset this image to 0
  n / SPACE / ENTER     save this image's count, go to next
  q / ESC               save progress and quit

Output CSV (image,gt_count) is written after EVERY image, so quitting keeps
progress and re-running RESUMES (already-counted images are skipped). Feed it to
the eval as the --gt file:
    python -m brassica_pods.eval.evaluate \
        --weights brassica_pods/weights/pods_best.pt \
        --images "<novel images dir>" \
        --gt brassica_pods/data/handcount_gt.csv \
        --out brassica_pods/data/eval_handcount

Needs a display — run on the laptop (like scripts/calibrate_pots.py), not headless.

Run:
    python -m brassica_pods.eval.count_tool \
        --images "C:/Users/LukeB/brassica_data/deepcanola/extracted/novel_data/br017/images" \
        --list brassica_pods/data/handcount_sample.txt \
        --out brassica_pods/data/handcount_gt.csv
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

import cv2

DISPLAY_H = 1000          # on-screen height; counting accuracy needs it large
WIN = "pod count  (click=pod  u/right=undo  r=reset  n=next  q=quit)"


def _load_done(out_path) -> dict:
    done = {}
    if Path(out_path).is_file():
        for r in csv.DictReader(open(out_path, newline="", encoding="utf-8")):
            done[r["image"]] = int(r["gt_count"])
    return done


def _save(out_path, counts: dict) -> None:
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["image", "gt_count"])
        for k, v in counts.items():
            w.writerow([k, v])


def count_one(path: str, name: str):
    """Return the clicked count for one image, or -1 to quit, or None if unreadable."""
    img = cv2.imread(path)
    if img is None:
        return None
    h, w = img.shape[:2]
    scale = DISPLAY_H / h
    disp = cv2.resize(img, (int(w * scale), DISPLAY_H))
    pts: list = []

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            pts.append((x, y))
        elif event == cv2.EVENT_RBUTTONDOWN and pts:
            pts.pop()

    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(WIN, on_mouse)
    while True:
        vis = disp.copy()
        for (x, y) in pts:
            cv2.circle(vis, (x, y), 5, (0, 0, 255), -1)
            cv2.circle(vis, (x, y), 6, (255, 255, 255), 1)
        cv2.rectangle(vis, (0, 0), (vis.shape[1], 30), (0, 0, 0), -1)
        cv2.putText(vis, f"{name}    count = {len(pts)}", (8, 21),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.imshow(WIN, vis)
        k = cv2.waitKey(20) & 0xFF
        if k in (ord("n"), ord(" "), 13):       # next
            return len(pts)
        if k == ord("u") and pts:               # undo
            pts.pop()
        if k == ord("r"):                        # reset
            pts.clear()
        if k in (ord("q"), 27):                  # quit
            return -1


def main():
    ap = argparse.ArgumentParser(description="Click-to-count manual pod ground truth")
    ap.add_argument("--images", required=True, help="dir of scans")
    ap.add_argument("--list", default=None, help="text file of image names (else all in dir)")
    ap.add_argument("--out", required=True, help="output CSV (image,gt_count)")
    a = ap.parse_args()

    if a.list:
        names = [l.strip() for l in open(a.list, encoding="utf-8") if l.strip()]
    else:
        names = sorted(n for n in os.listdir(a.images)
                       if n.lower().endswith((".jpg", ".jpeg", ".png")))

    counts = _load_done(a.out)
    todo = [n for n in names if n not in counts]
    print(f"{len(counts)} already counted, {len(todo)} to go.")

    for name in todo:
        c = count_one(os.path.join(a.images, name), name)
        if c is None:
            print(f"  skip (unreadable): {name}")
            continue
        if c == -1:
            print("Quit — progress saved.")
            break
        counts[name] = c
        _save(a.out, counts)
        print(f"  {name}: {c}   ({len(counts)}/{len(names)} done)")

    cv2.destroyAllWindows()
    print(f"Wrote {a.out}  ({len(counts)} images).")


if __name__ == "__main__":
    main()

"""
extract_komatsuna_hsv.py — Extract green-leaf HSV bounds from the KOMATSUNA dataset
EE496 | Luke Buckley | Maynooth University

Works with the DatasetNinja tar download (no segmentation masks required).
The KOMATSUNA hydroponic background is white, so green pixels = leaves.

Usage:
    python extract_komatsuna_hsv.py --tar C:/Users/LukeB/Downloads/komatsuna-DatasetNinja.tar

Options:
    --tar PATH       Path to komatsuna-DatasetNinja.tar
    --max-images N   Max images to sample (default 150, ~0 for all)
    --low-pct  F     Lower percentile for bounds (default 2.0)
    --high-pct F     Upper percentile for bounds (default 98.0)

Output:
    Prints recommended hsv_lower / hsv_upper for brassica.json segmentation.
    Saves hsv_distribution_komatsuna.csv alongside this script.
"""

import argparse
import csv
import io
import os
import sys
import tarfile

import cv2
import numpy as np


# Broad pre-filter to isolate "plant-like" pixels from white background.
# White background: high V, low S — plants: moderate-high S, green H.
# This captures all leaf pixels and almost no background.
PREFILTER_HSV_LOWER = np.array([15,  30,  25], dtype=np.uint8)
PREFILTER_HSV_UPPER = np.array([110, 255, 255], dtype=np.uint8)
MIN_PREFILTER_PIXELS = 500   # skip near-empty images (between-plant shots)


def extract_green_pixels(img_bytes: bytes) -> np.ndarray | None:
    """Return Nx3 (H,S,V) array of foreground leaf pixels from a JPEG/PNG byte buffer."""
    arr = np.frombuffer(img_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return None

    # Downsample large images for speed (multi-view are 2048×1536)
    h, w = img.shape[:2]
    if max(h, w) > 800:
        scale = 800 / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, PREFILTER_HSV_LOWER, PREFILTER_HSV_UPPER)

    pixel_count = int(mask.sum() / 255)
    if pixel_count < MIN_PREFILTER_PIXELS:
        return None

    return hsv[mask > 0]


def percentile_bounds(values: np.ndarray, low: float, high: float) -> tuple[int, int]:
    return int(np.percentile(values, low)), int(np.percentile(values, high))


def main():
    parser = argparse.ArgumentParser(description='Extract brassica leaf HSV from KOMATSUNA tar')
    parser.add_argument('--tar',        required=True, help='Path to komatsuna-DatasetNinja.tar')
    parser.add_argument('--max-images', type=int, default=150,
                        help='Max RGB images to sample (0 = all, default 150)')
    parser.add_argument('--low-pct',    type=float, default=2.0)
    parser.add_argument('--high-pct',   type=float, default=98.0)
    args = parser.parse_args()

    if not os.path.isfile(args.tar):
        print(f"Error: {args.tar} not found", file=sys.stderr)
        sys.exit(1)

    print(f"Opening {args.tar}...")
    print(f"Sampling up to {args.max_images or 'all'} multi-view RGB images.")
    print("(Background separation only — no mask files needed)")

    all_hsv = []
    processed = 0
    skipped = 0

    with tarfile.open(args.tar, 'r') as tf:
        members = [m for m in tf.getmembers()
                   if m.name.startswith('multi-view/img/') and m.name.endswith('.png') and m.size > 0]

        if args.max_images and args.max_images < len(members):
            # Spread samples evenly across the time-series rather than just taking first N
            step = len(members) // args.max_images
            members = members[::step][:args.max_images]

        print(f"Selected {len(members)} images from {args.tar}")

        for m in members:
            f = tf.extractfile(m)
            if f is None:
                continue
            img_bytes = f.read()
            hsv_pixels = extract_green_pixels(img_bytes)
            if hsv_pixels is None:
                skipped += 1
                continue
            all_hsv.append(hsv_pixels)
            processed += 1
            if processed % 25 == 0:
                print(f"  Processed {processed}/{len(members)} ({skipped} skipped — too sparse)...")

    if not all_hsv:
        print("No usable images found. Check tar path or try --max-images 0 for all images.")
        sys.exit(1)

    combined = np.vstack(all_hsv)
    H = combined[:, 0]
    S = combined[:, 1]
    V = combined[:, 2]

    print(f"\nImages processed: {processed}  |  Leaf pixels analysed: {len(combined):,}")

    h_low,  h_high  = percentile_bounds(H, args.low_pct, args.high_pct)
    s_low,  s_high  = percentile_bounds(S, args.low_pct, args.high_pct)
    v_low,  v_high  = percentile_bounds(V, args.low_pct, args.high_pct)

    print("\n--- HSV distribution (OpenCV: H 0-179, S/V 0-255) ---")
    print(f"  Hue:        mean={H.mean():.1f}  std={H.std():.1f}  "
          f"p{args.low_pct:.0f}={h_low}  p{args.high_pct:.0f}={h_high}")
    print(f"  Saturation: mean={S.mean():.1f}  std={S.std():.1f}  "
          f"p{args.low_pct:.0f}={s_low}  p{args.high_pct:.0f}={s_high}")
    print(f"  Value:      mean={V.mean():.1f}  std={V.std():.1f}  "
          f"p{args.low_pct:.0f}={v_low}  p{args.high_pct:.0f}={v_high}")

    # Safety margin: loosen lower bounds slightly to avoid clipping edge/shadowed leaves
    h_lo_safe = max(0,  h_low  - 5)
    s_lo_safe = max(0,  s_low  - 15)
    v_lo_safe = max(0,  v_low  - 15)

    print("\n--- Recommended brassica.json segmentation bounds ---")
    print(f'  "hsv_lower": [{h_lo_safe}, {s_lo_safe}, {v_lo_safe}],')
    print(f'  "hsv_upper": [{h_high}, 255, 255],')
    print()
    print("Notes:")
    print(f"  S/V upper left at 255 — over-bright leaf pixels are still leaves.")
    print(f"  Hue range {h_lo_safe}–{h_high} in OpenCV = {h_lo_safe*2}–{h_high*2}° standard.")
    print(f"  Source: KOMATSUNA dataset (Uchiyama et al. 2017, ICCVW)")

    out_csv = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           'hsv_distribution_komatsuna.csv')
    with open(out_csv, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['channel', 'mean', 'std', 'p2', 'p5', 'p25', 'p50', 'p75', 'p95', 'p98'])
        for name, arr in [('H', H), ('S', S), ('V', V)]:
            w.writerow([name, f'{arr.mean():.2f}', f'{arr.std():.2f}',
                        int(np.percentile(arr, 2)),  int(np.percentile(arr, 5)),
                        int(np.percentile(arr, 25)), int(np.percentile(arr, 50)),
                        int(np.percentile(arr, 75)), int(np.percentile(arr, 95)),
                        int(np.percentile(arr, 98))])
    print(f"\nFull distribution saved: {out_csv}")
    print("Paste hsv_lower / hsv_upper into brassica.json segmentation section.")


if __name__ == '__main__':
    main()

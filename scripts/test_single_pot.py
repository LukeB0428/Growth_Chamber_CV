"""
test_single_pot.py — Single-Pot Pipeline Test for Growth Chamber CV
EE496 | Luke Buckley | Maynooth University

Runs the full per-pot metric pipeline (analyse_pot) on a standalone image
without requiring a calibration JSON or the whole-chamber orchestration in
analyse_chamber.py. Useful for verifying that health metrics, leaf counting,
and bolting detection all work on a cropped pot image before a full run.

The pot ROI is assumed to be the full image area — the circle is centred
at the image midpoint with radius = half the shorter side.

Usage:
    python test_single_pot.py --image path/to/pot_image.jpg
    python test_single_pot.py --image path/to/pot_image.jpg --method model
"""
import sys, argparse
import cv2
import numpy as np
from pathlib import Path

# Ensure scripts directory is on path
sys.path.insert(0, str(Path(__file__).parent))

from analyse_chamber import analyse_pot, make_pot_mask

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', required=True, help='Path to single pot image')
    parser.add_argument('--method', default='hsv', choices=['hsv', 'model'])
    args = parser.parse_args()

    image_path = Path(args.image)
    if not image_path.exists():
        print(f"Image not found: {image_path}")
        sys.exit(1)

    image = cv2.imread(str(image_path))
    if image is None:
        print(f"Could not read image: {image_path}")
        sys.exit(1)

    h, w = image.shape[:2]
    # Pot covers the full image — circle centred, radius = half the shorter side
    pot = {
        'label': image_path.stem,
        'x': w // 2,
        'y': h // 2,
        'r': min(w, h) // 2,
    }

    print(f"\nImage : {image_path.name}  ({w}x{h})")
    print(f"Pot ROI : centre=({pot['x']},{pot['y']})  r={pot['r']}")
    print(f"Method  : {args.method}\n")

    metrics = analyse_pot(
        image      = image,
        depth_map  = None,
        pot        = pot,
        chamber_id = 'enriched',
        image_path = image_path,
        method     = args.method,
        run_health = True,
        run_leaf_count = True,
        run_bolting    = True,
    )

    print("\n-- Results --------------------------------------")
    for key, val in metrics.items():
        print(f"  {key:<25} {val}")

if __name__ == '__main__':
    main()

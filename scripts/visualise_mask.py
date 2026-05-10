"""
visualise_mask.py — Green Canopy Mask Verification for Growth Chamber CV
EE496 | Luke Buckley | Maynooth University

Overlays the detected green canopy mask on the original image for visual
verification of HSV threshold accuracy. Use this when tuning hue bounds
or checking why canopy cover looks wrong on a particular image.

Outputs a 3-panel figure:
  Left   — Original image
  Centre — Binary green mask (white = canopy detected)
  Right  — Original image with green mask overlay (semi-transparent green tint)

Saves the figure to: results/mask_visualisation_<timestamp>.png

Usage:
    python visualise_mask.py --image path/to/image.jpg
    python visualise_mask.py --image path/to/image.jpg --hue_low 25 --hue_high 90
    python visualise_mask.py --image path/to/image.jpg --denoise --kernel_size 7
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
from datetime import datetime
from config import IMAGES_DIR, RESULTS_DIR

# ── Default paths ─────────────────────────────────────────────────────────────
DEFAULT_IMAGE = str(IMAGES_DIR / "test_image.jpg")
RESULTS_DIR   = str(RESULTS_DIR)

# ── HSV thresholds — match your analyse_image.py values ──────────────────────
DEFAULT_HUE_LOW  = 25
DEFAULT_HUE_HIGH = 90
SAT_LOW  = 40    # ignore very desaturated (near-grey) pixels
VAL_LOW  = 40    # ignore very dark pixels


def build_green_mask(hsv, hue_low, hue_high):
    lower = np.array([hue_low, SAT_LOW, VAL_LOW])
    upper = np.array([hue_high, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)
    return mask


def denoise_mask(mask, kernel_size=5):
    """
    Morphological opening: erode then dilate.
    Removes small isolated noise blobs while preserving the main plant shape.
    kernel_size controls aggressiveness — increase to remove larger speckles.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return mask


def visualise(image_path, hue_low, hue_high, denoise=False, kernel_size=5, save=True):
    # Load image
    bgr = cv2.imread(image_path)
    if bgr is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    # Build mask
    mask_raw = build_green_mask(hsv, hue_low, hue_high)
    mask = denoise_mask(mask_raw, kernel_size) if denoise else mask_raw

    # Canopy cover %
    canopy_px   = np.count_nonzero(mask)
    total_px    = mask.size
    cover_pct   = (canopy_px / total_px) * 100

    # Overlay: tint detected canopy green on original image
    overlay = rgb.copy()
    green_tint = np.zeros_like(rgb)
    green_tint[:, :] = (0, 200, 80)
    alpha = 0.45
    mask_bool = mask.astype(bool)
    overlay[mask_bool] = (
        (1 - alpha) * rgb[mask_bool] + alpha * green_tint[mask_bool]
    ).astype(np.uint8)

    # ── Plot ─────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    denoise_label = f"  |  Denoised (kernel={kernel_size})" if denoise else ""
    fig.suptitle(
        f"Green Mask Verification  |  Hue range: {hue_low}–{hue_high}  |  "
        f"Canopy Cover: {cover_pct:.2f}%{denoise_label}",
        fontsize=13, fontweight="bold"
    )

    axes[0].imshow(rgb)
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    axes[1].imshow(mask, cmap="gray")
    axes[1].set_title("Green Mask  (white = canopy detected)")
    axes[1].axis("off")

    axes[2].imshow(overlay)
    axes[2].set_title("Overlay  (green tint = detected canopy)")
    axes[2].axis("off")

    plt.tight_layout()

    if save:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = os.path.join(RESULTS_DIR, f"mask_visualisation_{timestamp}.png")
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {out_path}")

    plt.show()
    print(f"Canopy Cover: {cover_pct:.2f}%  ({canopy_px} / {total_px} pixels)")


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualise green canopy mask.")
    parser.add_argument("--image",       default=DEFAULT_IMAGE, help="Path to input image")
    parser.add_argument("--hue_low",     type=int, default=DEFAULT_HUE_LOW,  help="Lower hue bound (0-179)")
    parser.add_argument("--hue_high",    type=int, default=DEFAULT_HUE_HIGH, help="Upper hue bound (0-179)")
    parser.add_argument("--denoise",     action="store_true", help="Apply morphological opening to remove soil speckle noise")
    parser.add_argument("--kernel_size", type=int, default=5, help="Denoising kernel size (default 5, increase for larger speckles)")
    parser.add_argument("--no_save",     action="store_true", help="Don't save output image")
    args = parser.parse_args()

    visualise(args.image, args.hue_low, args.hue_high, denoise=args.denoise, kernel_size=args.kernel_size, save=not args.no_save)

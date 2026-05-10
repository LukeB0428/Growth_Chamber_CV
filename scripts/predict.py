"""
predict.py — Single Image Prediction for Growth Chamber Project
EE496 | Luke Buckley | Maynooth University

Loads the trained U-Net model and runs it on a single input image,
returning a binary segmentation mask. This script is the bridge between
the trained model and the analysis pipeline (analyse_image.py).

Can be used in two ways:

1. Standalone — run directly to test on any image:
       python predict.py --image path/to/image.jpg

2. As a module — imported by analyse_image.py to replace HSV thresholding:
       from predict import get_model_mask
       mask = get_model_mask(image_bgr)

Outputs (standalone mode):
    prediction_result.png  — side by side: original image, mask, overlay
"""

import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
import argparse
import os
from config import MODEL_PATH, RESULTS_DIR

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

MODEL_PATH = str(MODEL_PATH)
PLOT_PATH  = str(RESULTS_DIR / "prediction_result.png")
IMAGE_SIZE = 256
THRESHOLD  = 0.5

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Preprocessing transform — same normalisation used during training
preprocess = A.Compose([
    A.Resize(IMAGE_SIZE, IMAGE_SIZE),
    A.Normalize(mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])


# ─────────────────────────────────────────────
# MODEL LOADER (cached — only loads once)
# ─────────────────────────────────────────────

_model = None  # Module-level cache so model is only loaded once per session

def load_model():
    global _model
    if _model is not None:
        return _model

    model = smp.Unet(
        encoder_name    = "resnet34",
        encoder_weights = None,
        in_channels     = 3,
        classes         = 1,
    ).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    _model = model
    print(f"Model loaded from {MODEL_PATH}")
    return model


# ─────────────────────────────────────────────
# CORE PREDICTION FUNCTION
# ─────────────────────────────────────────────

def get_model_mask(image_bgr):
    """
    Takes a BGR image (as loaded by OpenCV) and returns a binary mask
    of the same original dimensions as a numpy uint8 array.

    Mask values: 255 = plant, 0 = background
    (Same format as the HSV green_mask in analyse_image.py)

    Args:
        image_bgr: numpy array (H, W, 3) in BGR format

    Returns:
        mask: numpy array (H, W) uint8, values 0 or 255
    """
    original_h, original_w = image_bgr.shape[:2]

    # Convert BGR to RGB for the model
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    # Preprocess — resize to 256x256 and normalise
    augmented = preprocess(image=image_rgb)
    tensor    = augmented["image"].unsqueeze(0).to(DEVICE)  # Add batch dim

    # Run model
    model = load_model()
    with torch.no_grad():
        output = model(tensor)
        prob   = torch.sigmoid(output).squeeze().cpu().numpy()  # (256, 256)

    # Threshold to binary
    binary_mask = (prob > THRESHOLD).astype(np.uint8) * 255  # 0 or 255

    # Resize back to original image dimensions
    mask_resized = cv2.resize(
        binary_mask,
        (original_w, original_h),
        interpolation=cv2.INTER_NEAREST  # Nearest neighbour keeps mask binary
    )

    return mask_resized


# ─────────────────────────────────────────────
# STANDALONE VISUALISATION
# ─────────────────────────────────────────────

def visualise_prediction(image_path):
    """Load an image, run prediction, and display the result."""
    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        print(f"Error: Could not load image from {image_path}")
        return

    print(f"Running prediction on: {os.path.basename(image_path)}")
    mask = get_model_mask(image_bgr)

    # Convert for display
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    # Build overlay — green tint on predicted plant pixels
    overlay = image_rgb.copy().astype(float) / 255.0
    plant   = mask > 127
    overlay[plant, 0] = overlay[plant, 0] * 0.4
    overlay[plant, 1] = overlay[plant, 1] * 0.4 + 0.6
    overlay[plant, 2] = overlay[plant, 2] * 0.4
    overlay = (overlay.clip(0, 1) * 255).astype(np.uint8)

    # Canopy cover from model mask (for comparison with HSV method)
    canopy_cover = (np.sum(mask > 127) / mask.size) * 100
    print(f"Model canopy cover: {canopy_cover:.2f}%")

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    fig.suptitle(
        f"{os.path.basename(image_path)}  |  Model Canopy Cover: {canopy_cover:.2f}%",
        fontsize=12, fontweight="bold"
    )

    axes[0].imshow(image_rgb)
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    axes[1].imshow(mask, cmap="gray")
    axes[1].set_title("Predicted Mask")
    axes[1].axis("off")

    axes[2].imshow(overlay)
    axes[2].set_title("Overlay")
    axes[2].axis("off")

    plt.tight_layout()
    os.makedirs(os.path.dirname(PLOT_PATH), exist_ok=True)
    plt.savefig(PLOT_PATH, dpi=100, bbox_inches="tight")
    plt.show()
    print(f"Saved to {PLOT_PATH}")


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run U-Net prediction on a single image.")
    parser.add_argument(
        "--image",
        default=None,
        help="Path to the image to run prediction on"
    )
    args = parser.parse_args()
    visualise_prediction(args.image)

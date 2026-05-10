"""
evaluate.py — Model Evaluation and Visualisation for Growth Chamber Project
EE496 | Luke Buckley | Maynooth University

Loads the trained U-Net model (best_model.pth) and runs it on a sample of
validation images from the CVPPP A1 dataset. For each image it shows:
  - The original RGB image
  - The ground truth mask (from the dataset)
  - The model's predicted mask
  - An overlay of the prediction on the original image

Also prints the mean IoU score across all validation samples.

Usage:
    python evaluate.py
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp
from torch.utils.data import random_split
from dataset import CVPPPDataset, val_transform, DATASET_PATH
import os
from config import MODEL_PATH, RESULTS_DIR

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

MODEL_PATH  = str(MODEL_PATH)
PLOT_PATH   = str(RESULTS_DIR / "evaluation.png")

TRAIN_SPLIT = 0.8
SEED        = 42
NUM_SAMPLES = 6      # Number of validation images to visualise
THRESHOLD   = 0.5    # Sigmoid threshold for binary mask

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def load_model():
    model = smp.Unet(
        encoder_name    = "resnet34",
        encoder_weights = None,   # Don't re-download weights, we load our own
        in_channels     = 3,
        classes         = 1,
    ).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    print(f"Loaded model from {MODEL_PATH}")
    return model


def get_val_subset():
    full_dataset = CVPPPDataset(DATASET_PATH, transform=None)
    total   = len(full_dataset)
    n_train = int(total * TRAIN_SPLIT)
    n_val   = total - n_train

    generator = torch.Generator().manual_seed(SEED)
    _, val_idx = random_split(range(total), [n_train, n_val], generator=generator)

    val_dataset = CVPPPDataset(DATASET_PATH, transform=val_transform)
    return torch.utils.data.Subset(val_dataset, val_idx.indices)


def iou_score(pred, target, threshold=THRESHOLD):
    pred      = (torch.sigmoid(pred) > threshold).float()
    intersect = (pred * target).sum()
    union     = pred.sum() + target.sum() - intersect
    if union == 0:
        return 1.0
    return (intersect / union).item()


def denormalise(tensor):
    """Convert normalised image tensor back to displayable RGB array."""
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])
    img  = tensor.permute(1, 2, 0).numpy()
    img  = (img * std + mean).clip(0, 1)
    return img


# ─────────────────────────────────────────────
# EVALUATION
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print(f"Device: {DEVICE}")

    model      = load_model()
    val_subset = get_val_subset()
    print(f"Validation set: {len(val_subset)} samples")

    # --- Compute mean IoU across full validation set ---
    all_ious = []
    with torch.no_grad():
        for image, mask in val_subset:
            image_batch = image.unsqueeze(0).to(DEVICE)  # Add batch dim
            output      = model(image_batch)
            iou         = iou_score(output.squeeze(0), mask)
            all_ious.append(iou)

    mean_iou = np.mean(all_ious)
    print(f"\nMean IoU on validation set: {mean_iou:.4f}")
    print(f"Min IoU: {min(all_ious):.4f} | Max IoU: {max(all_ious):.4f}")

    # --- Visualise NUM_SAMPLES predictions ---
    # Pick evenly spaced samples from the validation set for variety
    indices = np.linspace(0, len(val_subset) - 1, NUM_SAMPLES, dtype=int)

    fig, axes = plt.subplots(NUM_SAMPLES, 4, figsize=(16, NUM_SAMPLES * 3.5))
    fig.suptitle(
        f"U-Net Predictions on CVPPP A1 Validation Set  |  Mean IoU: {mean_iou:.4f}",
        fontsize=13, fontweight="bold"
    )

    col_titles = ["Original Image", "Ground Truth Mask", "Predicted Mask", "Overlay"]
    for col, title in enumerate(col_titles):
        axes[0, col].set_title(title, fontsize=11, fontweight="bold")

    with torch.no_grad():
        for row, idx in enumerate(indices):
            image, mask = val_subset[idx]

            # Run model
            image_batch = image.unsqueeze(0).to(DEVICE)
            output      = model(image_batch).squeeze(0)
            pred_mask   = (torch.sigmoid(output) > THRESHOLD).float()
            sample_iou  = iou_score(output, mask)

            # Convert for display
            img_display  = denormalise(image)
            gt_display   = mask.squeeze().numpy()
            pred_display = pred_mask.squeeze().cpu().numpy()

            # Overlay: green tint on predicted plant pixels
            overlay = img_display.copy()
            plant_pixels = pred_display > 0.5
            overlay[plant_pixels, 0] = overlay[plant_pixels, 0] * 0.4          # reduce red
            overlay[plant_pixels, 1] = overlay[plant_pixels, 1] * 0.4 + 0.6   # boost green
            overlay[plant_pixels, 2] = overlay[plant_pixels, 2] * 0.4          # reduce blue
            overlay = overlay.clip(0, 1)

            # Plot
            axes[row, 0].imshow(img_display)
            axes[row, 0].set_ylabel(f"Sample {idx}\nIoU: {sample_iou:.3f}", fontsize=9)

            axes[row, 1].imshow(gt_display, cmap="gray")
            axes[row, 2].imshow(pred_display, cmap="gray")
            axes[row, 3].imshow(overlay)

            for col in range(4):
                axes[row, col].axis("off")

    plt.tight_layout()
    os.makedirs(os.path.dirname(PLOT_PATH), exist_ok=True)
    plt.savefig(PLOT_PATH, dpi=100, bbox_inches="tight")
    plt.show()
    print(f"\nEvaluation plot saved to {PLOT_PATH}")

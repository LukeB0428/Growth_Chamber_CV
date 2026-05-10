"""
train.py — U-Net Training Script for Growth Chamber Project
EE496 | Luke Buckley | Maynooth University

Trains a U-Net segmentation model with a pretrained ResNet34 encoder
on the CVPPP 2014 A1 Arabidopsis dataset.

The model learns to segment plant rosettes from background in top-down
RGB images. Once trained on CVPPP data it can be fine-tuned on chamber
images collected during the Arabidopsis trial.

Usage:
    python train.py

Outputs:
    best_model.pth       — saved weights of the best model (lowest val loss)
    training_curves.png  — plot of train/val loss and IoU over epochs
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import segmentation_models_pytorch as smp
import numpy as np
import matplotlib.pyplot as plt
from dataset import CVPPPDataset, train_transform, val_transform, DATASET_PATH
import os
from config import MODEL_PATH, RESULTS_DIR

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

EPOCHS      = 50
BATCH_SIZE  = 4
LR          = 1e-4        # Learning rate
TRAIN_SPLIT = 0.8
SEED        = 42

MODEL_SAVE_PATH = str(MODEL_PATH)
PLOT_SAVE_PATH  = str(RESULTS_DIR / "training_curves.png")

# Use GPU if available, otherwise CPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────

def get_dataloaders():
    full_dataset = CVPPPDataset(DATASET_PATH, transform=None)
    total   = len(full_dataset)
    n_train = int(total * TRAIN_SPLIT)
    n_val   = total - n_train

    generator = torch.Generator().manual_seed(SEED)
    train_idx, val_idx = random_split(range(total), [n_train, n_val], generator=generator)

    train_dataset = CVPPPDataset(DATASET_PATH, transform=train_transform)
    val_dataset   = CVPPPDataset(DATASET_PATH, transform=val_transform)

    train_subset = torch.utils.data.Subset(train_dataset, train_idx.indices)
    val_subset   = torch.utils.data.Subset(val_dataset,   val_idx.indices)

    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_subset,   batch_size=BATCH_SIZE, shuffle=False)

    print(f"Train: {len(train_subset)} samples | Val: {len(val_subset)} samples")
    return train_loader, val_loader


# ─────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────

def iou_score(pred, target, threshold=0.5):
    """
    Intersection over Union (IoU) — the standard metric for segmentation.
    A score of 1.0 means perfect overlap between predicted and true mask.
    A score of 0.0 means no overlap at all.
    We threshold the sigmoid output at 0.5 to get a binary prediction.
    """
    pred   = (torch.sigmoid(pred) > threshold).float()
    intersect = (pred * target).sum()
    union     = pred.sum() + target.sum() - intersect
    if union == 0:
        return torch.tensor(1.0)
    return (intersect / union).item()


# ─────────────────────────────────────────────
# TRAINING AND VALIDATION LOOPS
# ─────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0.0
    total_iou  = 0.0

    for images, masks in loader:
        images, masks = images.to(DEVICE), masks.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss    = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_iou  += iou_score(outputs, masks)

    n = len(loader)
    return total_loss / n, total_iou / n


def validate(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    total_iou  = 0.0

    with torch.no_grad():
        for images, masks in loader:
            images, masks = images.to(DEVICE), masks.to(DEVICE)
            outputs = model(images)
            loss    = criterion(outputs, masks)

            total_loss += loss.item()
            total_iou  += iou_score(outputs, masks)

    n = len(loader)
    return total_loss / n, total_iou / n


# ─────────────────────────────────────────────
# PLOT TRAINING CURVES
# ─────────────────────────────────────────────

def plot_curves(train_losses, val_losses, train_ious, val_ious):
    epochs = range(1, len(train_losses) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(epochs, train_losses, label="Train Loss")
    ax1.plot(epochs, val_losses,   label="Val Loss")
    ax1.set_title("Loss over Epochs")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("BCE Loss")
    ax1.legend()
    ax1.grid(True)

    ax2.plot(epochs, train_ious, label="Train IoU")
    ax2.plot(epochs, val_ious,   label="Val IoU")
    ax2.set_title("IoU Score over Epochs")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("IoU")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    os.makedirs(os.path.dirname(PLOT_SAVE_PATH), exist_ok=True)
    plt.savefig(PLOT_SAVE_PATH, dpi=100)
    plt.show()
    print(f"Training curves saved to {PLOT_SAVE_PATH}")


# ─────────────────────────────────────────────
# MAIN TRAINING LOOP
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print(f"Device: {DEVICE}")
    print(f"Epochs: {EPOCHS} | Batch size: {BATCH_SIZE} | LR: {LR}")
    print()

    # --- Data ---
    train_loader, val_loader = get_dataloaders()

    # --- Model ---
    # U-Net with a ResNet34 encoder pretrained on ImageNet.
    # The encoder already knows how to detect edges, textures and shapes.
    # We only need to fine-tune it to recognise Arabidopsis rosettes specifically.
    # encoder_weights="imagenet" downloads pretrained weights automatically.
    model = smp.Unet(
        encoder_name    = "resnet34",
        encoder_weights = "imagenet",
        in_channels     = 3,       # RGB input
        classes         = 1,       # Binary output: plant vs background
    ).to(DEVICE)

    print("Model: U-Net with ResNet34 encoder (pretrained on ImageNet)")
    print()

    # --- Loss and Optimiser ---
    # BCEWithLogitsLoss combines sigmoid + binary cross entropy in one step,
    # which is numerically more stable than applying them separately.
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    # Learning rate scheduler — reduces LR by half if val loss stops improving
    # for 5 epochs. Helps the model converge more precisely in later epochs.
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )

    # --- Training ---
    best_val_loss  = float("inf")
    train_losses, val_losses = [], []
    train_ious,   val_ious   = [], []

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_iou = train_one_epoch(model, train_loader, optimizer, criterion)
        val_loss,   val_iou   = validate(model, val_loader, criterion)

        scheduler.step(val_loss)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_ious.append(train_iou)
        val_ious.append(val_iou)

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            saved_marker = " ← saved"
        else:
            saved_marker = ""

        print(f"Epoch {epoch:02d}/{EPOCHS} | "
              f"Train Loss: {train_loss:.4f}  IoU: {train_iou:.4f} | "
              f"Val Loss: {val_loss:.4f}  IoU: {val_iou:.4f}"
              f"{saved_marker}")

    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")
    print(f"Best model saved to: {MODEL_SAVE_PATH}")

    # --- Plot ---
    plot_curves(train_losses, val_losses, train_ious, val_ious)

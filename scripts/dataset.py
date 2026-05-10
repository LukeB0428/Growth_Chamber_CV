"""
dataset.py — CVPPP A1 Dataset Loader for Growth Chamber Project
EE496 | Luke Buckley | Maynooth University

Loads the CVPPP 2014 A1 Arabidopsis dataset (plant RGB images + label masks)
into a PyTorch Dataset ready for training a U-Net segmentation model.

Each sample returns:
  - image  : (3, H, W) float tensor, pixel values normalised to [0, 1]
  - mask   : (1, H, W) float tensor, binary (0 = background, 1 = plant)

Usage:
    python dataset.py
    (runs a quick sanity check to verify images and masks are loading correctly)
"""

import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt
from config import DATASET_PATH

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

# Path to the A1 folder inside your project directory
DATASET_PATH = str(DATASET_PATH)

# Image size to resize all inputs to (U-Net works best with multiples of 32)
IMAGE_SIZE = 256

# Train/validation split (80% train, 20% validation)
TRAIN_SPLIT = 0.8

# Random seed for reproducibility
SEED = 42


# ─────────────────────────────────────────────
# AUGMENTATION PIPELINES
# ─────────────────────────────────────────────

# Training augmentations — random flips and rotations to increase variety
# These are applied to both the image and mask simultaneously
train_transform = A.Compose([
    A.Resize(IMAGE_SIZE, IMAGE_SIZE),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.Normalize(mean=(0.485, 0.456, 0.406),   # ImageNet mean/std
                std=(0.229, 0.224, 0.225)),    # (used by pretrained encoder)
    ToTensorV2(),
])

# Validation augmentations — resize and normalise only, no random transforms
val_transform = A.Compose([
    A.Resize(IMAGE_SIZE, IMAGE_SIZE),
    A.Normalize(mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])


# ─────────────────────────────────────────────
# DATASET CLASS
# ─────────────────────────────────────────────

class CVPPPDataset(Dataset):
    """
    PyTorch Dataset for the CVPPP 2014 A1 Arabidopsis dataset.

    Expects the following folder structure:
        A1/
          plant001/
            plant001_rgb.png
            plant001_label.png
          plant002/
            ...

    The label mask is converted to binary: any pixel > 0 becomes 1 (plant),
    0 stays 0 (background). The original label file uses different integer
    values to distinguish individual leaves, but for binary segmentation
    (plant vs background) we only need foreground/background.
    """

    def __init__(self, dataset_path, transform=None):
        self.transform = transform
        self.samples = []

        # All images sit directly in the dataset folder (flat structure)
        # Find all _rgb.png files and pair each with its matching _label.png
        for filename in sorted(os.listdir(dataset_path)):
            if not filename.endswith("_rgb.png"):
                continue

            plant_id   = filename.replace("_rgb.png", "")
            rgb_path   = os.path.join(dataset_path, filename)
            label_path = os.path.join(dataset_path, f"{plant_id}_label.png")

            if os.path.isfile(label_path):
                self.samples.append((rgb_path, label_path))

        print(f"Found {len(self.samples)} plant samples in {dataset_path}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        rgb_path, label_path = self.samples[idx]

        # Load RGB image (OpenCV loads as BGR, convert to RGB)
        image = cv2.imread(rgb_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Load label mask (greyscale) and convert to binary
        mask = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
        mask = (mask > 0).astype(np.uint8)  # 1 = plant, 0 = background

        # Apply augmentations
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]          # (3, H, W) float tensor
            mask  = augmented["mask"]           # (H, W) uint8 tensor

        # Add channel dimension to mask: (H, W) → (1, H, W)
        mask = mask.unsqueeze(0).float()

        return image, mask


# ─────────────────────────────────────────────
# HELPER: Build train/val dataloaders
# ─────────────────────────────────────────────

def get_dataloaders(dataset_path=DATASET_PATH, batch_size=4):
    """
    Loads the full dataset, splits into train/val, and returns DataLoaders.
    """
    # Load full dataset without transforms first (to get length for splitting)
    full_dataset = CVPPPDataset(dataset_path, transform=None)
    total = len(full_dataset)

    n_train = int(total * TRAIN_SPLIT)
    n_val   = total - n_train

    # Split indices
    generator = torch.Generator().manual_seed(SEED)
    train_indices, val_indices = random_split(
        range(total), [n_train, n_val], generator=generator
    )

    # Create separate dataset instances with correct transforms
    train_dataset = CVPPPDataset(dataset_path, transform=train_transform)
    val_dataset   = CVPPPDataset(dataset_path, transform=val_transform)

    # Subset each to the split indices
    train_subset = torch.utils.data.Subset(train_dataset, train_indices.indices)
    val_subset   = torch.utils.data.Subset(val_dataset,   val_indices.indices)

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_subset,   batch_size=batch_size, shuffle=False)

    print(f"Train: {len(train_subset)} samples | Val: {len(val_subset)} samples")
    return train_loader, val_loader


# ─────────────────────────────────────────────
# SANITY CHECK
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("=== Dataset Sanity Check ===")

    # Load one batch and print shapes
    dataset = CVPPPDataset(DATASET_PATH, transform=val_transform)

    if len(dataset) == 0:
        print("ERROR: No samples found. Check that DATASET_PATH is correct and")
        print(f"       that plant folders contain _rgb.png and _label.png files.")
        print(f"       Current path: {DATASET_PATH}")
    else:
        image, mask = dataset[0]
        print(f"Sample 0:")
        print(f"  Image shape : {image.shape}  (expected: torch.Size([3, 256, 256]))")
        print(f"  Mask shape  : {mask.shape}   (expected: torch.Size([1, 256, 256]))")
        print(f"  Image range : [{image.min():.2f}, {image.max():.2f}]")
        print(f"  Mask values : {mask.unique().tolist()}  (expected: [0.0, 1.0])")

        # Visualise first 4 samples — image alongside its binary mask
        fig, axes = plt.subplots(2, 4, figsize=(14, 7))
        fig.suptitle("CVPPP A1 Dataset — First 4 Samples (Image | Mask)", fontsize=13)

        mean = np.array([0.485, 0.456, 0.406])
        std  = np.array([0.229, 0.224, 0.225])

        for i in range(min(4, len(dataset))):
            img, msk = dataset[i]

            # Denormalise image for display
            img_display = img.permute(1, 2, 0).numpy()
            img_display = (img_display * std + mean).clip(0, 1)

            axes[0, i].imshow(img_display)
            axes[0, i].set_title(f"Plant {i+1} — RGB")
            axes[0, i].axis("off")

            axes[1, i].imshow(msk.squeeze().numpy(), cmap="gray")
            axes[1, i].set_title(f"Plant {i+1} — Mask")
            axes[1, i].axis("off")

        plt.tight_layout()
        plt.savefig("dataset_check.png", dpi=100)
        plt.show()
        print("\nSaved visualisation to dataset_check.png")
        print("\n=== Sanity check complete — dataset is loading correctly ===")

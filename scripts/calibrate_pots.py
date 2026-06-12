"""
calibrate_pots.py -- One-time pot calibration for Growth Chamber CV pipeline
EE496 | Luke Buckley | Maynooth University

Run this ONCE before the trial starts with a top-down image of the chamber.
You click to define each pot centre and adjust the radius with the scroll wheel.
The calibration is saved to calibration/{chamber}_calibration.json and used
by analyse_chamber.py daily.

Controls:
    Left click       -- place a circle for the next pot
    Scroll wheel     -- adjust radius of the most recently placed circle
    R                -- remove the last placed circle
    Enter / Space    -- confirm current calibration and save (all 8 pots placed)
    Q / Escape       -- quit without saving

Usage:
    python calibrate_pots.py --image "path/to/image.jpg" --chamber enriched
    python calibrate_pots.py --image "path/to/image.jpg" --chamber control
"""

import cv2
import numpy as np
import json
import os
import argparse
from config import CALIB_DIR

# ── Paths ─────────────────────────────────────────────────────────────────────
CALIB_DIR     = str(CALIB_DIR)
N_POTS        = 8

# ── Label normalisation ───────────────────────────────────────────────────────
def _normalise_label(label):
    """Normalise pot label casing: capitalise each word segment after underscores.
    e.g. 'co2_pot1' → 'Co2_Pot1', 'Control_pot3' → 'Control_Pot3'"""
    return "_".join(part.capitalize() for part in label.split("_"))

# ── Colours ───────────────────────────────────────────────────────────────────
POT_COLOUR    = (0,   255, 0)     # green — pots
ACTIVE_COLOUR = (0,   255, 255)   # yellow — currently being adjusted
TEXT_COLOUR   = (255, 255, 255)
FONT          = cv2.FONT_HERSHEY_SIMPLEX


class CalibrationTool:
    def __init__(self, image, chamber):
        self.orig        = image.copy()
        self.chamber     = chamber
        self.circles     = []   # list of [x, y, r, label]
        self.pot_count   = 0
        self.state       = "pots"   # "pots" or "done"
        self.display     = image.copy()
        self.window_name = f"Calibration — {chamber} chamber"

    def draw(self):
        img = self.orig.copy()

        for i, (x, y, r, label) in enumerate(self.circles):
            is_last = (i == len(self.circles) - 1)
            colour = ACTIVE_COLOUR if is_last else POT_COLOUR
            cv2.circle(img, (x, y), r, colour, 2)
            cv2.putText(img, label, (x - 15, y + 5), FONT, 0.6, TEXT_COLOUR, 2)

        # Instructions overlay
        if self.state == "pots":
            if self.pot_count == 0:
                radius_hint = "Scroll after first click to set radius (all pots share it)"
            else:
                radius_hint = f"Radius locked to {self.circles[0][2]}px (scroll pot 1 to change)"
            lines = [
                f"Click centre of pot {self.pot_count + 1} of {N_POTS}",
                radius_hint,
                "R = undo last  |  Enter = save when all placed",
            ]
        else:
            lines = [
                f"All {N_POTS} pots placed.",
                "Press Enter to save  |  R to undo last",
            ]

        for i, line in enumerate(lines):
            cv2.putText(img, line, (10, 30 + i * 28), FONT, 0.65, (0, 0, 0),   3)
            cv2.putText(img, line, (10, 30 + i * 28), FONT, 0.65, TEXT_COLOUR, 1)

        # Progress indicator
        prog = f"Pots: {self.pot_count}/{N_POTS}"
        cv2.putText(img, prog, (10, img.shape[0] - 15), FONT, 0.6, (0, 0, 0),   2)
        cv2.putText(img, prog, (10, img.shape[0] - 15), FONT, 0.6, TEXT_COLOUR, 1)

        self.display = img
        cv2.imshow(self.window_name, img)

    def click(self, x, y):
        if self.pot_count >= N_POTS:
            print("All pots placed. Press Enter to save or R to undo last.")
            return
        # Pot 1: use default radius. Pots 2–8: inherit radius from pot 1.
        if self.circles:
            default_r = self.circles[0][2]
        else:
            default_r = min(self.orig.shape[0], self.orig.shape[1]) // 15
        pot_label = f"P{self.pot_count + 1}"
        self.circles.append([x, y, default_r, pot_label])
        self.pot_count += 1
        if self.pot_count == 1:
            msg = "Scroll to adjust radius — all other pots will use the same size. Click next pot."
        elif self.pot_count < N_POTS:
            msg = f"Click next pot."
        else:
            msg = "All pots placed. Press Enter to save."
        print(f"Pot {pot_label} placed at ({x}, {y}). {msg}")
        if self.pot_count >= N_POTS:
            self.state = "done"

    def scroll(self, direction):
        if not self.circles:
            return
        self.circles[-1][2] = max(10, self.circles[-1][2] + direction * 5)

    def undo(self):
        if not self.circles:
            return
        removed = self.circles.pop()
        self.pot_count = max(0, self.pot_count - 1)
        self.state = "pots"
        print(f"Removed: {removed[3]}")

    def get_pot_names(self):
        """After calibration, prompt user to rename each pot (Enter = keep default)."""
        print("\nName each pot (press Enter to keep default label e.g. P1):")
        for c in self.circles:
            name = input(f"  Name for {c[3]} at ({c[0]}, {c[1]}): ").strip()
            if name:
                c[3] = _normalise_label(name)
        return self.circles

    def save(self, output_path, orig_size=None):
        """Save calibration to JSON. orig_size=(w,h) should be the original image dimensions."""
        if len(self.circles) != N_POTS:
            print(f"ERROR: Expected {N_POTS} pots, got {len(self.circles)}.")
            return False

        img_w, img_h = orig_size if orig_size else (self.orig.shape[1], self.orig.shape[0])
        calib = {
            "chamber":    self.chamber,
            "image_size": [img_w, img_h],
            "locked":     True,
            "pots":       [
                {"label": p[3], "x": p[0], "y": p[1], "r": p[2]}
                for p in self.circles
            ]
        }

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(calib, f, indent=2)

        print(f"\nCalibration saved to {output_path}")
        for p in self.circles:
            print(f"  {p[3]}: centre=({p[0]}, {p[1]}), radius={p[2]}px")
        return True


def mouse_callback(event, x, y, flags, tool):
    if event == cv2.EVENT_LBUTTONDOWN:
        tool.click(x, y)
        tool.draw()
    elif event == cv2.EVENT_MOUSEWHEEL:
        direction = 1 if flags > 0 else -1
        tool.scroll(direction)
        tool.draw()


def run_calibration(image_path, chamber):
    image = cv2.imread(image_path)
    if image is None:
        print(f"ERROR: Could not load image from {image_path}")
        return

    # Resize for display if very large
    h, w = image.shape[:2]
    max_dim = 1200
    scale = 1.0
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        image = cv2.resize(image, (int(w * scale), int(h * scale)))
        print(f"Image resized to {image.shape[1]}x{image.shape[0]} for display. "
              f"Coordinates will be scaled back automatically.")

    tool = CalibrationTool(image, chamber)
    cv2.namedWindow(tool.window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(tool.window_name, mouse_callback, tool)
    tool.draw()

    print(f"\nCalibration started for {chamber} chamber.")
    print("1. Click the centre of each pot (8 total).")
    print("2. Scroll to adjust the radius of the last placed circle.")
    print("3. Press Enter to save when all 8 pots are placed.\n")

    while True:
        key = cv2.waitKey(50) & 0xFF
        if key in (13, 32):  # Enter or Space
            if tool.state == "done":
                cv2.destroyAllWindows()
                pot_circles = tool.get_pot_names()
                # Scale coordinates back to original image size if resized
                if scale != 1.0:
                    for c in tool.circles:
                        c[0] = int(c[0] / scale)
                        c[1] = int(c[1] / scale)
                        c[2] = int(c[2] / scale)
                output_path = os.path.join(CALIB_DIR, f"{chamber}_calibration.json")
                orig = cv2.imread(image_path)
                orig_h_full, orig_w_full = orig.shape[:2]
                tool.save(output_path, orig_size=(orig_w_full, orig_h_full))

                # Save a preview image showing all circles on the original
                for p in tool.circles:
                    cv2.circle(orig, (p[0], p[1]), p[2], POT_COLOUR, 3)
                    cv2.putText(orig, p[3], (p[0]-20, p[1]+8), FONT, 0.8, POT_COLOUR, 2)
                preview_path = os.path.join(CALIB_DIR, f"{chamber}_calibration_preview.jpg")
                cv2.imwrite(preview_path, orig)
                print(f"Preview saved to {preview_path}")
                break
            else:
                print(f"Not ready — {tool.pot_count}/{N_POTS} pots placed. Place all pots first.")
        elif key in (ord('r'), ord('R')):
            tool.undo()
            tool.draw()
        elif key in (ord('q'), ord('Q'), 27):  # Q or Escape
            print("Cancelled — no calibration saved.")
            cv2.destroyAllWindows()
            break

    cv2.destroyAllWindows()


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="One-time pot and hive calibration for Growth Chamber CV pipeline."
    )
    parser.add_argument("--image",   required=True,
                        help="Path to a top-down chamber image for calibration")
    parser.add_argument("--chamber", required=True, choices=["enriched", "control"],
                        help="Which chamber to calibrate")
    args = parser.parse_args()

    run_calibration(args.image, args.chamber)

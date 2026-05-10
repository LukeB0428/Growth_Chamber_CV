"""
live_view.py -- Live camera preview for OAK-D Lite physical setup
EE496 | Luke Buckley | Maynooth University

Opens a real-time window showing the RGB feed (and optionally depth).
Use this to aim and focus the camera before running calibrate_pots.py.

If a calibration JSON already exists for the selected chamber, the
saved pot circles are drawn on the live feed so you can verify alignment.

Controls:
    Q / Esc  -- quit
    S        -- save a snapshot to images/{chamber}/snapshot_HHMMSS.jpg
    D        -- toggle depth overlay on/off
    G        -- toggle greyscale green-mask overlay (shows what HSV will detect)

Usage:
    python live_view.py --chamber enriched
    python live_view.py --chamber control
    python live_view.py --chamber enriched --no-depth   # RGB only (faster)
"""

import depthai as dai
import cv2
import numpy as np
import os
import json
import argparse
import time
from datetime import datetime
from config import IMAGES_DIR, CALIB_DIR

# ── Paths ─────────────────────────────────────────────────────────────────────
IMAGES_DIR = str(IMAGES_DIR)
CALIB_DIR  = str(CALIB_DIR)

# ── Camera settings ───────────────────────────────────────────────────────────
RGB_SIZE   = (1920, 1080)
DEPTH_SIZE = (640, 400)

# Preview window is scaled down so it fits a typical monitor
PREVIEW_W  = 1280
PREVIEW_H  = int(PREVIEW_W * 1080 / 1920)   # 720

# HSV green-mask thresholds (mirrors analyse_image.py)
HSV_LOWER = np.array([25,  40,  40])
HSV_UPPER = np.array([90, 255, 255])


def load_calibration(chamber_id):
    """Load pot circles from calibration JSON, or return None."""
    path = os.path.join(CALIB_DIR, f"{chamber_id}_calibration.json")
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        return json.load(f)


def draw_calibration(frame, calib, scale_x, scale_y):
    """Draw hive boundary and pot circles onto the preview frame."""
    if calib is None:
        return

    # Hive boundary
    hive = calib.get("hive", {})
    if hive:
        cx = int(hive["x"] * scale_x)
        cy = int(hive["y"] * scale_y)
        r  = int(hive["r"] * scale_x)
        cv2.circle(frame, (cx, cy), r, (0, 200, 255), 2)
        cv2.putText(frame, "hive", (cx - 20, cy - r - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)

    # Individual pots
    for pot in calib.get("pots", []):
        cx = int(pot["x"] * scale_x)
        cy = int(pot["y"] * scale_y)
        r  = int(pot["r"] * scale_x)
        cv2.circle(frame, (cx, cy), r, (0, 255, 80), 2)
        cv2.putText(frame, pot["label"], (cx - 10, cy + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 80), 2)


def make_depth_overlay(depth_frame, alpha=0.4):
    """Convert 16-bit depth to a colourmap and return a 1920×1080 BGR overlay."""
    valid = depth_frame > 0
    preview = np.zeros((*depth_frame.shape, 3), dtype=np.uint8)
    if valid.any():
        d = depth_frame.astype(np.float32)
        d_min = d[valid].min()
        d_max = d[valid].max()
        norm = np.zeros_like(d, dtype=np.uint8)
        norm[valid] = np.clip(
            255.0 * (1.0 - (d[valid] - d_min) / (d_max - d_min + 1e-6)),
            0, 255
        ).astype(np.uint8)
        coloured = cv2.applyColorMap(norm, cv2.COLORMAP_TURBO)
        coloured[~valid] = 0
        preview = coloured
    # Upscale depth (640×400) to RGB size (1920×1080)
    preview_full = cv2.resize(preview, RGB_SIZE, interpolation=cv2.INTER_NEAREST)
    return preview_full


def run_live_view(chamber_id, use_depth):
    os.makedirs(os.path.join(IMAGES_DIR, chamber_id), exist_ok=True)

    calib = load_calibration(chamber_id)
    if calib:
        print(f"Calibration loaded — {len(calib.get('pots', []))} pots will be drawn.")
    else:
        print("No calibration JSON found — pot circles will not be shown.")

    scale_x = PREVIEW_W / RGB_SIZE[0]
    scale_y = PREVIEW_H / RGB_SIZE[1]

    print("Connecting to OAK-D Lite...")
    pipeline = dai.Pipeline()

    # RGB
    cam = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
    rgb_out   = cam.requestOutput(RGB_SIZE, type=dai.ImgFrame.Type.BGR888p)
    rgb_queue = rgb_out.createOutputQueue()

    # Stereo depth (optional)
    depth_queue = None
    if use_depth:
        left  = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_B)
        right = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_C)
        stereo = pipeline.create(dai.node.StereoDepth)
        stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.FAST_DENSITY)
        stereo.setLeftRightCheck(True)
        stereo.setSubpixel(True)
        left.requestOutput(DEPTH_SIZE,  type=dai.ImgFrame.Type.GRAY8).link(stereo.left)
        right.requestOutput(DEPTH_SIZE, type=dai.ImgFrame.Type.GRAY8).link(stereo.right)
        depth_queue = stereo.depth.createOutputQueue()

    pipeline.start()
    print("Warming up (3s)...")
    time.sleep(3)

    show_depth  = False
    show_green  = False
    depth_frame = None

    print("\nLive view running.")
    print("  Q / Esc -- quit")
    print("  S       -- save snapshot")
    if use_depth:
        print("  D       -- toggle depth overlay")
    print("  G       -- toggle green-mask overlay")
    print()

    cv2.namedWindow("Growth Chamber — Live View", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Growth Chamber — Live View", PREVIEW_W, PREVIEW_H)

    while True:
        # Grab latest RGB frame (non-blocking loop)
        if not rgb_queue.has():
            time.sleep(0.02)
            continue

        rgb_frame = rgb_queue.get().getCvFrame()   # 1920×1080 BGR

        # Grab latest depth if available
        if depth_queue is not None and depth_queue.has():
            depth_frame = depth_queue.get().getFrame()

        display = rgb_frame.copy()

        # Depth overlay
        if show_depth and depth_frame is not None:
            overlay = make_depth_overlay(depth_frame)
            cv2.addWeighted(overlay, 0.45, display, 0.55, 0, display)

        # Green-mask overlay
        if show_green:
            hsv  = cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2HSV)
            gmask = cv2.inRange(hsv, HSV_LOWER, HSV_UPPER)
            green_tint = np.zeros_like(display)
            green_tint[:, :, 1] = gmask   # green channel only
            cv2.addWeighted(green_tint, 0.4, display, 0.6, 0, display)

        # Status bar
        mode_str = []
        if show_depth and use_depth:
            mode_str.append("DEPTH")
        if show_green:
            mode_str.append("GREEN-MASK")
        status = f"Chamber: {chamber_id}  |  {datetime.now().strftime('%H:%M:%S')}"
        if mode_str:
            status += f"  |  [{', '.join(mode_str)}]"
        cv2.putText(display, status, (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)

        # Resize for display
        preview = cv2.resize(display, (PREVIEW_W, PREVIEW_H))
        cv2.imshow("Growth Chamber — Live View", preview)

        key = cv2.waitKey(1) & 0xFF

        if key in (ord('q'), 27):   # Q or Esc
            break

        elif key == ord('s'):
            ts   = datetime.now().strftime("%H%M%S")
            path = os.path.join(IMAGES_DIR, chamber_id, f"snapshot_{ts}.jpg")
            cv2.imwrite(path, rgb_frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            print(f"Snapshot saved: {path}")

        elif key == ord('d') and use_depth:
            show_depth = not show_depth
            print(f"Depth overlay: {'ON' if show_depth else 'OFF'}")

        elif key == ord('g'):
            show_green = not show_green
            print(f"Green-mask overlay: {'ON' if show_green else 'OFF'}")

    pipeline.stop()
    cv2.destroyAllWindows()
    print("Live view closed.")


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Live preview from OAK-D Lite for physical camera setup."
    )
    parser.add_argument("--chamber", required=True, choices=["enriched", "control"],
                        help="Which chamber to preview")
    parser.add_argument("--no-depth", action="store_true",
                        help="Disable stereo depth (faster, RGB only)")
    args = parser.parse_args()

    run_live_view(args.chamber, use_depth=not args.no_depth)

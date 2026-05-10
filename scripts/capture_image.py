"""
capture_image.py -- OAK-D Lite image capture for Growth Chamber CV pipeline
EE496 | Luke Buckley | Maynooth University

Captures a single RGB frame and depth frame from the OAK-D Lite camera
and saves them to the correct images folder for the specified chamber.

Written for depthai v3 API (pipeline.start(), Camera.build(), createOutputQueue())

Saved files:
    images/{chamber}/YYYY-MM-DD_{chamber}.jpg              -- RGB image for analyse_image.py
    images/{chamber}/YYYY-MM-DD_{chamber}_depth.png        -- 16-bit depth map (mm) for Stage 4
    images/{chamber}/YYYY-MM-DD_{chamber}_depth_preview.jpg -- normalised greyscale depth for viewing

Usage:
    python capture_image.py --chamber enriched
    python capture_image.py --chamber control
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

# ── Camera settings ───────────────────────────────────────────────────────────
RGB_SIZE        = (1920, 1080)   # capture resolution
DEPTH_SIZE      = (640, 400)     # stereo depth resolution
WARMUP_SECS     = 3              # frames to drain before capturing
CAMERA_MAP_PATH = str(CALIB_DIR / "camera_map.json")


def _get_device_info(chamber_id):
    """
    Return a dai.DeviceInfo for the camera assigned to this chamber, or None
    to fall back to the first available device (single-camera setup).
    """
    if not os.path.isfile(CAMERA_MAP_PATH):
        return None  # no map — use first available camera

    with open(CAMERA_MAP_PATH) as f:
        camera_map = json.load(f)

    mx_id = camera_map.get(chamber_id)
    if not mx_id:
        print(f"  [camera_map] No entry for '{chamber_id}' — using first available camera.")
        return None

    # Verify the device is currently connected
    available = {d.deviceId: d for d in dai.Device.getAllAvailableDevices()}
    if mx_id not in available:
        raise RuntimeError(
            f"Camera for '{chamber_id}' (MX ID: {mx_id}) not found. "
            f"Check USB connection. Available: {list(available.keys())}"
        )

    print(f"  Using camera MX ID: {mx_id} for {chamber_id}")
    return available[mx_id]


def capture(chamber_id):
    """
    Connect to OAK-D Lite, warm up, capture one RGB frame and one depth frame.
    Saves to images/{chamber_id}/ and returns (rgb_path, depth_path).
    """
    out_dir = os.path.join(IMAGES_DIR, chamber_id)
    os.makedirs(out_dir, exist_ok=True)

    date_str     = datetime.now().strftime("%Y-%m-%d")
    rgb_path     = os.path.join(out_dir, f"{date_str}_{chamber_id}.jpg")
    depth_path   = os.path.join(out_dir, f"{date_str}_{chamber_id}_depth.png")
    preview_path = os.path.join(out_dir, f"{date_str}_{chamber_id}_depth_preview.jpg")

    print("Connecting to OAK-D Lite...")
    device_info = _get_device_info(chamber_id)
    if device_info is not None:
        device = dai.Device(device_info)
        pipeline = dai.Pipeline(device)
    else:
        pipeline = dai.Pipeline()

    # ── RGB camera ────────────────────────────────────────────────────────────
    cam = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
    rgb_out   = cam.requestOutput(RGB_SIZE, type=dai.ImgFrame.Type.BGR888p)
    rgb_queue = rgb_out.createOutputQueue()

    # ── Stereo depth ──────────────────────────────────────────────────────────
    left  = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_B)
    right = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_C)
    stereo = pipeline.create(dai.node.StereoDepth)
    stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.FAST_DENSITY)
    stereo.setLeftRightCheck(True)
    stereo.setSubpixel(True)

    left.requestOutput(DEPTH_SIZE,  type=dai.ImgFrame.Type.GRAY8).link(stereo.left)
    right.requestOutput(DEPTH_SIZE, type=dai.ImgFrame.Type.GRAY8).link(stereo.right)

    depth_queue = stereo.depth.createOutputQueue()

    # ── Start pipeline ────────────────────────────────────────────────────────
    pipeline.start()
    print(f"Warming up for {WARMUP_SECS}s (auto-exposure settling)...")
    time.sleep(WARMUP_SECS)

    # Drain stale frames that built up during warmup
    while rgb_queue.has():
        rgb_queue.get()
    while depth_queue.has():
        depth_queue.get()

    # ── Grab one clean frame pair ─────────────────────────────────────────────
    print("Capturing...")
    rgb_frame   = rgb_queue.get().getCvFrame()
    depth_frame = depth_queue.get().getFrame()

    pipeline.stop()

    # ── Save RGB ──────────────────────────────────────────────────────────────
    cv2.imwrite(rgb_path, rgb_frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
    print(f"RGB saved:        {rgb_path}")
    print(f"Centre pixel (B,G,R): {rgb_frame[rgb_frame.shape[0]//2, rgb_frame.shape[1]//2]}")

    # ── Save depth as 16-bit PNG (values in mm) ───────────────────────────────
    cv2.imwrite(depth_path, depth_frame.astype(np.uint16))
    print(f"Depth saved:      {depth_path}")

    # Normalised greyscale preview (closer = brighter)
    valid_mask = depth_frame > 0
    if valid_mask.any():
        preview = np.zeros_like(depth_frame, dtype=np.uint8)
        d_valid = depth_frame[valid_mask].astype(np.float32)
        normalised = np.clip(
            255.0 * (1.0 - (d_valid - d_valid.min()) /
                     (d_valid.max() - d_valid.min() + 1e-6)),
            0, 255
        ).astype(np.uint8)
        preview[valid_mask] = normalised
        cv2.imwrite(preview_path, preview)
        print(f"Depth preview:    {preview_path}")

    # Sanity check on depth values
    h, w   = depth_frame.shape
    centre = depth_frame[h//4:3*h//4, w//4:3*w//4]
    valid  = centre[centre > 0]
    if len(valid) > 0:
        print(f"Centre depth:     {np.mean(valid):.0f} mm "
              f"(min {np.min(valid)} mm, max {np.max(valid)} mm)")

    return rgb_path, depth_path


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Capture RGB + depth from OAK-D Lite for a growth chamber."
    )
    parser.add_argument("--chamber", required=True, choices=["enriched", "control"],
                        help="Which chamber to capture")
    args = parser.parse_args()

    rgb_path, depth_path = capture(args.chamber)

    if rgb_path:
        print(f"\nCapture complete. Run analysis with:")
        print(f'  python analyse_image.py --image "{rgb_path}" --chamber {args.chamber}')
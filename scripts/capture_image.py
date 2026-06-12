"""
capture_image.py -- OAK-D Lite image capture for Growth Chamber CV pipeline
EE496 | Luke Buckley | Maynooth University

Captures a single RGB frame and depth frame from the OAK-D Lite camera
and saves them to the correct images folder for the specified chamber.

Written for depthai v3 API (pipeline.start(), Camera.build(), createOutputQueue())

Exposure behaviour:
    First run (or --recalibrate-exposure): auto-exposure settles for WARMUP_SECS,
    then settings are locked and saved to calibration/{chamber}_exposure.json.
    Subsequent runs load saved settings and apply them directly — ensuring
    consistent, reproducible images across days regardless of transient lighting.

Saved files:
    images/{chamber}/YYYY-MM-DD_{chamber}.jpg              -- RGB image for analyse_image.py
    images/{chamber}/YYYY-MM-DD_{chamber}_depth.png        -- 16-bit depth map (mm) for Stage 4
    images/{chamber}/YYYY-MM-DD_{chamber}_depth_preview.jpg -- normalised greyscale depth for viewing

Usage:
    python capture_image.py --chamber enriched
    python capture_image.py --chamber enriched --recalibrate-exposure
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
WARMUP_SECS     = 3              # warmup duration when running auto-exposure calibration
WARMUP_SECS_MANUAL = 1           # shorter warmup when applying saved exposure settings
CAMERA_MAP_PATH = str(CALIB_DIR / "camera_map.json")


def _load_exposure_config(chamber_id):
    path = CALIB_DIR / f"{chamber_id}_exposure.json"
    if path.is_file():
        with open(path) as f:
            return json.load(f)
    return None


def _save_exposure_config(chamber_id, exposure_us, iso, wb_k):
    # DepthAI getExposureTime() returns a timedelta on some firmware versions
    if hasattr(exposure_us, 'total_seconds'):
        exposure_us = int(exposure_us.total_seconds() * 1_000_000)
    path = CALIB_DIR / f"{chamber_id}_exposure.json"
    config = {
        "exposure_us": int(exposure_us),
        "iso":         int(iso),
        "wb_k":        int(wb_k),
        "calibrated_at": datetime.now().isoformat()
    }
    with open(path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"  Exposure saved: {exposure_us}us  ISO {iso}  WB {wb_k}K → {path.name}")
    return config


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


def capture(chamber_id, recalibrate_exposure=False):
    """
    Connect to OAK-D Lite, warm up, capture one RGB frame and one depth frame.
    Saves to images/{chamber_id}/ and returns (rgb_path, depth_path).

    On first run (or recalibrate_exposure=True): lets auto-exposure settle then
    locks and saves settings to calibration/{chamber}_exposure.json.
    On subsequent runs: applies saved settings directly for reproducible images.
    """
    out_dir = os.path.join(IMAGES_DIR, chamber_id)
    os.makedirs(out_dir, exist_ok=True)

    date_str     = datetime.now().strftime("%Y-%m-%d")
    rgb_path     = os.path.join(out_dir, f"{date_str}_{chamber_id}.jpg")
    depth_path   = os.path.join(out_dir, f"{date_str}_{chamber_id}_depth.png")
    preview_path = os.path.join(out_dir, f"{date_str}_{chamber_id}_depth_preview.jpg")

    # ── Load saved exposure settings ──────────────────────────────────────────
    exposure_config = None if recalibrate_exposure else _load_exposure_config(chamber_id)
    if exposure_config:
        print(f"  Applying saved exposure: {exposure_config['exposure_us']}us "
              f"ISO {exposure_config['iso']}  WB {exposure_config['wb_k']}K")
    else:
        print("  No saved exposure — running auto-exposure calibration...")

    print("Connecting to OAK-D Lite...")
    device_info = _get_device_info(chamber_id)
    if device_info is not None:
        device = dai.Device(device_info)
        pipeline = dai.Pipeline(device)
    else:
        pipeline = dai.Pipeline()

    # ── RGB camera ────────────────────────────────────────────────────────────
    cam = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)

    if exposure_config:
        cam.initialControl.setManualExposure(
            exposure_config["exposure_us"], exposure_config["iso"]
        )
        cam.initialControl.setManualWhiteBalance(exposure_config["wb_k"])

    rgb_out    = cam.requestOutput(RGB_SIZE, type=dai.ImgFrame.Type.BGR888p)
    rgb_queue  = rgb_out.createOutputQueue()
    ctrl_queue = cam.inputControl.createInputQueue()

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
    warmup = WARMUP_SECS_MANUAL if exposure_config else WARMUP_SECS
    print(f"Warming up for {warmup}s...")
    time.sleep(warmup)

    # If auto-exposure was used, lock current settings before capturing
    if not exposure_config:
        ctrl = dai.CameraControl()
        ctrl.setAutoExposureLock(True)
        ctrl.setAutoWhiteBalanceLock(True)
        ctrl_queue.send(ctrl)
        time.sleep(0.5)

    # Drain stale frames that built up during warmup
    while rgb_queue.has():
        rgb_queue.get()
    while depth_queue.has():
        depth_queue.get()

    # ── Grab one clean frame pair ─────────────────────────────────────────────
    print("Capturing...")
    rgb_frame_obj = rgb_queue.get()
    rgb_frame     = rgb_frame_obj.getCvFrame()
    depth_frame   = depth_queue.get().getFrame()

    # Save exposure settings after first auto-calibration run
    if not exposure_config:
        try:
            exp_us = rgb_frame_obj.getExposureTime()
            iso    = rgb_frame_obj.getSensitivity()
            wb_k   = rgb_frame_obj.getColorTemperature()
            _save_exposure_config(chamber_id, exp_us, iso, wb_k)
        except Exception as e:
            print(f"  Note: could not read exposure metadata ({e}) — settings not saved")

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
    parser.add_argument("--recalibrate-exposure", action="store_true",
                        help="Ignore saved exposure settings and re-run auto-exposure calibration")
    args = parser.parse_args()

    rgb_path, depth_path = capture(args.chamber, recalibrate_exposure=args.recalibrate_exposure)

    if rgb_path:
        print(f"\nCapture complete. Run analysis with:")
        print(f'  python analyse_image.py --image "{rgb_path}" --chamber {args.chamber}')
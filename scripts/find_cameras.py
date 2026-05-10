"""
find_cameras.py — List all connected OAK-D Lite cameras and their serial numbers.
EE496 | Luke Buckley | Maynooth University

Run this with both cameras plugged in to find their MX IDs, then update
calibration/camera_map.json to assign each camera to a chamber.

Usage:
    python find_cameras.py
"""

import depthai as dai
import json
import os
from config import CALIB_DIR

CAMERA_MAP_PATH = str(CALIB_DIR / "camera_map.json")


def list_cameras():
    devices = dai.Device.getAllAvailableDevices()

    if not devices:
        print("No OAK-D cameras found. Check USB connections.")
        return

    print(f"\nFound {len(devices)} camera(s):\n")
    for i, d in enumerate(devices):
        print(f"  [{i}] MX ID: {d.deviceId}  |  State: {d.state.name}")

    print()

    # Load existing map if present
    existing = {}
    if os.path.isfile(CAMERA_MAP_PATH):
        with open(CAMERA_MAP_PATH) as f:
            existing = json.load(f)
        print("Current camera_map.json:")
        for chamber, mx_id in existing.items():
            print(f"  {chamber}: {mx_id}")
        print()

    # If exactly 2 cameras found and no map exists, prompt to assign
    if len(devices) == 2 and not existing:
        print("Two cameras detected. Let's assign them to chambers.")
        print("Unplug one camera at a time to identify which is which, OR")
        print("enter the MX IDs manually below.\n")

        enriched_id = input(f"Enter MX ID for ENRICHED chamber camera: ").strip()
        control_id  = input(f"Enter MX ID for CONTROL chamber camera: ").strip()

        valid_ids = {d.deviceId for d in devices}
        if enriched_id not in valid_ids or control_id not in valid_ids:
            print("\nError: one or both IDs not recognised. Check the list above.")
            return

        camera_map = {"enriched": enriched_id, "control": control_id}

        with open(CAMERA_MAP_PATH, "w") as f:
            json.dump(camera_map, f, indent=2)

        print(f"\nSaved to {CAMERA_MAP_PATH}")
        print("Camera assignment complete. capture_image.py will now use the correct camera per chamber.")

    elif len(devices) == 1:
        print("Only one camera connected.")
        print("Plug in both cameras together to assign them to chambers.")

    else:
        print("To reassign cameras, delete calibration/camera_map.json and re-run.")


if __name__ == "__main__":
    list_cameras()

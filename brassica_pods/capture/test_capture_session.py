"""
test_capture_session.py — capture-loop tests with a mocked frame source.

    python -m brassica_pods.capture.test_capture_session

Verifies the loop logic (validate_frame -> persist_or_quarantine -> operator
status) end-to-end without a camera: MockFrameSource supplies frames, the cheap
detectors (sharpness/exposure) run for real and drive pass/fail by frame
content, ArUco/QR/ColorChecker are stubbed present. The only unverified part is
the real depthai I/O (OakDFrameSource), by design.
"""

from __future__ import annotations

import contextlib
import tempfile
from pathlib import Path

import numpy as np

from brassica_pods.capture import capture_profile as cp
from brassica_pods.capture.capture_session import (
    CaptureSession, MockFrameSource, RawFrame)

VALID_ID = "EXP1-001-PLANT-000"
MANIFEST = {"EXP1-001": {"plant_id": "EXP1-001"}}


def _good_image():
    rng = np.random.RandomState(0)
    img = np.full((400, 500, 3), 120, np.uint8)
    return np.clip(img.astype(np.int16) + rng.randint(-25, 25, img.shape), 0, 255).astype(np.uint8)


def _bad_image():
    return np.full((400, 500, 3), 255, np.uint8)   # blown-out -> BAD_EXPOSURE


_ARUCO_OK = lambda i, *a, **k: {"marker_id": 7, "side_px": 200.0,
                                "px_per_mm": 6.67, "corners": np.zeros((4, 2))}


@contextlib.contextmanager
def _present_detectors(qr=VALID_ID):
    saved = {k: getattr(cp, k) for k in
             ("detect_aruco", "decode_qr", "detect_colorchecker", "MARKER_SIZE_CONFIRMED")}
    try:
        cp.detect_aruco = _ARUCO_OK
        cp.decode_qr = lambda img: qr
        cp.detect_colorchecker = lambda img: 24
        cp.MARKER_SIZE_CONFIRMED = True
        yield
    finally:
        for k, v in saved.items():
            setattr(cp, k, v)


def test_session_refuses_until_confirmed():
    assert cp.MARKER_SIZE_CONFIRMED is False
    sess = CaptureSession(MockFrameSource([RawFrame(_good_image())]), "spread",
                          tempfile.mkdtemp(), tempfile.mkdtemp())
    raised = False
    try:
        sess.run()
    except cp.SessionNotReady:
        raised = True
    assert raised, "session must refuse to start while marker sizes unconfirmed"


def test_loop_keeps_and_quarantines():
    kept, quar = tempfile.mkdtemp(), tempfile.mkdtemp()
    frames = [RawFrame(_good_image()), RawFrame(_bad_image()), RawFrame(_good_image())]
    statuses = []
    with _present_detectors():
        sess = CaptureSession(MockFrameSource(frames), "spread", kept, quar,
                              on_status=lambda s, r: statuses.append(s))
        stats = sess.run()
    assert stats.kept == 2 and stats.rejected == 1
    assert len(list(Path(kept).glob("*.jpg"))) == 2
    assert len(list(Path(quar).glob("*.jpg"))) == 1
    # every kept frame has a sidecar; rejected sidecar carries reasons
    assert len(list(Path(kept).glob("*.json"))) == 2
    assert len(statuses) == 3
    assert any("FAIL" in s for s in statuses)


def test_kept_frame_has_no_reasons_rejected_has():
    kept, quar = tempfile.mkdtemp(), tempfile.mkdtemp()
    with _present_detectors():
        stats = CaptureSession(
            MockFrameSource([RawFrame(_good_image())]), "spread", kept, quar).run()
    assert stats.kept == 1 and stats.rejected == 0


def test_warn_unknown_sample_passes_and_counts_warned():
    kept, quar = tempfile.mkdtemp(), tempfile.mkdtemp()
    statuses = []
    with _present_detectors(qr="EXP1-999-PLANT-000"):
        stats = CaptureSession(
            MockFrameSource([RawFrame(_good_image())]), "spread", kept, quar,
            manifest=MANIFEST, manifest_mode="warn",
            on_status=lambda s, r: statuses.append(s)).run()
    assert stats.kept == 1 and stats.warned == 1
    assert any("WARNING" in s for s in statuses)


def test_insitu_rejects_without_depth():
    kept, quar = tempfile.mkdtemp(), tempfile.mkdtemp()
    with _present_detectors():
        stats = CaptureSession(
            MockFrameSource([RawFrame(_good_image(), depth=None)]),
            "in_situ_multiview", kept, quar).run()
    assert stats.rejected == 1 and stats.kept == 0


def _main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t(); print(f"PASS  {t.__name__}")
        except AssertionError as e:
            failed += 1; print(f"FAIL  {t.__name__}: {e}")
        except Exception as e:
            failed += 1; print(f"ERROR {t.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return failed


if __name__ == "__main__":
    import sys
    sys.exit(1 if _main() else 0)

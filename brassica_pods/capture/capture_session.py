"""
capture_session.py — the capture loop that makes enforcement LIVE.

Glue only: frame source → (undistort) → validate_frame → persist_or_quarantine
→ operator status. The compliance logic lives in capture_profile.py; this wires
it to a stream of frames.

The depthai/OAK-D I/O is abstracted behind FrameSource so the LOOP logic is
unit-testable now with MockFrameSource (see test_capture_session.py). At the rig
you swap in OakDFrameSource and nothing else changes — "verified except the
camera I/O".

Kept frames (passed) go to kept_dir; rejected frames + their reasons go to
quarantine_dir. Nothing rejected enters the kept dataset.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterator, Optional

import numpy as np

from brassica_pods.capture import capture_profile as cp


# ─────────────────────────────────────────────────────────────────────────────
# Frame source abstraction
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class RawFrame:
    image: np.ndarray                       # BGR
    depth: Optional[np.ndarray] = None      # HxW mm (in_situ_multiview)
    view_angle_deg: float = 0.0             # turntable angle, if any
    meta: dict = field(default_factory=dict)  # optional per-frame overrides


class FrameSource(ABC):
    """A stream of frames. Iterate to capture; close() to release hardware."""
    @abstractmethod
    def __iter__(self) -> Iterator[RawFrame]:
        ...

    def close(self) -> None:
        pass


class MockFrameSource(FrameSource):
    """Replays a fixed list of RawFrames — for unit tests / dry runs."""
    def __init__(self, frames: list[RawFrame]):
        self._frames = frames

    def __iter__(self) -> Iterator[RawFrame]:
        return iter(self._frames)


class OakDFrameSource(FrameSource):
    """OAK-D Lite RGB(+depth) source via depthai v3. Rig-time only (lazy import,
    not unit-tested here — that's the one part the mock cannot cover).

    Mirrors scripts/capture_image.py's depthai v3 usage; yields one RawFrame per
    trigger. depth is populated only when want_depth=True (in_situ_multiview).
    """
    def __init__(self, want_depth: bool = False, n_frames: Optional[int] = None):
        self.want_depth = want_depth
        self.n_frames = n_frames

    def __iter__(self) -> Iterator[RawFrame]:        # pragma: no cover - needs camera
        import depthai as dai   # lazy; only present on the rig
        raise NotImplementedError(
            "OakDFrameSource is a rig-time skeleton — wire to the depthai v3 "
            "pipeline from scripts/capture_image.py (RGB + optional stereo depth) "
            "and yield RawFrame(image=bgr, depth=depth_mm, view_angle_deg=...).")


# ─────────────────────────────────────────────────────────────────────────────
# Session
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class SessionStats:
    kept: int = 0
    rejected: int = 0
    warned: int = 0
    kept_paths: list = field(default_factory=list)
    rejected_paths: list = field(default_factory=list)


def _default_stem(index: int, result: cp.ComplianceResult) -> str:
    qr = result.details.get("qr", {})
    sid = qr.get("canonical") if qr.get("valid") else "NOID"
    return f"{sid}_{index:04d}"


class CaptureSession:
    """Drives a FrameSource through the compliance gate and persists results.

    Args:
        source:        a FrameSource.
        mode:          capture mode (one of cp.MODES).
        kept_dir/quarantine_dir: output dirs.
        manifest / manifest_mode: passed through to validate_frame.
        calibration:   loaded intrinsics (cp.load_calibration); if given, frames
                       are undistorted before validation and fx is taken from it.
        stem_fn:       (index, result) -> filename stem.
        base_meta_fn:  (index, frame, result) -> dict of base sidecar metadata.
        on_status:     (status_str, result) callback for the live operator view;
                       defaults to print (warnings already render LOUD).
        require_confirmed: enforce the marker-size session guard (default True).
    """
    def __init__(self, source: FrameSource, mode: str, kept_dir, quarantine_dir,
                 *, manifest: Optional[dict] = None, manifest_mode: str = "warn",
                 calibration: Optional[dict] = None,
                 stem_fn: Callable = _default_stem,
                 base_meta_fn: Optional[Callable] = None,
                 on_status: Optional[Callable] = None,
                 require_confirmed: bool = True):
        if mode not in cp.MODES:
            raise ValueError(f"mode must be one of {cp.MODES}")
        self.source = source
        self.mode = mode
        self.kept_dir = Path(kept_dir)
        self.quarantine_dir = Path(quarantine_dir)
        self.manifest = manifest
        self.manifest_mode = manifest_mode
        self.calibration = calibration
        self.stem_fn = stem_fn
        self.base_meta_fn = base_meta_fn
        self.on_status = on_status or (lambda s, r: print(s))
        self.require_confirmed = require_confirmed
        self._fx = (float(calibration["camera_matrix"][0, 0])
                    if calibration is not None else None)

    def _base_meta(self, index, frame: RawFrame, result) -> dict:
        if self.base_meta_fn:
            return self.base_meta_fn(index, frame, result)
        qr = result.details.get("qr", {})
        return {
            "capture_mode": self.mode,
            "sample_id": qr.get("canonical") if qr.get("valid") else None,
            "view_angle_deg": frame.view_angle_deg,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            **frame.meta,
        }

    def run(self) -> SessionStats:
        # Session-start precondition: marker sizes must be confirmed.
        if self.require_confirmed:
            cp.start_capture_session()

        stats = SessionStats()
        try:
            for i, frame in enumerate(self.source):
                img = frame.image
                if self.calibration is not None:
                    img = cp.undistort(img, self.calibration)

                result = cp.validate_frame(
                    img, self.mode, depth=frame.depth, fx_px=self._fx,
                    manifest=self.manifest, manifest_mode=self.manifest_mode)

                stem = self.stem_fn(i, result)
                path = cp.persist_or_quarantine(
                    img, result, self._base_meta(i, frame, result),
                    self.kept_dir, self.quarantine_dir, stem)

                self.on_status(cp.format_compliance(result), result)
                if result.passed:
                    stats.kept += 1
                    stats.kept_paths.append(path)
                else:
                    stats.rejected += 1
                    stats.rejected_paths.append(path)
                if result.warnings:
                    stats.warned += 1
        finally:
            self.source.close()
        return stats

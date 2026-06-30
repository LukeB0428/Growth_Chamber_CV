"""
scale.py — pixels-per-millimetre conversion for a capture session.

Promoted verbatim from brassica_pods.common (re-exported there for compat). Every
capture session must record a px->mm scale from a fiducial of known size, or all
size metrics are meaningless. Never hard-code a scale — working-distance drift
invalidates it.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class PixelScale:
    px_per_mm: float
    source: str = ""          # e.g. "ruler 50mm = 412px" or "ArUco 30mm tag"

    @property
    def mm_per_px(self) -> float:
        return 1.0 / self.px_per_mm

    def length_mm(self, length_px: float) -> float:
        return length_px * self.mm_per_px

    def area_mm2(self, area_px: float) -> float:
        return area_px * (self.mm_per_px ** 2)

    @classmethod
    def from_fiducial(cls, fiducial_px: float, fiducial_mm: float,
                      source: str = "") -> "PixelScale":
        """Build a scale from a measured fiducial: known length in px and mm."""
        if fiducial_px <= 0 or fiducial_mm <= 0:
            raise ValueError("fiducial_px and fiducial_mm must be positive")
        return cls(px_per_mm=fiducial_px / fiducial_mm,
                   source=source or f"{fiducial_mm}mm = {fiducial_px}px")

    def to_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(
            {"px_per_mm": self.px_per_mm, "source": self.source}, indent=2))

    @classmethod
    def from_json(cls, path: str | Path) -> "PixelScale":
        d = json.loads(Path(path).read_text())
        return cls(px_per_mm=float(d["px_per_mm"]), source=d.get("source", ""))

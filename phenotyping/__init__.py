"""
phenotyping — shared CV engine for counting + sizing repeated plant structures.

Promoted from brassica_pods (whose README flagged common.py as "the natural seed
for a promoted top-level /common library"). Species modules are thin wrappers over
detect_and_count():
  - brassica_pods : YOLO-seg, class "pod"    (siliques)
  - barley        : YOLO-det, class "spike"  (cereal heads, GWHD-pretrained)

The Arabidopsis pipeline in scripts/ is untouched; greenness is reused by reference.
"""
from phenotyping.scale import PixelScale
from phenotyping.object_count import (
    detect_and_count, instance_size, instance_greenness,
)

__all__ = ["PixelScale", "detect_and_count", "instance_size", "instance_greenness"]

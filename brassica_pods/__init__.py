"""
brassica_pods — Brassica seed-pod (silique) instance segmentation + counting.

Standalone module. Does NOT modify the thesis-critical Arabidopsis pipeline in
scripts/. Shared infrastructure (SAM2 weights, project paths) is reused by
reference via brassica_pods.common — never by editing scripts/.

Public entry point:
    from brassica_pods.infer.analyze import analyze
    result = analyze(image)   # -> {count, masks, per_pod: [{length_mm, width_mm, area_mm2}]}
"""

__version__ = "0.1.0"

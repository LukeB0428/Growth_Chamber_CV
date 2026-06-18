"""
train_seg.py — YOLO-seg trainer for Brassica pods.

╔════════════════════════════════════════════════════════════════════════════╗
║  PHASE 0 STOP POINT.                                                         ║
║  This is a DELIBERATE PLACEHOLDER, not the real trainer. Per the kickoff    ║
║  brief, the scaffold + Phase 1 capture plan must be approved before the     ║
║  trainer is written. The recommended approach is sketched below so it is    ║
║  ready to flesh out on approval.                                            ║
╚════════════════════════════════════════════════════════════════════════════╝

PLANNED APPROACH (Phase 4):
  - Base model: yolo11n-seg.pt (nano) for a fast CPU-runnable counter; step up
    to yolo11s/m-seg if accuracy on dense clusters needs it.
  - Training runs on GPU/Colab (local torch is CPU-only — confirmed Phase 0).
    Inference runs locally on CPU via infer/analyze.py.
  - Bootstrap on data/synthetic, then fine-tune on data/dataset (real+synthetic)
    via active learning: label the hardest real failures, add, retrain.

  The actual call will be roughly:
      from ultralytics import YOLO
      YOLO("yolo11n-seg.pt").train(data="pods.yaml", epochs=100, imgsz=1280,
                                   batch=8, device=0)
  imgsz is large (1280) on purpose — siliques are thin and small relative to
  the frame, so resolution matters more than depth here.
"""

import sys


def main():
    print(__doc__)
    print("\n[train_seg.py is a Phase 0 placeholder — trainer not yet implemented]")
    print("Approve the scaffold + Phase 1 capture plan, then this gets built.")
    sys.exit(0)


if __name__ == "__main__":
    main()

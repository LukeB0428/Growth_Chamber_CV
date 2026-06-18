# Colab training — Brassica pod YOLO11-seg

`train_pods_colab.ipynb` trains the pod segmenter on a GPU. It is **self-contained**.

## Important design note
The real deepcanola scan images have **no segmentation masks** (their COCO file
lists images but zero annotations). So they **cannot be training data**. We follow
deepcanola's own method:
- **Train** on semi-synthetic images (real pod cut-outs composited onto
  backgrounds → exact YOLO-seg masks).
- **Evaluate pod COUNT** against the real images, using the manual measurements
  in `POD SCAN DATA.csv` (rows-per-barcode = measured pod count per image).

This is why the notebook has no "convert br017.json → masks" step — there are no
masks in it to convert.

## How to run
1. Open `train_pods_colab.ipynb` in Google Colab.
2. **Runtime ▸ Change runtime type ▸ GPU**.
3. **Runtime ▸ Run all.**

The notebook will: download the 25 MB data pools from Zenodo → regenerate the
synthetic dataset on the Colab box → train YOLO11-seg → download the 1.4 GB real
set → report count MAE → let you download `pods_best.pt`.

No upload needed. (~10 min data build + ~1 h training on a T4 at the default
1000 images / 80 epochs / imgsz 1536.)

### Optional: skip the in-Colab regeneration
A ready dataset is also generated locally at `brassica_pods/data/dataset/`
(zipped to `brassica_pods/colab/pods_dataset.zip` if present). Upload that zip to
Colab/Drive and unzip to `dataset/` to skip the build cell — same result.

## After training
Drop `pods_best.pt` into `brassica_pods/weights/`, then locally:
```python
from brassica_pods.infer.analyze import analyze
from brassica_pods.common import PixelScale
r = analyze("img.jpg", scale=PixelScale.from_fiducial(412, 50),
            weights="brassica_pods/weights/pods_best.pt", greenness=True)
print(r["count"], r["per_pod"][:2])
```

## Reality check
Trained on **excised-pods-on-black** synthetic scenes → will **not** transfer
cleanly to **whole-plant in-situ** rig images. That gap closes by fine-tuning on
a small set of real rig captures later (active learning). This run validates the
full machinery (synth → train → infer → count-eval) end-to-end on real B. napus.

Data: deepcanola, CC-BY-4.0, Zenodo 10.5281/zenodo.13903900 — cite in any output.

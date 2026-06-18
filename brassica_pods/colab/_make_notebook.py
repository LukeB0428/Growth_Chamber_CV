"""
_make_notebook.py — generate the self-contained Colab training notebook.

Run locally:  python brassica_pods/colab/_make_notebook.py
Produces:     brassica_pods/colab/train_pods_colab.ipynb

The notebook is deliberately self-contained (no repo clone, no big upload): it
downloads the deepcanola data pools from Zenodo, regenerates the synthetic
YOLO-seg dataset on the Colab box, trains YOLO11-seg on GPU, then evaluates
COUNT against the real 'novel' images. The embedded synth/eval logic mirrors
brassica_pods/synth/generate_dataset.py + eval/build_ground_truth.py — keep them
in sync if the canonical module changes.
"""
import json
from pathlib import Path

def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}

def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": text.strip("\n").splitlines(keepends=True)}

cells = []

cells.append(md(
"""# Brassica napus pod (silique) segmentation — YOLO11-seg training

**Self-contained.** Runtime ▸ *Change runtime type* ▸ **GPU**, then Runtime ▸ **Run all**.

**Design (matches the deepcanola method):** the real scan images have *no masks*
(their COCO file lists images but zero annotations), so they cannot be training
data. We therefore **train on semi-synthetic images** (real pod cut-outs composited
onto backgrounds → exact YOLO-seg masks) and **evaluate pod COUNT against the real
images** using the manual measurements CSV.

Data: deepcanola, *Brassica napus*, **CC-BY-4.0**
(Atkins et al., Zenodo [10.5281/zenodo.13903900](https://doi.org/10.5281/zenodo.13903900)).
Cite it in any paper/figure. The deepcanola *code* is GPL-3.0 — this notebook
**reimplements the methodology**, it does not copy that code.
"""))

cells.append(md("## 0. Config — tweak these"))
cells.append(code(
"""
N_SYNTH_IMAGES = 1000      # synthetic training images to generate
MIN_PODS, MAX_PODS = 10, 55
VAL_FRAC = 0.15
DOWNSCALE = 0.44           # shrink ~3500px scans toward training resolution
SEED = 42

MODEL = "yolo11n-seg.pt"   # nano: fast + CPU-runnable inference. Step up to s/m if needed.
EPOCHS = 80
IMGSZ = 1536               # siliques are thin/small -> keep resolution high
BATCH = 8
"""))

cells.append(md("## 1. GPU check + install"))
cells.append(code(
"""
import subprocess, torch
print(subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True).stdout or "NO GPU")
print("torch", torch.__version__, "CUDA", torch.cuda.is_available())
assert torch.cuda.is_available(), "Set Runtime > Change runtime type > GPU, then Run all."
"""))
cells.append(code(
"""
%pip -q install ultralytics
import ultralytics; ultralytics.__version__
"""))

cells.append(md("## 2. Download the deepcanola data pools (25 MB) from Zenodo"))
cells.append(code(
"""
import os, urllib.request, zipfile
os.makedirs("dc", exist_ok=True)
GT_URL = "https://zenodo.org/api/records/13903900/files/ground_truth_data.zip/content"
if not os.path.exists("dc/ground_truth_data.zip"):
    print("downloading ground_truth_data.zip ...")
    urllib.request.urlretrieve(GT_URL, "dc/ground_truth_data.zip")
with zipfile.ZipFile("dc/ground_truth_data.zip") as z:
    z.extractall("dc")
GT_ROOT = "dc/ground_truth_data"
print("pools:", os.listdir(GT_ROOT))
"""))

cells.append(md(
"""## 3. Build the synthetic YOLO-seg dataset
Embedded compositor — mirrors `brassica_pods/synth/generate_dataset.py`."""))
cells.append(code(
'''
import re, glob, random
import cv2, numpy as np
from pathlib import Path

POD_RE = re.compile(r"^(?P<scan>.+?)_(?:mask_)?pod(?P<id>\\d+)$", re.I)
def key(stem):
    m = POD_RE.match(stem); return (m.group("scan"), m.group("id")) if m else None

def stage_sprites(gt_root, downscale):
    """Pair pod cut-outs with masks -> list of (bgr, alpha) at `downscale`."""
    sprites = []
    for set_dir in sorted(Path(gt_root, "pod_pools").iterdir()):
        idir, mdir = set_dir/"images", set_dir/"masks"
        if not (idir.is_dir() and mdir.is_dir()): continue
        masks = {key(p.stem): p for p in mdir.glob("*.jpg") if key(p.stem)}
        for ip in sorted(idir.glob("*.jpg")):
            k = key(ip.stem)
            if not k or k not in masks: continue
            bgr = cv2.imread(str(ip)); al = cv2.imread(str(masks[k]), 0)
            if bgr is None or al is None: continue
            if al.shape[:2] != bgr.shape[:2]:
                al = cv2.resize(al, (bgr.shape[1], bgr.shape[0]), interpolation=cv2.INTER_NEAREST)
            if downscale != 1.0:
                f = downscale
                bgr = cv2.resize(bgr, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
                al  = cv2.resize(al,  None, fx=f, fy=f, interpolation=cv2.INTER_NEAREST)
            al = (al > 127).astype(np.uint8)*255
            if al.sum() == 0: continue
            sprites.append((bgr, al))
    return sprites

def load_bgs(gt_root, downscale):
    out = []
    for p in glob.glob(str(Path(gt_root,"background_pool","*"))):
        b = cv2.imread(p)
        if b is None: continue
        if downscale != 1.0:
            b = cv2.resize(b, None, fx=downscale, fy=downscale, interpolation=cv2.INTER_AREA)
        out.append(b)
    return out

def transform(bgr, al, scale, angle):
    h, w = al.shape[:2]
    nw, nh = max(1,int(w*scale)), max(1,int(h*scale))
    bgr = cv2.resize(bgr,(nw,nh),interpolation=cv2.INTER_AREA)
    al  = cv2.resize(al,(nw,nh),interpolation=cv2.INTER_NEAREST)
    M = cv2.getRotationMatrix2D((nw/2,nh/2), angle, 1.0)
    cos, sin = abs(M[0,0]), abs(M[0,1])
    bw, bh = int(nh*sin+nw*cos), int(nh*cos+nw*sin)
    M[0,2]+=(bw-nw)/2; M[1,2]+=(bh-nh)/2
    bgr = cv2.warpAffine(bgr,M,(bw,bh),flags=cv2.INTER_LINEAR,borderValue=(0,0,0))
    al  = cv2.warpAffine(al, M,(bw,bh),flags=cv2.INTER_NEAREST,borderValue=0)
    return bgr, (al>127).astype(np.uint8)*255

def mask_to_poly(m):
    cs,_ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cs: return None
    c = max(cs, key=cv2.contourArea)
    if cv2.contourArea(c) < 80: return None
    eps = 0.004*cv2.arcLength(c, True)
    a = cv2.approxPolyDP(c, eps, True).reshape(-1,2)
    return a if len(a) >= 3 else None

def compose(bg, sprites, rng, min_p, max_p, max_overlap=0.6):
    H,W = bg.shape[:2]; canvas = bg.copy()
    occ = np.zeros((H,W),np.uint8); placed=[]
    for _ in range(rng.randint(min_p,max_p)):
        bgr,al = sprites[rng.randrange(len(sprites))]
        bgr,al = transform(bgr,al, rng.uniform(0.6,1.4), rng.uniform(0,360))
        ph,pw = al.shape[:2]
        if ph>=H or pw>=W: continue
        x,y = rng.randint(0,W-pw), rng.randint(0,H-ph)
        fm = np.zeros((H,W),np.uint8); fm[y:y+ph,x:x+pw]=al
        area=int((fm>0).sum())
        if area==0: continue
        if int(np.logical_and(fm>0,occ>0).sum())/area > max_overlap: continue
        a3 = (al[:,:,None]/255.0)
        roi = canvas[y:y+ph,x:x+pw].astype(np.float32)
        canvas[y:y+ph,x:x+pw] = (a3*bgr.astype(np.float32)+(1-a3)*roi).astype(np.uint8)
        for pm in placed: pm[fm>0]=0
        placed.append(fm); occ[fm>0]=255
    polys=[p for m in placed if (p:=mask_to_poly(m)) is not None]
    return canvas, polys

def build_dataset(gt_root, out, n, min_p, max_p, val_frac, downscale, seed):
    sprites = stage_sprites(gt_root, downscale); bgs = load_bgs(gt_root, downscale)
    print(f"{len(sprites)} sprites, {len(bgs)} backgrounds")
    rng = random.Random(seed); n_val=int(round(n*val_frac)); pods=0
    for i in range(n):
        split = "val" if i < n_val else "train"
        comp, polys = compose(bgs[rng.randrange(len(bgs))], sprites, rng, min_p, max_p)
        h,w = comp.shape[:2]; stem=f"synth_{i:05d}"
        idir = Path(out,"images",split); ldir = Path(out,"labels",split)
        idir.mkdir(parents=True,exist_ok=True); ldir.mkdir(parents=True,exist_ok=True)
        cv2.imwrite(str(idir/f"{stem}.jpg"), comp)
        lines=[]
        for p in polys:
            nrm=p.astype(np.float32).copy(); nrm[:,0]/=w; nrm[:,1]/=h
            nrm=np.clip(nrm,0,1)
            lines.append("0 "+" ".join(f"{v:.6f}" for v in nrm.reshape(-1)))
        (ldir/f"{stem}.txt").write_text("\\n".join(lines))
        pods+=len(polys)
        if (i+1)%100==0: print(f"  {i+1}/{n}")
    print(f"done: {n} images, {pods} pods")

build_dataset(GT_ROOT, "dataset", N_SYNTH_IMAGES, MIN_PODS, MAX_PODS, VAL_FRAC, DOWNSCALE, SEED)

import yaml
yaml.safe_dump({"path":"dataset","train":"images/train","val":"images/val","names":{0:"pod"}},
               open("pods.yaml","w"))
print(open("pods.yaml").read())
'''))

cells.append(md("## 4. Sanity-check a few labels"))
cells.append(code(
"""
import glob, cv2, numpy as np
from matplotlib import pyplot as plt
imgs = sorted(glob.glob("dataset/images/train/*.jpg"))[:3]
fig,ax = plt.subplots(1,len(imgs),figsize=(15,7))
for a,ip in zip(np.atleast_1d(ax), imgs):
    im=cv2.cvtColor(cv2.imread(ip),cv2.COLOR_BGR2RGB); h,w=im.shape[:2]
    lp=ip.replace("images","labels").replace(".jpg",".txt")
    n=0
    for line in open(lp):
        v=line.split()
        if len(v)<7: continue
        xy=np.array(v[1:],float).reshape(-1,2)*[w,h]
        cv2.polylines(im,[xy.astype(np.int32)],True,(0,255,0),2); n+=1
    a.imshow(im); a.set_title(f"{n} pods"); a.axis("off")
plt.tight_layout(); plt.show()
"""))

cells.append(md("## 5. Train YOLO11-seg"))
cells.append(code(
"""
from ultralytics import YOLO
model = YOLO(MODEL)
results = model.train(data="pods.yaml", epochs=EPOCHS, imgsz=IMGSZ, batch=BATCH,
                      device=0, patience=20, name="brassica_pods")
best = results.save_dir + "/weights/best.pt"
print("best weights:", best)
"""))

cells.append(md(
"""## 6. Evaluate COUNT on the real images
Download the real 'novel' set + manual measurements, build per-image count
ground truth (rows per barcode), run inference, report MAE / mean % error."""))
cells.append(code(
'''
import urllib.request, zipfile, os, glob, csv, re
VAL_URL = "https://zenodo.org/api/records/13903900/files/validation_data.zip/content"
if not os.path.exists("dc/validation_data.zip"):
    print("downloading validation_data.zip (~1.4 GB) ...")
    urllib.request.urlretrieve(VAL_URL, "dc/validation_data.zip")
    with zipfile.ZipFile("dc/validation_data.zip") as z: z.extractall("dc")
NOVEL = "dc/novel_data/br017"

def num(s):
    s=(s or "").strip().replace(",", ".")
    try: return float(s)
    except: return None
BC = re.compile(r"-(\\d{4,})\\s")
rows=list(csv.DictReader(open(glob.glob(NOVEL+"/*POD SCAN DATA*.csv")[0], encoding="utf-8-sig")))
by_bc={}
for r in rows:
    bc=(r.get("Barcode") or "").strip()
    if bc: by_bc.setdefault(bc,[]).append(r)
gt={}
for ip in glob.glob(NOVEL+"/images/*.jpg"):
    m=BC.search(os.path.basename(ip))
    if m and m.group(1) in by_bc: gt[ip]=len(by_bc[m.group(1)])
print("images with count GT:", len(gt))
'''))
cells.append(code(
"""
import numpy as np
from ultralytics import YOLO
model = YOLO(best)
errs, preds, gts = [], [], []
items = list(gt.items())
for ip, g in items:
    r = model.predict(ip, imgsz=IMGSZ, conf=0.25, verbose=False)[0]
    n = 0 if r.masks is None else len(r.masks)
    preds.append(n); gts.append(g); errs.append(abs(n-g))
preds, gts, errs = map(np.array, (preds, gts, errs))
mae = errs.mean()
mape = (np.abs(preds-gts)/np.clip(gts,1,None)).mean()*100
print(f"n={len(items)}  count MAE={mae:.2f}  mean%err={mape:.1f}%")
print(f"pred mean={preds.mean():.1f}  gt mean={gts.mean():.1f}")
print("NOTE: GT = manually *measured* pods (often a subset), so it is a")
print("reference, not an exhaustive count. Expect a positive bias if the model")
print("counts pods the scorers did not measure.")
"""))
cells.append(code(
"""
from matplotlib import pyplot as plt
plt.figure(figsize=(5,5))
plt.scatter(gts, preds, alpha=0.5)
lim=[0, max(preds.max(), gts.max())+2]
plt.plot(lim, lim, 'r--'); plt.xlim(lim); plt.ylim(lim)
plt.xlabel("manual count (GT)"); plt.ylabel("predicted count")
plt.title(f"Count agreement (MAE={mae:.1f})"); plt.grid(True, alpha=0.3); plt.show()
"""))

cells.append(md("## 7. Visualize predictions on real images"))
cells.append(code(
"""
import random as _r
from matplotlib import pyplot as plt
sample = _r.Random(0).sample(list(gt), 3)
fig,ax = plt.subplots(1,3,figsize=(16,8))
for a,ip in zip(ax, sample):
    r = model.predict(ip, imgsz=IMGSZ, conf=0.25, verbose=False)[0]
    a.imshow(r.plot()[:,:,::-1]); a.axis("off")
    a.set_title(f"pred {0 if r.masks is None else len(r.masks)} / gt {gt[ip]}")
plt.tight_layout(); plt.show()
"""))

cells.append(md("## 8. Export weights + results"))
cells.append(code(
"""
import shutil
from google.colab import files
shutil.copy(best, "pods_best.pt")
print("Download pods_best.pt and drop it into brassica_pods/weights/ locally.")
print("Then: analyze(img, weights='brassica_pods/weights/pods_best.pt', greenness=True)")
files.download("pods_best.pt")
"""))

cells.append(md(
"""## Next steps
- This model is trained on **excised-pods-on-black** synthetic scenes. It will
  **not** transfer cleanly to your eventual **whole-plant in-situ** rig — that
  gap closes by fine-tuning on a small set of real rig captures (active learning).
- To push count accuracy: raise `N_SYNTH_IMAGES`, add real-rig backgrounds to the
  background pool, and step `MODEL` up to `yolo11s-seg.pt` / `yolo11m-seg.pt`.
- Drop `pods_best.pt` into `brassica_pods/weights/` to use the local
  `analyze()` (count + per-pod size + greenness)."""))

nb = {
    "cells": cells,
    "metadata": {
        "accelerator": "GPU",
        "colab": {"provenance": [], "toc_visible": True},
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4, "nbformat_minor": 0,
}

out = Path(__file__).resolve().parent / "train_pods_colab.ipynb"
out.write_text(json.dumps(nb, indent=1))
print(f"wrote {out} ({len(cells)} cells)")

"""
_make_sam3_test.py — generate a Kaggle notebook that tests SAM 3 concept-prompting
("seed pod") for pod counting against our exhaustive hand-counts.

Run:      python brassica_pods/colab/_make_sam3_test.py
Produces: brassica_pods/colab/sam3_concept_test_kaggle.ipynb

Goal: decide ONE thing — does text-prompting a niche object ("seed pod") on our
real scans count accurately enough to use SAM 3 as an AUTO-LABELLER for real-rig
data, or does the fine-grained concept underperform (so we keep the trained
YOLO-seg)? Compares SAM 3's count to the 12 hand-counts.

Notes baked in:
- SAM 3 needs Ultralytics >= 8.3.237 and a GPU (it is ~3.45 GB / slow on CPU).
- SAM 3 weights may NOT auto-download — the load cell flags that clearly.
- The result object structure is not fully documented, so we INTROSPECT it on
  one image before looping (defensive — don't burn the session on a wrong API).
"""
import json
from pathlib import Path


def md(t):
    return {"cell_type": "markdown", "metadata": {}, "source": t.splitlines(keepends=True)}


def code(t):
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": t.strip("\n").splitlines(keepends=True)}


cells = []

cells.append(md(
"""# SAM 3 concept-prompt test — can "seed pod" count our pods?

**One question:** does SAM 3 text-prompting (`"seed pod"`) count pods accurately
on our real B. napus scans, vs our exhaustive hand-counts?
- **Accurate** → use SAM 3 as an auto-labeller for real-rig data (collapses hand-labelling).
- **Poor** (likely if the niche/fine-grained concept underperforms) → keep the trained YOLO-seg.

**Setup:** Settings ▸ Accelerator ▸ **GPU**, Settings ▸ Internet ▸ **On**, then Run All.
SAM 3 is ~3.45 GB and slow — GPU required. Needs Ultralytics ≥ 8.3.237.

**⚠️ PREREQUISITE — SAM 3 weights are GATED:**
1. On Hugging Face, **request access** at https://huggingface.co/facebook/sam3 and
   wait for approval (can take minutes–hours).
2. Create a token at https://huggingface.co/settings/tokens (read scope).
3. In Kaggle: **Add-ons ▸ Secrets ▸ add a secret named `HF_TOKEN`** = that token.
The weights cell below uses it to download `sam3.pt`."""))

cells.append(md("## 1. Install + GPU check"))
cells.append(code(
"""
import subprocess, torch
print(subprocess.run(["nvidia-smi","-L"], capture_output=True, text=True).stdout or "NO GPU")
print("torch", torch.__version__, "CUDA", torch.cuda.is_available())
assert torch.cuda.is_available(), "Enable a GPU (Settings > Accelerator)."
"""))
cells.append(code(
"""
%pip -q install -U ultralytics
import ultralytics
print("ultralytics", ultralytics.__version__)
from packaging import version
assert version.parse(ultralytics.__version__) >= version.parse("8.3.237"), "Need ultralytics >= 8.3.237 for SAM 3"
"""))

cells.append(md(
"""## 1b. Download the gated SAM 3 weights from Hugging Face
Needs your approved access to `facebook/sam3` + an `HF_TOKEN` Kaggle secret (see
prerequisite above). Lists the repo files first (the exact checkpoint name isn't
public), then downloads the `.pt` and places it as `sam3.pt`."""))
cells.append(code(
'''
import os, shutil
from huggingface_hub import login, list_repo_files, hf_hub_download
try:
    from kaggle_secrets import UserSecretsClient
    HF_TOKEN = UserSecretsClient().get_secret("HF_TOKEN")
except Exception:
    HF_TOKEN = ""          # fallback: paste your token here -> HF_TOKEN = "hf_..."
assert HF_TOKEN, "Add a Kaggle secret named HF_TOKEN (Add-ons > Secrets)."
login(token=HF_TOKEN)

REPO = "facebook/sam3"
files = list_repo_files(REPO)
print("repo files:", files)
pts = [f for f in files if f.endswith(".pt")]
fname = "sam3.pt" if "sam3.pt" in files else (pts[0] if pts else None)
assert fname, f"No .pt checkpoint found in {REPO}; pick from the list above."
path = hf_hub_download(repo_id=REPO, filename=fname)
shutil.copy(path, "sam3.pt")
print(f"downloaded {fname} -> sam3.pt ({os.path.getsize('sam3.pt')/1e9:.2f} GB)")
# also grab a BPE vocab if the repo ships one (needed for text prompts)
for f in files:
    if "vocab" in f.lower() and f.endswith((".gz", ".txt")):
        shutil.copy(hf_hub_download(repo_id=REPO, filename=f), os.path.basename(f))
        print("fetched vocab:", f)
'''))

cells.append(md("## 2. Hand-count ground truth (the 12 scans you counted)"))
cells.append(code(
'''
HANDCOUNTS = {
    "BR017-212122 001.jpg": 1,  "BR017-102213 001.jpg": 20, "BR017-102223 001.jpg": 20,
    "BR017-512213 001.jpg": 20, "BR017-258213 001.jpg": 20, "BR017-531213 001.jpg": 20,
    "BR017-193121 001.jpg": 19, "BR017-230111 001.jpg": 20, "BR017-513212 001.jpg": 20,
    "BR017-168122 001.jpg": 20, "BR017-237212 001.jpg": 20, "BR017-239212 001.jpg": 20,
}
print(len(HANDCOUNTS), "images, true counts:", sorted(HANDCOUNTS.values()))
'''))

cells.append(md(
"""## 3. Fetch just those 12 images from Zenodo
Downloads the deepcanola validation set and extracts ONLY the 12 needed images."""))
cells.append(code(
'''
import os, urllib.request, zipfile
os.makedirs("imgs", exist_ok=True)
VAL_URL = "https://zenodo.org/api/records/13903900/files/validation_data.zip/content"
if not os.path.exists("validation_data.zip"):
    print("downloading validation_data.zip (~1.4 GB)...")
    urllib.request.urlretrieve(VAL_URL, "validation_data.zip")
with zipfile.ZipFile("validation_data.zip") as z:
    names = z.namelist()
    for img in HANDCOUNTS:
        hit = [n for n in names if n.endswith("/" + img) or n.endswith(img)]
        if hit:
            with z.open(hit[0]) as src, open(f"imgs/{img}", "wb") as dst:
                dst.write(src.read())
got = sorted(os.listdir("imgs"))
print(f"extracted {len(got)}/12 images")
'''))

cells.append(md(
"""## 4. Load SAM 3 + INTROSPECT the result on one image
SAM 3's result object isn't fully documented — so before looping all 12 we run
ONE image and print what comes back, then derive a robust instance-count helper.
If the model fails to load, the weights likely need a manual fetch — see
https://docs.ultralytics.com/models/sam-3 ."""))
cells.append(code(
'''
from ultralytics.models.sam import SAM3SemanticPredictor
overrides = dict(conf=0.25, task="segment", mode="predict", model="sam3.pt", save=False)
predictor = SAM3SemanticPredictor(overrides=overrides)

import glob
probe = sorted(glob.glob("imgs/*.jpg"))[0]
predictor.set_image(probe)
res = predictor(text=["seed pod"])
print("type:", type(res))
r = res[0] if isinstance(res, (list, tuple)) and len(res) else res
print("repr (truncated):", str(r)[:400])
for attr in ("masks", "boxes"):
    v = getattr(r, attr, None)
    print(f"  .{attr}:", type(v).__name__, ("len=" + str(len(v))) if v is not None else "None")
'''))

cells.append(md("## 5. Robust instance-count helper (adapt here if §4 shows a different shape)"))
cells.append(code(
'''
def count_instances(res):
    """Best-effort instance count from SAM 3 output (handles list/Results/tuple)."""
    r = res[0] if isinstance(res, (list, tuple)) and len(res) else res
    for attr in ("masks", "boxes"):
        v = getattr(r, attr, None)
        if v is not None:
            try:
                return len(v)
            except TypeError:
                d = getattr(v, "data", None)
                if d is not None:
                    return int(d.shape[0])
    try:
        return len(res[0])     # tuple (masks, boxes)
    except Exception:
        return 0
print("helper ready")
'''))

cells.append(md(
"""## 6. Run prompts × 12 images → compare to hand-counts
Tries several phrasings of the niche concept; reports MAE per prompt."""))
cells.append(code(
'''
import glob, numpy as np
PROMPTS = ["seed pod", "pod", "silique", "bean pod"]
imgs = sorted(glob.glob("imgs/*.jpg"))
true = np.array([HANDCOUNTS[os.path.basename(p)] for p in imgs])
results = {}
for prompt in PROMPTS:
    preds = []
    for p in imgs:
        predictor.set_image(p)
        preds.append(count_instances(predictor(text=[prompt])))
    preds = np.array(preds)
    mae = float(np.mean(np.abs(preds - true)))
    results[prompt] = (preds, mae)
    print(f'prompt={prompt!r:12}  MAE={mae:5.1f}  preds={list(preds)}')
print("\\ntrue:", list(true))
best = min(results, key=lambda k: results[k][1])
print(f"\\nBEST prompt: {best!r}  (MAE {results[best][1]:.1f})  vs YOLO-seg MAE 0.3")
'''))

cells.append(md("## 7. Save a few annotated previews for the best prompt"))
cells.append(code(
'''
from ultralytics.utils.plotting import Annotator
import cv2
out = "/kaggle/working/sam3_previews"; os.makedirs(out, exist_ok=True)
for p in imgs[:4]:
    predictor.set_image(p)
    res = predictor(text=[best])
    r = res[0] if isinstance(res,(list,tuple)) and len(res) else res
    try:
        im = r.plot()
        cv2.imwrite(f"{out}/{os.path.basename(p)}", im)
    except Exception as e:
        print("plot failed:", e)
print("previews ->", out)
'''))

cells.append(md(
"""## Interpreting the result
- **Best MAE ≲ 2** → SAM 3 counts our niche pods well → adopt it as the
  **auto-labeller** for real-rig data (run on GPU, then train the small
  CPU-deployable YOLO-seg on its masks).
- **Best MAE ≫ a few** → the fine-grained concept underperforms → keep the
  trained YOLO-seg (MAE 0.3, 6 MB, CPU). No loss — this test cost ~1 GPU hour.
- Either way: SAM 3 does NOT solve the intact-plant occlusion problem (still the
  feasibility probe) and is too heavy (~3.45 GB) for local CPU deployment."""))

nb = {"cells": cells,
      "metadata": {"accelerator": "GPU",
                   "kernelspec": {"display_name": "Python 3", "name": "python3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 0}

out = Path(__file__).resolve().parent / "sam3_concept_test_kaggle.ipynb"
out.write_text(json.dumps(nb, indent=1))
print(f"wrote {out} ({len(cells)} cells)")

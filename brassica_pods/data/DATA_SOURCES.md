# Data sources & attribution

## deepcanola (Brassica napus) — primary reference dataset
- **Stored at:** `C:\Users\LukeB\brassica_data\deepcanola\` — deliberately OUTSIDE
  the OneDrive-synced repo (~28 GB would otherwise sync to the cloud) and never
  git-committed. Override path with the `BRASSICA_DATA_DIR` env var; the module
  resolves it via `brassica_pods.common.DEEPCANOLA_DIR`.
- **Source:** Atkins, K. et al. *DeepCanola: Phenotyping Brassica Pods Using
  Semi-Synthetic Data and Active Learning.* Zenodo. DOI:
  [10.5281/zenodo.13903900](https://doi.org/10.5281/zenodo.13903900)
- **Licence:** **CC-BY-4.0** — free to use with attribution. (The deepcanola
  *code*, separately, is GPL-3.0 — we reimplement its methodology, not its code.)
- **Cite this dataset** in any paper/figure that uses these images.

### Files
| File | Size | Use |
|---|---|---|
| `ground_truth_data.zip` | 25 MB | Manual pod counts / phenotype truth → eval/ |
| `deepcanola_outputs.zip` | 0.4 MB | Example model outputs (format reference) |
| `validation_data.zip` | 1.37 GB | Real B. napus pod images + masks → test images |
| `training_data.zip` | 26.7 GB | Their synthetic set + data pools (cut-outs/masks/backgrounds) |
| `deepcanola.pth` | 168 MB | deepcanola's own model — NOT loadable by our YOLO-seg `analyze()` |

Species match: **Brassica napus** = our confirmed target, so this is a direct
warm-start / baseline source, not just an analogue.

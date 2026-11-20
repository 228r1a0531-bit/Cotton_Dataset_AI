# CottonLeaf AI

CottonLeaf AI is a Streamlit dashboard for presenting research on hybrid deep
learning and cotton leaf disease classification. It covers the research
objective, dataset and training, image classification, model comparisons,
evaluation, and conceptual architecture.

## Install and run

From the repository root with Python 3.11, run:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

The dashboard starts without datasets, results, or model weights. Visit
**Upload & Classification** to preview a JPEG or PNG. To enable classification
with a compatible trained model, install the optional dependencies:

```bash
python -m pip install -r requirements-ml.txt
```

See [the research artifact contract](results/README.md) for checkpoint setup,
labels, benchmark citations, and the required formats for dataset counts,
training curves, and evaluation charts.

## Run checks

```bash
python -m pip check
python -m unittest discover -s tests -v
```

## Project layout

| Path | Purpose |
| --- | --- |
| `app.py`, `config.py` | Dashboard entry point and shared settings |
| `pages/` | Six research dashboard views |
| `models/` | Optional Keras checkpoint loading and inference |
| `training/` | Baseline and hybrid training pipeline |
| `utils/` | Shared styling, artifact loading, and image processing |
| `results/` | Local research artifacts and their documented formats |
| `tests/` | Dashboard and image-processing smoke tests |
| `.streamlit/config.toml` | Theme and server settings |

## Set up a dataset on Windows with PyCharm

1. Keep the dataset in a local folder such as `C:\Data\Cotton Leaves`. Give it
   four class subfolders: `bacterial_blight`, `curl_virus`,
   `fusarium_wilt` (or `fussarium_wilt`), and `healthy`. Names such as
   `Bacterial Blight` and `Leaf Curl Virus` also work. No download is required.
2. Open this repository in PyCharm and select a Python 3.11 virtual environment
   as the project interpreter. In the PyCharm terminal run
   `python -m pip install -r requirements.txt`. For training, also run
   `python -m pip install -r requirements-ml.txt`.
3. Run `python -m streamlit run app.py` from the repository root and open
   **Dataset & Training**. Paste the dataset root into the path field and choose
   **Scan dataset**. Alternatively, set
   `$env:COTTONLEAF_DATASET_ROOT = 'C:\Data\Cotton Leaves'` in PowerShell before
   launching Streamlit, or add the same environment variable in the PyCharm run
   configuration. A PyCharm Python run configuration can use module
   `streamlit`, parameters `run app.py`, and the repository as working directory.
   With no configured path, the app checks `datasets/`, `dataset/`, and
   `data/` and their immediate children.
4. Inspect the counts and unreadable paths. Save the splits, or select Fast Demo
   and start a short baseline run. Full hybrid training is also available. Training artifacts are written under
   `results/`; see [their formats](results/README.md).

The scanner fingerprints decoded, EXIF-oriented RGB pixels. Identical copies
within a class collapse to one representative. Exact duplicates with conflicting
labels are excluded until the labels are resolved. Recompressed or cropped
near-duplicates can require manual review. Each class needs at least three
eligible images to appear in all three splits.

## Model implementations

| Builder | Architecture |
| --- | --- |
| `models.cnn.build_cnn()` | Compact convolutional baseline with global average pooling and a dense head |
| `models.cnn.build_cnn(use_se=True)` | Matched CNN with channel-wise squeeze and excitation |
| `models.inception.build_inception_v3()` | Keras Applications InceptionV3 with a four-class head |
| `models.inception.build_modified_inception()` | Short 24/32/48-channel stem and two compact multi-scale blocks |
| `models.hybrid.build_hybrid()` | CN2-SE and modified Inception feature concatenation followed by a dropout-regularized MLP |

Standard InceptionV3 defaults to `weights=None`, so it can be built offline.
Pass `weights="imagenet"` explicitly to use Keras's cached or downloaded
ImageNet weights, or provide a path to compatible local backbone weights.
Explicitly supplied backbone weights are frozen by default. The modified
architecture uses a shorter stem, fewer blocks and narrower feature channels
than standard InceptionV3. It is research-inspired and does not reproduce a
paper's exact topology.

Install `requirements-ml.txt` to run model tests and training. Synthetic
forward-pass tests validate software behavior only. See [training](training/README.md)
for the baseline and hybrid workflows.

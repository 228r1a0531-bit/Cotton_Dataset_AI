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
| `training/` | Location for future training code |
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
   and start a short baseline run. Training artifacts are written under
   `results/`; see [their formats](results/README.md).

The scanner fingerprints decoded, EXIF-oriented RGB pixels. Identical copies
within a class collapse to one representative. Exact duplicates with conflicting
labels are excluded until the labels are resolved. Recompressed or cropped
near-duplicates can require manual review. Each class needs at least three
eligible images to appear in all three splits.

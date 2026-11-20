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

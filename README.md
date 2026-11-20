# CottonLeaf AI

CottonLeaf AI is a research demo for hybrid deep learning and cotton leaf disease
classification. The current Streamlit GUI previews uploaded JPEG or PNG images
and plots the RGB distribution. Disease predictions require a trained model,
its preprocessing, and its class labels. The demo starts without those assets.

## Install and run

From the repository root with Python 3.11, run:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Open the address printed by Streamlit, choose **Inspect a leaf image**, and
upload a JPEG or PNG. The chart describes the displayed preview.

Install the optional TensorFlow/Keras environment for checkpoint work:

```bash
python -m pip install -r requirements-ml.txt
```

The optional loader in `models/keras_io.py` accepts an existing `.keras`
checkpoint and loads it for inference. The GUI can run with the base requirements.

## Run checks

```bash
python -m pip check
python -m unittest discover -s tests -v
```

## Project layout

| Path | Purpose |
| --- | --- |
| `app.py`, `config.py` | Streamlit entry point and shared settings |
| `pages/` | Overview and image inspection |
| `models/` | Optional Keras checkpoint loading |
| `training/` | Location for future training code |
| `utils/` | Image decoding and analysis |
| `results/` | Local generated evaluation outputs |
| `tests/` | Streamlit and image-processing smoke tests |
| `.streamlit/config.toml` | Local Streamlit settings |

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

## Windows 10/11 and PyCharm

Use 64-bit Python 3.11. On Windows, install the Python launcher when installing
Python. Confirm that PowerShell can find the interpreter.

```powershell
py -3.11 --version
```

Open a terminal in the repository folder and run `.\run_app.bat`. The launcher
uses `.venv`, creates it when missing, and installs the core packages when
needed. Open the Streamlit URL printed in the terminal. Press Ctrl+C to stop
the app.

For a manual install, run these commands from the repository root. Calling
the virtual environment's Python directly also works when PowerShell blocks
activation scripts.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m streamlit run app.py --server.headless true
```

For checkpoint inference or training, install the optional packages into the
same interpreter with
`.\.venv\Scripts\python.exe -m pip install -r requirements-ml.txt`. The
native Windows setup is intended for CPU use. If port 8501 is already taken,
launch with `.\run_app.bat --server.port 8502` and open the updated URL.

### PyCharm run configuration

1. Open the repository folder in PyCharm. Under **Settings → Project →
   Python Interpreter**, select **Add Interpreter → Add Local Interpreter** and
   use the existing `.venv\Scripts\python.exe`. Create the environment with
   the PowerShell commands above if it does not exist.
2. Under **Run → Edit Configurations**, add a **Python** configuration.
   Select **Module name** and enter `streamlit`. Set **Parameters** to
   `run app.py --server.headless true`, **Working directory** to the
   repository root, and **Python interpreter** to the project's `.venv`.
3. Click **Run** and open the URL shown in PyCharm's Run console. To use a
   local dataset, add `COTTONLEAF_DATASET_ROOT` to the configuration's
   environment variables, or enter the dataset path in the dashboard.

### Optional dataset and checkpoint

Arrange your dataset in four class directories. The dashboard accepts
`fussarium_wilt` as an alias for `fusarium_wilt`, and recognizes names such
as `Bacterial Blight` and `Leaf Curl Virus`.

```text
C:\Data\Cotton Leaves\
  bacterial_blight\
  curl_virus\
  fusarium_wilt\
  healthy\
```

Set the root in **Dataset & Training**, or set it before starting Streamlit.

```powershell
$env:COTTONLEAF_DATASET_ROOT = 'C:\Data\Cotton Leaves'
.\run_app.bat
```

With no configured path, the app checks `datasets/`, `dataset/`, and
`data/` and their immediate children. Inspect valid image counts and
unreadable files, then save the split metadata. Each class needs at least
three eligible images for the dashboard's three-way split. Identical
decoded images within a class share one representative. Recompressed or
cropped copies can still need manual review.

**Upload & Classification** previews JPEG and PNG uploads without a
checkpoint. For inference, use a completed dashboard run or pair a compatible
local `checkpoints/classifier.keras` with `results/inference.json`.
The manifest records class output order, model input size, and pixel scale.
See [the artifact contract](results/README.md) and the
[presentation guide](DEMO_GUIDE.md).

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

## Training cards and research preview

Open **Dataset & Training** to see **Fast Demo** and **Full Hybrid** side by
side. Fast Demo defaults to two epochs. Full Hybrid defaults to a 50-epoch
budget and saves the weights restored from the best validation-loss epoch.
Both require verified local images and save their own observed metrics and
checkpoints. The scan counts valid and distinct eligible images separately.
The app preserves source files and derives eligible counts from the scan.

Click **Preview 50-Epoch Research Target** to replay the **Simulated prototype
training preview** in about 40 seconds. The illustrative validation curve
ends at 99.41%, the published reference. The replay is available without data
or TensorFlow and writes no artifacts. Real training has separate buttons.
When both checkpoints exist, choose one in **Upload & Classification**.

In **Model Comparisons → Our Experiment**, use **Comparison** for the
architecture and loss comparisons. Pending comparisons show architecture
descriptions or documented published references. Run the protocols in
[the training guide](training/README.md) to populate matched local charts.

# CottonLeaf AI final test report

Tested with Python 3.11 on Linux using Streamlit 1.51.0, Chromium with Playwright,
and TensorFlow 2.20.0 in the optional ML environment. The dashboard checkout
contained only the published benchmark JSON and artifact documentation.

## Passed checks

| Check | Result |
| --- | --- |
| Dependency installation and `pip check` | Core and optional ML dependency sets passed. |
| Compilation, imports, and Ruff | Application, training, model, and utility modules passed. Streamlit rendered the page modules through their normal entry points. |
| Pytest with TensorFlow | 20 passed, including five model builders, metrics, split leakage checks, and checkpoint serialization. |
| Dependency-light test run | 17 passed, with three TensorFlow-specific tests skipped as expected. |
| Headless Streamlit and Chromium | The health endpoint responded. All six pages opened through sidebar links and direct routes. Nested tabs, diagrams, and the narrow-layout sidebar worked. Content was checked at 1440, 1024, and 390 px. |
| Upload and checkpoint behavior | Chromium previewed PNG and JPEG files and handled a damaged file and absent checkpoint. Streamlit AppTest loaded a disposable trained Keras checkpoint and rendered the prediction, then disabled inference for a damaged checkpoint. |
| Benchmark provenance and downloads | Tests checked all 13 supplied benchmark metrics and the five fold references. Chromium verified the chart and table downloads as PNG and CSV. Published stages and local results remain labeled separately. |
| Training and persistence | A one-epoch CPU Fast Demo used 12 disposable image files. It saved a checkpoint, inference manifest, training history, predictions, evaluation metrics, confusion counts, ROC points, and complexity data in a temporary directory. No fixture outputs entered the dashboard's research results. |

[Chromium screenshot of Overview at 1440 × 960](docs/CottonLeaf-Overview.png)

## Failures found and resolved

- Incomplete saved-run metadata could raise an exception when opening the
  training or evaluation page. Both pages now render the fields that exist.
- A CSV download triggered a Streamlit rerun that could reset the active tab.
  Downloads now keep the current view.
- Ruff found an unused import, which was removed. Pytest filters one specific
  upstream warning from Keras's NumPy conversion. Other warnings remain visible.

No application failures remained in the passing checks. Streamlit probes
deep-link health and host configuration under the page path before retrying them
at the app root. The browser test verifies the successful retry and still rejects
other local HTTP errors. Streamlit's test harness can emit its bare-mode context
notice when running scripted tests outside the live server.

## Dataset and checkpoint availability

- No research dataset root was configured or found by auto-discovery. Actual
  research image counts, label quality, and split balance could not be verified.
  Dataset scanning and counts were tested with temporary inputs.
- No project or user-provided trained checkpoint was available. Model loading
  and prediction were verified with the Fast Demo test checkpoint only.

## Untested features and limits

Native Windows and PyCharm execution, CUDA training, ImageNet weight download,
and full research training and evaluation on real primary and secondary
datasets were outside the tested environment. Grad-CAM remains unavailable.
The disposable test run provides no research performance estimate.

## Exact commands

Run from the repository root. The first environment contains core packages,
and `/tmp/cottonleaf-ml-venv` is the local Python 3.11 environment with TensorFlow.

```bash
.venv/bin/python -m pip --disable-pip-version-check --no-cache-dir install -r requirements.txt
.venv/bin/python -m pip --disable-pip-version-check --no-cache-dir check
/tmp/cottonleaf-ml-venv/bin/python -m pip --disable-pip-version-check --no-cache-dir install -r requirements-ml.txt 'pytest==8.4.2' 'ruff==0.14.14'
/tmp/cottonleaf-ml-venv/bin/python -m pip --disable-pip-version-check --no-cache-dir check
.venv/bin/python -m compileall -q app.py config.py pages models training utils tests
TF_CPP_MIN_LOG_LEVEL=2 CUDA_VISIBLE_DEVICES=-1 /tmp/cottonleaf-ml-venv/bin/python -c 'import app, config, models.cnn, models.hybrid, models.inception, models.inference, models.keras_io, models.losses, training.run, training.experiments, utils.dashboard, utils.dataset, utils.evaluation, utils.images, utils.research_results'
/tmp/cottonleaf-ml-venv/bin/ruff check app.py config.py pages models training utils tests
.venv/bin/python -m unittest discover -s tests -v
TF_CPP_MIN_LOG_LEVEL=2 TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1 CUDA_VISIBLE_DEVICES=-1 /tmp/cottonleaf-ml-venv/bin/python -m pytest -q
.venv/bin/python -m streamlit run app.py --server.headless true --server.address 127.0.0.1 --server.port 8519
```

With Streamlit running and Playwright available to Node:

```bash
curl -fsS http://127.0.0.1:8519/_stcore/health
.venv/bin/python - <<'PY'
from PIL import Image
image = Image.new("RGB", (40, 30), (70, 120, 90))
image.save("/tmp/cottonleaf-qa.png")
image.save("/tmp/cottonleaf-qa.jpg")
PY
CHROMIUM_PATH=/usr/bin/chromium COTTONLEAF_TEST_PNG=/tmp/cottonleaf-qa.png COTTONLEAF_TEST_JPG=/tmp/cottonleaf-qa.jpg COTTONLEAF_SCREENSHOT=docs/CottonLeaf-Overview.png node tests/browser_smoke.cjs
```

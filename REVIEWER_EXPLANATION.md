# CottonLeaf AI reviewer explanation

## Five-minute presentation script

Use the timed sections as speaking notes. Keep the architecture and benchmark
pages open so the review can continue without a dataset or checkpoint.

**0:00–0:40 · The problem and objective**

CottonLeaf AI is a research dashboard for classifying a cotton leaf image into
one of four categories: bacterial blight, curl virus, fusarium wilt, or healthy.
Several conditions can produce similar visible symptoms, and lighting and
background can change how those symptoms look. The project asks whether channel
attention and multi-scale features can help a model distinguish the four
classes under a controlled evaluation protocol.

**0:40–1:25 · The data and experimental setup**

The dataset is organized into one folder per class. The app counts readable
images and checks for identical image content before splitting. It keeps one
copy when the labels agree and excludes duplicate groups with conflicting
labels. The holdout split targets 70% training, 15% validation, and 15% test,
with stratification and a fixed seed of 42. Images are oriented, converted to
RGB, and resized to 224 by 224. This checkout has no research dataset, so the
actual class counts and total are unavailable here.

**1:25–2:40 · How the model works**

The baseline CNN uses convolutions to detect patterns, ReLU for nonlinearity,
and pooling to reduce the feature maps. Global average pooling turns the
final maps into a short feature vector. The CN2-SE branch adds
squeeze-and-excitation. It averages each channel, passes those averages
through two dense layers, and uses the resulting weights to scale the
original channels. The other branch uses a shorter Inception-style stem and
parallel operations to capture patterns at different scales. The hybrid model
concatenates those two feature vectors. An MLP combines them, and a four-way
softmax gives the class probabilities. Cross-entropy penalizes low probability
on the true label. Focal loss reduces the influence of easy examples. Our
hybrid loss combines the two with configurable weights.

**2:40–3:30 · How we evaluate**

For five-fold cross-validation, each fold takes a turn as the outer test set.
A fresh model uses a separate validation subset from the remaining data.
The app can report accuracy, precision, recall, F1, specificity, and ROC-AUC
from saved labels and predictions. The confusion matrix shows which classes are
mistaken for each other. Matched ablations compare architectures or losses
using the same split and training budget.

**3:30–4:20 · Results and what we contributed**

Madhu and Ravisankar report **98.83%** for the proposed feature extractor and
**99.41%** for final classification with the hybrid MLP. Those are two stages
of the published study. This project adds an interactive workflow for
inspecting the architecture, checking data and splits, and presenting local
results separately. The compact modified Inception branch is our
research-inspired implementation. We verified the software with temporary
inputs and a one-epoch smoke test. A locally reproduced research accuracy is
unavailable.

**4:20–5:00 · Show the app and close**

On Upload & Classification, I can preview a JPEG or PNG. With a compatible
checkpoint, the app applies the same image preparation as training and
displays the predicted class and all four probabilities. This checkout has
no trained research checkpoint, so I’ll show the preview and the disabled
classification button. The next research step is to run the documented
protocol on a verified dataset and report the held-out results separately
from the paper’s benchmarks.

## Suggested click-by-click demo

Start Streamlit before the review. Keep a permitted JPEG or PNG ready for the
upload preview. The timestamps below match the speaking notes.

| Time | Clicks | What to show |
| --- | --- | --- |
| 0:00 | Sidebar → **Overview** | Point out the four classes and the published final classification card. |
| 0:40 | Sidebar → **Dataset & Training** | Show the dataset path field and training controls. If you brought a dataset, enter its root and click **Scan dataset**. Show the counts and split summary. |
| 1:25 | Sidebar → **Model Architecture** → **Hybrid feature fusion** → **Squeeze-and-excitation** → **Modified stem** | Trace both branches, explain the two dense layers in the SE block, then show how the stem reduces spatial resolution. |
| 2:40 | Sidebar → **Evaluation Metrics** → **Five-fold cross-validation** | Identify the published fold references and the separate local evaluation panel. Show **Confusion matrix** or **ROC / AUC** if you have a completed local run. |
| 3:30 | Sidebar → **Model Comparisons** → **Published Research** → **Our Experiment** | Point to the 98.83% feature-extraction entry and the 99.41% final-classification entry. Show the source labels in the local tab. |
| 4:20 | Sidebar → **Upload & Classification** → upload your image | Show the preview. Click **Classify image** only when a compatible checkpoint is available. Otherwise, show the unavailable message and disabled button. |

## Methodology cheat sheet

### Dataset, preprocessing, and splits

The scanner expects one local root with four labeled folders. The locations
can be entered in the dashboard or configured via `COTTONLEAF_DATASET_ROOT`.
Names such as `Bacterial Blight`, `Leaf Curl Virus`, and `fussarium_wilt`
are accepted as aliases.

| Folder / class | Actual research images in this checkout |
| --- | --- |
| `bacterial_blight` | Unavailable |
| `curl_virus` | Unavailable |
| `fusarium_wilt` | Unavailable |
| `healthy` | Unavailable |
| **Total** | **Unavailable. No research dataset was found.** |

The scanner verifies files and fingerprints decoded, orientation-corrected
RGB pixels. Exact copies with the same label share one representative for
splitting. Groups with conflicting labels are excluded. Cropped or
recompressed near-duplicates may require manual review. Use the dashboard's
valid-image count for scanned files and its split summary for eligible images
after exclusions.

Training and classification use EXIF orientation, RGB conversion, and
bilinear resizing. New training runs use 224 × 224 × 3 input. The baseline
and modified branches scale pixel values to [0, 1] inside the model. Standard
InceptionV3 uses its expected [−1, 1] range. A manual checkpoint's manifest
can specify another supported input size or external scaling contract.

In holdout evaluation, training updates the weights, validation loss is used
for early stopping and learning-rate reduction, and the test split is held out
for final scoring. Split proportions may round differently for small classes.
The training dataset is shuffled. Validation and test preprocessing is
deterministic, and there is no random image augmentation.

In five-fold evaluation, shuffled `StratifiedKFold` with seed 42 preserves
class proportions across outer folds. Each eligible image appears in exactly
one outer test set. Each fold uses a fresh model and reserves roughly 15% of
its development data per class for validation. The report records per-fold
metrics and pools out-of-fold predictions for an aggregate report. The
five-fold protocol needs at least five eligible images per class.

### Architecture and loss

| Component | What to say |
| --- | --- |
| Baseline CNN | Three convolution stages use 16, 32, and 64 filters with ReLU. Max pooling follows the first two stages. Global average pooling, a 64-unit dense layer, dropout, and softmax form the classification head. |
| Squeeze-and-Excitation | Global average pooling makes one descriptor per channel. A dense ReLU layer compresses the vector, and a dense sigmoid layer produces one gate per channel. Each gate multiplies its original feature map. |
| Standard InceptionV3 | Uses Keras Applications with a four-class head. It initializes offline by default. ImageNet weights require an explicit request or a compatible local weight file. |
| Modified Inception and stem | The stem uses 24, 32, and 48 channels, with an early stride, max pooling, and convolution–normalization–ReLU blocks. Two compact Inception-style blocks combine pointwise, 3×3, stacked 3×3, and pooled paths. The compact design is inspired by the paper. |
| Hybrid feature fusion | Both branches receive the same image and independently produce globally pooled feature vectors. Concatenation joins the vectors before classification. |
| MLP classifier | Dense layers with 128 and 64 units, ReLU, and dropout learn how to combine features. The final four-unit softmax produces one probability per class. |
| Cross-entropy | Uses the negative log of the probability assigned to the true class. |
| Focal loss | Downweights already confident correct examples using a factor based on `(1 − p_true)^gamma`. The default gamma is 2 and alpha is 0.25. |
| Hybrid loss | Weighted cross-entropy plus weighted focal loss. Both coefficients default to 0.5. It is used during training and does not change the inference workflow. |

Fast Demo trains a compact CNN. Full hybrid mode allows up to 50 epochs with
early stopping. Both use Adam with an initial learning rate of 0.0001, batch
size 32, and dropout 0.5. Those are configuration values. The project records
actual epochs, fit time, and saved artifacts for completed runs.

### Metrics and interpretation

For each class, one-vs-rest counts treat that class as positive and all other
classes as negative. Macro metrics give each class equal weight.

| Metric | Meaning |
| --- | --- |
| Accuracy | Fraction of all images assigned the correct class. |
| Precision | Of images predicted as this class, the fraction that truly belong to it. |
| Recall / sensitivity | Of images that truly belong to this class, the fraction that the model detects. |
| F1-score | Harmonic mean of precision and recall. Both must be strong for F1 to be high. |
| Specificity | Of images outside this class, the fraction correctly rejected for that class. |
| ROC-AUC | ROC traces true-positive versus false-positive rates as the probability threshold changes. AUC is the area under that curve. Macro AUC averages the four one-vs-rest AUCs when all are defined. |

The confusion matrix uses actual labels in rows and model predictions in
columns. ROC uses predicted probabilities. Saved fit time, parameter count,
and checkpoint size are measurements. Big-O discussions are analytical
estimates. Device peak memory and kernel timing require a profiler.

### Upload, published results, and contribution

Upload & Classification decodes a JPEG or PNG for preview. It uses a
configured manual checkpoint or finds the newest completed
dashboard run with the required files and reads its inference manifest. The manifest determines
class order, input size, and scaling. The app checks the model shape,
preprocesses the image, runs the model, and validates the four
probabilities. The highest probability determines the displayed class.
Missing, unreadable, or shape-incompatible checkpoints keep classification disabled.

| Published reference | Reported accuracy |
| --- | ---: |
| Proposed CN2-SE + Modified InceptionV3 feature extraction | 98.83% |
| Final classification with hybrid features and MLP | 99.41% |
| Published five-fold accuracies, in order | 99.3%, 99.5%, 99.4%, 99.4%, 99.5% |
| Local research reproduction | Unavailable |

Source is Sailaja Madhu and V. Ravisankar, *Robust cotton leaf disease
classification using hybrid deep learning approach with multi-loss function*.
All supplied reference values are stored in
[the benchmark JSON](research/published_benchmarks.json).

The contribution is an engineering implementation around the research idea.
It includes a compact two-branch model, reusable model and loss builders,
controlled comparisons for five architectures and two losses, exact-duplicate
checks, training and evaluation artifacts, and an interactive presentation
that labels results by source. These are software and experimental workflow
contributions. A new accuracy gain requires matched experiments on real data.

## Important Python files

| File | Role |
| --- | --- |
| [`app.py`](app.py) and [`config.py`](config.py) | Define Streamlit navigation, shared paths, the app title, and upload settings. |
| [`pages/overview.py`](pages/overview.py) | Explains the objective and displays classes, saved split counts when present, and the published final-classification reference. |
| [`pages/dataset.py`](pages/dataset.py) | Scans data, previews splits, configures and starts training, and displays recorded epochs. |
| [`pages/inspect_image.py`](pages/inspect_image.py) | Handles uploads, checkpoint availability, inference, and probability charts. |
| [`pages/comparisons.py`](pages/comparisons.py) | Renders published and local comparisons in separate tabs and supports chart and table export. |
| [`pages/evaluation.py`](pages/evaluation.py) | Loads completed evaluation outputs for metrics, confusion, ROC, folds, and measured complexity. |
| [`pages/architecture.py`](pages/architecture.py) | Draws the hybrid, SE, and stem flows and explains the components. |
| [`models/cnn.py`](models/cnn.py) | Builds the baseline CNN, the SE block, and the CNN+SE feature branch. |
| [`models/inception.py`](models/inception.py) | Builds standard InceptionV3 and the compact modified stem and multi-scale feature branch. |
| [`models/hybrid.py`](models/hybrid.py) | Concatenates branch features and builds the MLP and softmax classifier. |
| [`models/losses.py`](models/losses.py) | Selects categorical cross-entropy, categorical focal loss, or their weighted combination. |
| [`models/keras_io.py`](models/keras_io.py) | Loads a saved Keras checkpoint for inference without compiling it. |
| [`models/inference.py`](models/inference.py) | Finds a checkpoint, validates its manifest and input shape, and runs a single-image prediction. |
| [`training/run.py`](training/run.py) | Batches verified images, fits a model, scores the held-out test set, and saves checkpoints and measurements. |
| [`training/experiments.py`](training/experiments.py) | Runs five-fold CV, architecture ablation, matched loss comparison, and optional second-dataset evaluation. |
| [`utils/dataset.py`](utils/dataset.py) | Discovers class folders, checks files and duplicates, and builds or saves split metadata. |
| [`utils/images.py`](utils/images.py) | Prepares model pixels and image previews. |
| [`utils/evaluation.py`](utils/evaluation.py) | Computes label- and probability-based metrics and writes the evaluation artifacts. |
| [`utils/research_results.py`](utils/research_results.py) | Loads published benchmarks and normalizes local metric units while retaining provenance. |
| [`utils/dashboard.py`](utils/dashboard.py) | Holds shared styling, file readers, and chart rendering helpers. |

## Launch commands

On Windows, from the repository root in PowerShell:

```powershell
.\run_app.bat
```

For manual setup or to use the same interpreter in PyCharm:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py --server.headless true
```

On Linux or macOS with Python 3.11:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m streamlit run app.py --server.headless true
```

For training or checkpoint inference, use the same virtual environment's
Python to run `python -m pip install -r requirements-ml.txt`. Open the URL
printed by Streamlit. For PyCharm, use module `streamlit`, parameters
`run app.py --server.headless true`, and the repository as working directory.
See the [full setup instructions](README.md).

## Five likely reviewer questions

1. **Did you achieve 99.41% accuracy?** The paper reports 99.41% for final
   classification with its hybrid MLP. Local testing verified the software
   on temporary files. A measured local research accuracy is unavailable.
2. **Why use both SE and Inception-style processing?** SE scores which
   channels deserve more emphasis. The multi-scale branch processes leaf
   patterns at several spatial scales. An ablation can test whether combining
   them helps on the same data.
3. **How do you prevent data leakage?** Identical decoded images share one
   eligible representative, and conflicting duplicate groups are excluded.
   We check for repeated digests across partitions. In cross-validation,
   the outer test fold stays separate from model fitting and validation.
4. **Why report more than accuracy?** Accuracy can hide failures in a small
   class. Macro F1 gives each class equal weight, per-class recall shows
   missed detections, and specificity shows how often other classes are
   correctly rejected.
5. **What is new in this project?** The project adds a compact implementation
   and an auditable workflow to demonstrate feature fusion, compare losses and
   architectures, and display the source of each result. Whether it improves
   on the paper is an open experimental question.

## Verification and unfinished work

The browser walkthrough and automated checks are documented in the
[test report](TEST_REPORT.md). The 12 temporary images and one-epoch CPU run
were software test fixtures. They verified artifact saving and checkpoint
loading. Actual research image counts, a trained research checkpoint, local
research metrics, full training, GPU execution, ImageNet weight download, and
native Windows execution remain unverified. Cross-dataset evaluation is
implemented and needs a second compatible research dataset to produce
meaningful measurements. Grad-CAM remains unavailable.

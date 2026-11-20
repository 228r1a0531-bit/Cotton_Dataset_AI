# Training

Install `requirements-ml.txt` and open **Dataset & Training** in the Streamlit
dashboard. Fast Demo fits the compact baseline CNN for a small configurable
number of epochs. Full hybrid training fits the CN2-SE and modified Inception
fusion model for up to 50 epochs. Use suitable GPU resources and a verified
dataset for a full hybrid experiment.

The data pipeline loads the saved split's images as EXIF-oriented RGB, resizes
them to 224 × 224 using bilinear sampling, and passes float32 pixels to the
model. The model applies its own pixel scaling. Only training is shuffled.
Labels use the saved four-class order and are encoded as one-hot vectors.
Choose categorical cross-entropy, focal loss, or weighted CE + focal loss.

Adam starts at 0.0001. The learning rate can drop on a validation loss plateau.
Early stopping restores the weights from the best validation epoch. The
trainer evaluates the held-out test split once and saves the resulting
checkpoint, configuration, class mapping, and history. Inference automatically
finds the newest completed run unless a manual checkpoint is configured.

See the repository README for local dataset setup and [the artifact contract](../results/README.md)
for the run outputs.

## Reproduce evaluation and ablations

Commands assume a local four-class dataset and an environment installed from
`requirements-ml.txt`. The dashboard's **Evaluation Metrics** page lets you
select completed result bundles. **Model Comparisons → Our Experiment**
also reads completed comparison and fold artifacts.

```bash
python -m training.experiments cv "path/to/dataset" --model baseline --epochs 5
python -m training.experiments ablation "path/to/dataset" --epochs 5
python -m training.experiments stem "path/to/dataset" --epochs 5
python -m training.experiments loss "path/to/dataset" --epochs 5
python -m training.experiments external "path/to/second-dataset" --run "results/training_runs/<run-id>"
```

- **CV** uses shuffled `StratifiedKFold(n_splits=5, random_state=42)`. Each
  development partition reserves roughly 15% per class for validation using
  seed `42 + fold`. Each eligible image is scored in exactly one outer test
  fold. The aggregate report pools out-of-fold predictions, and `folds.csv`
  records metrics for each fold. Five distinct eligible images per class
  are required.
- **Ablation** trains CNN, CNN+SE, standard InceptionV3 from scratch, Modified
  InceptionV3, and the fused model with categorical cross-entropy on the
  same holdout split, seed, and epoch budget.
- **STEM ablation** keeps the same modified Inception feature blocks and head.
  It compares the learned STEM against a fixed average-pooling downsample,
  using the same holdout split, loss, seed, and epoch budget.
- **Loss comparison** fixes the fused architecture, holdout split, seed,
  and epoch budget. It compares categorical cross-entropy with hybrid
  CE + focal loss (coefficients 0.5 and 0.5, gamma 2, alpha 0.25).
- **External** loads one completed checkpoint with its stored class order and
  preprocessing contract. It removes exact decoded-image matches to any
  source partition and requires at least one remaining image in each class.
  No fitting or threshold selection is done on the second dataset.

All partitions use decoded RGB digests to exclude exact duplicates and
conflicting labels. Resized, cropped, or recompressed copies may need separate
review. Fixed seeds make splits reproducible. Floating-point training can
still vary across devices or kernels. A five-epoch run is a short protocol
check. Choose a research epoch budget and appropriate hardware when comparing
results for presentation.

Fit seconds are wall-clock time inside Keras `fit`, including validation and
callbacks. Parameter counts come from `model.count_params()`. Decimal MB is
the checkpoint file size and may include optimizer state. Tensor element
counts and convolution operation estimates describe analytical scaling.
Peak device memory and kernel timing require separate profiler measurements.

## Audit or refine a completed Full Hybrid run

Use the original dataset root and the existing run directory. Keep
`classifier.keras`, `training.json`, `inference.json`, and `splits.json`
together. Run from the repository root with the TensorFlow environment:

```bash
python -m training.refine "path/to/dataset" "results/training_runs/<original-run>" --audit-only
python -m training.refine "path/to/dataset" "results/training_runs/<original-run>" --epochs 15
```

The audit checks the saved class order and RGB input contract, required hybrid
layers, and all original path, label, and decoded pixel hashes. It checks for
exact duplicates across splits and reports class support and per-class metrics.
Near duplicates from cropping or recompression require separate review.

The original Full Hybrid combines CN2-SE and a compact, research-inspired
Inception branch with a feature-fusion MLP. Standard Keras InceptionV3 can
accept ImageNet weights when explicitly requested. The smaller hybrid branch
has different layers and widths, so standard weights cannot initialize it
directly. This refinement starts from the original local checkpoint as saved.

The candidate keeps the original split and loss, lowers Adam's initial
learning rate to 0.00003, reduces the two MLP dropout rates from 0.5 to 0.25,
and retains validation-loss early stopping and learning-rate reduction.
It applies small rotations and flips only during training and uses
normalized inverse-square-root class weights derived from training labels.
Evaluation and inference keep the original deterministic preprocessing.

Historical training accuracy is measured with dropout enabled as weights
change across batches. Validation runs with dropout disabled using the
weights at the end of an epoch. Those differences can contribute to
validation accuracy being higher. The audit also scores the original model's
train and validation sets with dropout off and the same fixed checkpoint to
help distinguish that effect from differences in split difficulty.

New reports go to `results/evaluations/`. In **Evaluation Metrics**, select
the audit or refinement bundle for class support, confusion, and per-class
results. In **Model Comparisons → Our Experiment**, the refinement bundle
shows the original and candidate measured on the identical held-out test set.
Use validation to choose a candidate. Avoid repeated tuning against the
reported test figures.

The original checkpoint and logs stay in their run directory. **Upload &
Classification** prioritizes an explicit manual checkpoint override. Without
one, it prefers the latest saved Full Hybrid over Fast Demo. A completed
refinement of that hybrid takes priority only if its measured accuracy on
the same validation split improved. Both checkpoints remain selectable.

## Local comparisons for the published model families

Run these with the verified dataset to populate **Model Comparisons → Our Experiment**:

```bash
python -m training.experiments heads "path/to/dataset" --epochs 5
python -m training.experiments features "path/to/dataset" --epochs 5
```

`heads` fits ANN, LSTM, and MLP on the same split with the same hybrid loss.
The ANN uses one hidden dense layer, the LSTM processes two branch embeddings
as a short sequence, and the MLP retains the existing two-layer head.
`features` compares CNN, standard InceptionV3, fixed regional color/Sobel
descriptors, and hybrid fusion with a common split and cross-entropy.
These are local, research-inspired designs. Measured accuracy depends on the
dataset and training budget. The 99.41% published benchmark remains separate.

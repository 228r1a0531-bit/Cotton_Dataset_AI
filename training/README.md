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

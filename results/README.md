# Research artifacts

The dashboard reads local, Git-ignored artifacts from this directory. Add only
recorded measurements and cite any externally published benchmark.

| File | Required CSV columns or JSON fields |
| --- | --- |
| `research.json` | `classes` (four labels), `benchmark` (object with `accuracy` and `citation` strings) |
| `dataset.csv` | `class,split,count`, one row per class and split |
| `training.json` | `configuration` (object), `status` (string) |
| `training.log` | Plain text. The dashboard displays the tail. |
| `history.csv` | `epoch,split,metric,value` |
| `comparisons.csv` | `comparison,variant,metric,value`. Use a comparison name displayed on the page. |
| `metrics.csv` | `metric,value` |
| `complexity.csv` | `model,training_seconds,parameters` |
| `confusion.csv` | `actual,predicted,count` |
| `roc.csv` | `class,fpr,tpr,auc` |
| `folds.csv` | `fold,metric,value` |
| `inference.json` | `classes` (four labels in checkpoint output order), `image_size` ([width, height]), `pixel_scale` (`0_255`, `0_1`, or `minus1_1`) |

For inference, install the optional machine learning dependencies and place a
trusted checkpoint at `checkpoints/classifier.keras` in the repository root.
The model must accept a single batch of RGB float32 images and return four
softmax probabilities. Preprocessing applies EXIF orientation, converts to RGB,
resizes directly with bilinear sampling, and uses the declared pixel scale.
Only connect checkpoints trained with that preprocessing.

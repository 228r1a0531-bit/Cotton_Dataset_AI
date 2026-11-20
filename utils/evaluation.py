"""Calculate and export metrics from real four-class labels and probabilities.

Metric functions are independent of TensorFlow. Prediction batches use the
same Pillow preprocessing as training and pass pixels through the saved model.
All persisted rates use fractions and class order is explicit.
"""

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, auc, confusion_matrix, precision_recall_fscore_support, roc_curve

from utils.dataset import CLASSES, Sample, atomic_write
from utils.images import model_pixels


@dataclass
class Evaluation:
    """Computed metrics and the ordered predictions used to derive them."""

    summary: dict[str, float]
    per_class: list[tuple]
    confusion: np.ndarray
    roc: list[tuple]
    predicted: np.ndarray


def measure(labels, probabilities, classes=CLASSES) -> Evaluation:
    """Compute one-vs-rest metrics in the checkpoint's class order.

    Undefined precision, recall, F1, or specificity uses zero. ROC is omitted
    for classes without positives or negatives. Macro AUC requires all classes.
    """
    truth = np.asarray(labels)
    scores = np.asarray(probabilities, dtype=float)
    count = len(classes)
    if (
        truth.ndim != 1 or not len(truth) or not np.issubdtype(truth.dtype, np.integer)
        or (truth < 0).any() or (truth >= count).any()
    ):
        raise ValueError("Labels must be nonempty integer class indices.")
    if (scores.shape != (len(truth), count) or not np.isfinite(scores).all()
            or (scores < 0).any() or (scores > 1).any()
            or not np.allclose(scores.sum(axis=1), 1, atol=1e-4)):
        raise ValueError("Expected one normalized probability vector per ground-truth label.")

    predicted = scores.argmax(axis=1)
    indices = np.arange(count)
    matrix = confusion_matrix(truth, predicted, labels=indices)
    precision, recall, f1, support = precision_recall_fscore_support(
        truth, predicted, labels=indices, zero_division=0
    )
    negatives = len(truth) - support
    false_positives = matrix.sum(axis=0) - matrix.diagonal()
    specificity = np.divide(negatives - false_positives, negatives,
                            out=np.zeros(count, dtype=float), where=negatives != 0)
    summary = {
        "accuracy": float(accuracy_score(truth, predicted)),
        "precision": float(precision.mean()),
        "recall": float(recall.mean()),
        "f1": float(f1.mean()),
        "sensitivity": float(recall.mean()),
        "specificity": float(specificity.mean()),
    }
    rows, curves, class_auc = [], [], []
    for index, name in enumerate(classes):
        rows.append((name, precision[index], recall[index], f1[index], recall[index],
                     specificity[index], support[index]))
        binary = truth == index
        if binary.any() and not binary.all():
            fpr, tpr, _ = roc_curve(binary, scores[:, index])
            area = auc(fpr, tpr)
            class_auc.append(area)
            curves.extend((name, x, y, area) for x, y in zip(fpr, tpr))
    if len(class_auc) == count:
        summary["macro_auc"] = float(np.mean(class_auc))
    return Evaluation(summary, rows, matrix, curves, predicted)


def write_csv(path: Path, fields, rows) -> None:
    """Replace one CSV atomically once its header and rows have been written."""
    def write(handle):
        writer = csv.writer(handle)
        writer.writerow(fields)
        writer.writerows(rows)
    atomic_write(path, write)


def score_samples(model, root: Path, samples: list[Sample], size, batch_size, classes=CLASSES, scale="0_255"):
    """Score verified images in their original order using bounded batches."""
    labels = np.array([classes.index(sample.label) for sample in samples], dtype=int)
    probabilities = []
    for start in range(0, len(samples), batch_size):
        batch = np.stack([model_pixels(root / sample.path, size, scale)
                          for sample in samples[start:start + batch_size]])
        probabilities.append(np.asarray(model(batch, training=False)))
    return labels, np.concatenate(probabilities) if probabilities else np.empty((0, len(classes)))


def save_evaluation(directory: Path, samples: list[Sample], labels, probabilities, classes=CLASSES) -> Evaluation:
    """Persist audit rows, overall and class metrics, confusion counts, and ROC points."""
    result = measure(labels, probabilities, classes)
    write_csv(directory / "metrics.csv", ("metric", "value"), result.summary.items())
    write_csv(directory / "class_metrics.csv",
              ("class", "precision", "recall", "f1", "sensitivity", "specificity", "support"), result.per_class)
    write_csv(directory / "confusion.csv", ("actual", "predicted", "count"),
              ((actual, predicted, result.confusion[i, j])
               for i, actual in enumerate(classes) for j, predicted in enumerate(classes)))
    write_csv(directory / "roc.csv", ("class", "fpr", "tpr", "auc"), result.roc)
    write_csv(directory / "predictions.csv", ("path", "digest", "actual", "predicted", *classes),
              ((sample.path, sample.digest, classes[true], classes[predicted], *scores)
               for sample, true, predicted, scores in zip(samples, labels, result.predicted, probabilities)))
    return result

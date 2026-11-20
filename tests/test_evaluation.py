"""Validate measured metric arithmetic and leak-free fold assignments in isolation."""

import unittest
from pathlib import Path

import numpy as np

from training.experiments import five_fold_splits
from utils.dataset import CLASSES, Sample, Scan, check_split_leakage
from utils.evaluation import measure


class EvaluationTests(unittest.TestCase):
    def test_metrics_match_hand_calculated_fixture(self):
        truth = np.repeat(np.arange(4), 2)
        predicted = [0, 1, 1, 1, 2, 3, 3, 3]
        probabilities = np.full((8, 4), 0.05)
        probabilities[np.arange(8), predicted] = 0.85
        result = measure(truth, probabilities)

        self.assertEqual(result.summary["accuracy"], 0.75)
        self.assertAlmostEqual(result.summary["precision"], 5 / 6)
        self.assertEqual(result.summary["recall"], 0.75)
        self.assertAlmostEqual(result.summary["f1"], (2 / 3 + 0.8) / 2)
        self.assertEqual(result.summary["sensitivity"], 0.75)
        self.assertAlmostEqual(result.summary["specificity"], 11 / 12)
        self.assertAlmostEqual(result.summary["macro_auc"], 5 / 6)
        np.testing.assert_array_equal(result.confusion, [[1, 1, 0, 0], [0, 2, 0, 0],
                                                        [0, 0, 1, 1], [0, 0, 0, 2]])
        self.assertEqual(result.per_class[0][0], CLASSES[0])
        np.testing.assert_allclose(result.per_class[0][1:], [1, 0.5, 2 / 3, 0.5, 1, 2])
        self.assertEqual({row[0] for row in result.roc}, set(CLASSES))

    def test_auc_omits_unsupported_classes_and_rejects_invalid_scores(self):
        self.assertNotIn("macro_auc", measure([0], [[1, 0, 0, 0]]).summary)
        with self.assertRaises(ValueError):
            measure([0], [[0.25, 0.25, 0.25, 0.50]])

    def test_cross_validation_is_reproducible_and_rejects_leakage(self):
        samples = [Sample(f"{label}/{index}.png", label, f"{label}-{index}")
                   for label in CLASSES for index in range(6)]
        samples += [Sample("healthy/copy.png", "healthy", "healthy-5"),
                    Sample("healthy/conflict.png", "healthy", f"{CLASSES[0]}-5")]
        dataset = Scan(Path("."), samples, {}, set(CLASSES))
        folds = five_fold_splits(dataset)
        self.assertEqual(folds, five_fold_splits(dataset))
        self.assertEqual(len(folds), 5)
        tested = []
        for split in folds:
            check_split_leakage(split)
            for partition in split.values():
                self.assertEqual({sample.label for sample in partition}, set(CLASSES))
            tested.extend(sample.digest for sample in split["test"])
        self.assertEqual(len(tested), 23)
        self.assertEqual(len(set(tested)), 23)
        self.assertNotIn(f"{CLASSES[0]}-5", tested)
        with self.assertRaises(ValueError):
            check_split_leakage({"train": [samples[-3]], "test": [samples[-2]]})

    def test_cross_validation_requires_five_eligible_samples_per_class(self):
        dataset = Scan(Path("."), [Sample(f"{label}/0.png", label, label) for label in CLASSES], {}, set(CLASSES))
        with self.assertRaisesRegex(ValueError, "at least five eligible"):
            five_fold_splits(dataset)

    def test_dashboard_recovers_curves_and_metrics_from_saved_predictions(self):
        from tempfile import TemporaryDirectory
        from unittest.mock import patch

        from streamlit.testing.v1 import AppTest

        from config import ROOT

        with TemporaryDirectory() as directory:
            results = Path(directory)
            rows = [",".join(("actual", *CLASSES))]
            for index, name in enumerate(CLASSES):
                rows.append(",".join((name, *("1" if index == col else "0" for col in range(4)))))
            (results / "predictions.csv").write_text("\n".join(rows), encoding="utf-8")
            with patch("config.RESULTS", results), patch("utils.dashboard.RESULTS", results):
                app = AppTest.from_file(str(ROOT / "app.py")).run().switch_page("pages/evaluation.py").run()
        self.assertFalse(app.exception)
        self.assertEqual(len(app.get("plotly_chart")), 3)
        metrics = {metric.label: metric.value for metric in app.metric}
        self.assertEqual(metrics["Published final classification reference"], "99.41%")
        self.assertEqual(metrics["Actual model accuracy"], "100.00%")
        self.assertEqual(metrics["Macro one-vs-rest AUC"], "1.0000")

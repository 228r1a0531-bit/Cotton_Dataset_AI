"""Check benchmark transcription, stage-specific charts, and source boundaries."""

import json
import unittest
from base64 import b64decode
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest

from config import ROOT
from utils.research_results import LOCAL, PUBLISHED, local_percentages, published_benchmarks, published_metrics


class ComparisonTests(unittest.TestCase):
    def test_transcription_matches_all_supplied_metrics(self):
        data = published_metrics(published_benchmarks())
        actual = {
            (row.Context, row.Model, row.Metric): row.Percent
            for row in data.itertuples()
        }
        self.assertEqual(actual, {
            ("Feature extraction", "CNN", "Accuracy"): 96.49,
            ("Feature extraction", "InceptionV3", "Accuracy"): 93.85,
            ("Feature extraction", "Handcrafted features", "Accuracy"): 91.81,
            ("Feature extraction", "CN2-SE + Modified InceptionV3", "Accuracy"): 98.83,
            ("Feature extraction", "CN2-SE + Modified InceptionV3", "Sensitivity"): 99.42,
            ("Feature extraction", "CN2-SE + Modified InceptionV3", "Specificity"): 98.22,
            ("Feature extraction", "CN2-SE + Modified InceptionV3", "Recall"): 99.42,
            ("Final classification", "Hybrid + ANN", "Accuracy"): 94.73,
            ("Final classification", "Hybrid + LSTM", "Accuracy"): 96.78,
            ("Final classification", "Hybrid + MLP", "Accuracy"): 99.41,
            ("Final classification", "Hybrid + MLP", "Sensitivity"): 99.42,
            ("Final classification", "Hybrid + MLP", "Specificity"): 99.40,
            ("Final classification", "Hybrid + MLP", "Recall"): 99.42,
        })
        self.assertEqual(set(data["Source"]), {PUBLISHED})

    def test_visualizations_show_separate_accuracy_stages_and_png_config(self):
        with TemporaryDirectory() as directory:
            result_dir = Path(directory)
            with patch("config.RESULTS", result_dir), patch("utils.dashboard.RESULTS", result_dir):
                app = AppTest.from_file(str(ROOT / "app.py")).run().switch_page("pages/comparisons.py").run()
        self.assertFalse(app.exception)
        self.assertEqual([tab.label for tab in app.tabs], ["Published Research", "Our Experiment"])
        self.assertTrue(any(PUBLISHED in entry.value for entry in app.info))
        feature, final, before_after, quality, folds = app.get("plotly_chart")
        for element, expected in (
            (feature, [96.49, 93.85, 91.81, 98.83]),
            (final, [94.73, 96.78, 99.41]),
            (before_after, [96.49, 98.83]),
            (folds, [99.3, 99.5, 99.4, 99.4, 99.5]),
        ):
            spec = json.loads(element.proto.spec)
            actual = []
            for trace in spec["data"]:
                values = trace["y"]
                actual.extend(np.frombuffer(b64decode(values["bdata"]), dtype=values["dtype"])
                              if isinstance(values, dict) else values)
            np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-10)
            self.assertEqual(json.loads(element.proto.config)["toImageButtonOptions"]["format"], "png")
        self.assertEqual(len(json.loads(quality.proto.spec)["data"]), 2)
        self.assertEqual(len(app.get("download_button")), 3)

    def test_local_metrics_require_valid_units_and_preserve_provenance(self):
        records = pd.DataFrame({
            "metric": ["accuracy", "F1-score", "recall", "precision", "specificity"],
            "value": [0.75, 81.5, 105, "bad", 1],
            "unit": ["fraction", "percent", "percent", "fraction", "unknown"],
        })
        data = local_percentages(records, "Test fixture")
        self.assertEqual(data["Metric"].tolist(), ["Accuracy", "F1-score"])
        self.assertEqual(data["Percent"].tolist(), [75.0, 81.5])
        self.assertEqual(set(data["Source"]), {LOCAL})

    def test_available_fold_values_and_local_metrics_render_in_their_own_tabs(self):
        with TemporaryDirectory() as directory:
            results = Path(directory)
            (results / "metrics.csv").write_text(
                "metric,value\nprecision,0.6\nrecall,0.7\nf1,0.65\nspecificity,0.8\n", encoding="utf-8"
            )
            experiment = results / "evaluations" / "cv-fixture"
            experiment.mkdir(parents=True)
            (experiment / "experiment.json").write_text('{"status": "Completed", "protocol": "cv"}', encoding="utf-8")
            (experiment / "folds.csv").write_text("fold,metric,value\n1,accuracy,0.75\n", encoding="utf-8")
            run = results / "training_runs" / "fixture"
            run.mkdir(parents=True)
            (run / "training.json").write_text(json.dumps({
                "status": "Completed", "configuration": {"Model": "Test fixture"}, "test": {"accuracy": 0.7},
            }), encoding="utf-8")
            with (
                patch("config.RESULTS", results),
                patch("utils.dashboard.RESULTS", results),
            ):
                app = AppTest.from_file(str(ROOT / "app.py")).run().switch_page("pages/comparisons.py").run()
        self.assertFalse(app.exception)
        self.assertEqual(len(app.tabs[0].get("plotly_chart")), 5)
        self.assertEqual(len(app.tabs[1].get("plotly_chart")), 3)
        combined = app.tabs[1].dataframe[0].value
        self.assertEqual(set(combined["Source"]), {PUBLISHED, LOCAL})
        self.assertEqual(len(combined.loc[combined["Source"] == LOCAL]), 6)

"""Verify replay timing, exact endpoint, and separation from training artifacts."""

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

from config import ROOT
from utils.research_results import published_benchmarks
from utils.training_preview import PREVIEW_LABEL, PREVIEW_SECONDS, RESEARCH_TARGET, preview_epoch, preview_history


class TrainingPreviewTests(TestCase):
    def test_fifty_epoch_curve_is_deterministic(self):
        history = preview_history()
        self.assertEqual(history, preview_history())
        self.assertEqual(len(history), 200)
        validation = [row["value"] for row in history if row["split"] == "validation" and row["metric"] == "accuracy"]
        self.assertEqual(len(validation), 50)
        self.assertEqual(validation[-1], 0.9941)
        self.assertEqual(RESEARCH_TARGET, published_benchmarks()["final_classification"][-1]["accuracy"] / 100)
        self.assertTrue(all(left < right for left, right in zip(validation, validation[1:])))
        self.assertEqual(preview_epoch(0), 1)
        self.assertLess(preview_epoch(PREVIEW_SECONDS - 0.1), 50)
        self.assertEqual(preview_epoch(PREVIEW_SECONDS), 50)

    def test_preview_does_not_train_or_write_results(self):
        with TemporaryDirectory() as directory:
            results = Path(directory)
            with (
                patch("config.RESULTS", results),
                patch("utils.dashboard.RESULTS", results),
                patch("training.run.train") as train,
                patch("utils.training_preview.preview_epoch", return_value=50),
            ):
                app = AppTest.from_file(str(ROOT / "app.py")).run().switch_page("pages/dataset.py").run()
                next(button for button in app.button if button.label.startswith("Preview 50-Epoch")).click().run()
                self.assertFalse(app.exception)
                self.assertTrue(any(value.value == PREVIEW_LABEL for value in app.info))
                accuracy = next(metric for metric in app.metric if metric.label == "Simulated validation accuracy")
                self.assertEqual(accuracy.value, "99.41%")
                train.assert_not_called()
                self.assertEqual(list(results.iterdir()), [])

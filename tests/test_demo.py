"""Smoke-test the dashboard with no local dataset or model."""

import unittest
from html import unescape
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from PIL import Image
from streamlit.testing.v1 import AppTest

from models.inference import model_settings, predict
from models.keras_io import load_classifier
from utils.images import load_preview, rgb_histogram

ROOT = Path(__file__).resolve().parents[1]


def sample_image():
    source = BytesIO()
    Image.new("RGB", (3, 2), (10, 20, 30)).save(source, format="PNG")
    source.seek(0)
    return source


class DemoTests(unittest.TestCase):
    def test_all_navigation_pages_render(self):
        app = AppTest.from_file(str(ROOT / "app.py")).run()
        self.assertFalse(app.exception)
        self.assertIn("CottonLeaf AI", app.markdown[1].value)

        pages = [
            ("dataset", "Dataset & Training"),
            ("inspect_image", "Upload & Classification"),
            ("comparisons", "Model Comparisons"),
            ("evaluation", "Evaluation Metrics"),
            ("architecture", "Model Architecture"),
        ]
        for slug, title in pages:
            with self.subTest(page=slug):
                app.switch_page(f"pages/{slug}.py").run()
                self.assertFalse(app.exception)
                self.assertTrue(any(f"<h1>{title}</h1>" in unescape(item.value) for item in app.markdown))

    def test_image_page_previews_upload_without_checkpoint(self):
        app = AppTest.from_file(str(ROOT / "app.py")).run()
        with patch("streamlit.file_uploader", return_value=sample_image()):
            app.switch_page("pages/inspect_image.py").run()

        self.assertFalse(app.exception)
        self.assertEqual(len(app.get("imgs")), 1)
        self.assertFalse(app.get("plotly_chart"))
        self.assertTrue(any("Inference unavailable" in item.value for item in app.info))

    def test_recorded_artifacts_render_charts(self):
        tables = {
            "dataset.csv": "class,split,count\nfixture,train,8\nfixture,test,2\n",
            "history.csv": "epoch,split,metric,value\n1,train,loss,0.5\n",
            "comparisons.csv": "comparison,variant,metric,value\nCNN vs CNN+SE,CNN,accuracy,0.5\n",
            "confusion.csv": "actual,predicted,count\nfixture,fixture,1\n",
            "roc.csv": "class,fpr,tpr,auc\nfixture,0,0,0.5\nfixture,1,1,0.5\n",
            "folds.csv": "fold,metric,value\n1,accuracy,0.5\n",
        }
        with TemporaryDirectory() as directory:
            for filename, contents in tables.items():
                (Path(directory) / filename).write_text(contents, encoding="utf-8")
            with patch("utils.dashboard.RESULTS", Path(directory)):
                app = AppTest.from_file(str(ROOT / "app.py")).run()
                for slug, chart_count in [("dataset", 1), ("comparisons", 6), ("evaluation", 3)]:
                    with self.subTest(page=slug):
                        app.switch_page(f"pages/{slug}.py").run()
                        self.assertFalse(app.exception)
                        self.assertEqual(len(app.get("plotly_chart")), chart_count)

    def test_histogram_counts_decoded_pixels(self):
        histogram = rgb_histogram(load_preview(sample_image()))

        self.assertEqual(histogram.sum().to_dict(), {"Red": 6, "Green": 6, "Blue": 6})
        self.assertEqual(histogram.loc[10, "Red"], 6)
        self.assertEqual(histogram.loc[20, "Green"], 6)
        self.assertEqual(histogram.loc[30, "Blue"], 6)

    def test_missing_checkpoint_is_reported_without_tensorflow(self):
        with self.assertRaises(FileNotFoundError):
            load_classifier(ROOT / "missing-model.keras")

    def test_inference_contract(self):
        import numpy as np

        class TestClassifier:
            def predict(self, pixels, verbose):
                np.testing.assert_equal(pixels.shape, (1, 2, 3, 3))
                np.testing.assert_allclose(pixels[0, 0, 0], np.array([10, 20, 30]) / 255)
                return [[0.1, 0.6, 0.2, 0.1]]

        labels, size, scale = model_settings({
            "classes": ["a", "b", "c", "d"], "image_size": [3, 2], "pixel_scale": "0_1"
        })
        scores = predict(TestClassifier(), sample_image(), size, scale)
        self.assertEqual(labels[scores.argmax()], "b")

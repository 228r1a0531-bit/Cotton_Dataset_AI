"""Exercise image analysis and the Streamlit navigation without TensorFlow."""

import unittest
from io import BytesIO
from pathlib import Path
from unittest.mock import patch

from PIL import Image
from streamlit.testing.v1 import AppTest

from models.keras_io import load_classifier
from utils.images import load_preview, rgb_histogram

ROOT = Path(__file__).resolve().parents[1]


class DemoTests(unittest.TestCase):
    def test_overview_and_image_page(self):
        app = AppTest.from_file(str(ROOT / "app.py")).run()
        self.assertFalse(app.exception)
        self.assertEqual(app.title[0].value, "CottonLeaf AI")

        app.switch_page("pages/inspect_image.py").run()
        self.assertFalse(app.exception)
        self.assertEqual(app.title[0].value, "Inspect a leaf image")
        self.assertEqual(len(app.get("file_uploader")), 1)

    def test_histogram_counts_decoded_pixels(self):
        source = BytesIO()
        Image.new("RGB", (3, 2), (10, 20, 30)).save(source, format="PNG")
        source.seek(0)

        histogram = rgb_histogram(load_preview(source))

        self.assertEqual(histogram.sum().to_dict(), {"Red": 6, "Green": 6, "Blue": 6})
        self.assertEqual(histogram.loc[10, "Red"], 6)
        self.assertEqual(histogram.loc[20, "Green"], 6)
        self.assertEqual(histogram.loc[30, "Blue"], 6)

    def test_image_page_renders_uploaded_image_and_chart(self):
        source = BytesIO()
        Image.new("RGB", (3, 2), (10, 20, 30)).save(source, format="PNG")
        source.seek(0)

        app = AppTest.from_file(str(ROOT / "app.py")).run()
        with patch("streamlit.file_uploader", return_value=source):
            app.switch_page("pages/inspect_image.py").run()

        self.assertFalse(app.exception)
        self.assertEqual(len(app.get("imgs")), 1)
        self.assertEqual(len(app.get("plotly_chart")), 1)

    def test_missing_checkpoint_is_reported_without_tensorflow(self):
        with self.assertRaises(FileNotFoundError):
            load_classifier(ROOT / "missing-model.keras")

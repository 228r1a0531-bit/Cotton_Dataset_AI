"""Exercise dashboard training and checkpoint inference with disposable assets."""

import importlib.util
import json
import os
import unittest
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from PIL import Image
from streamlit.testing.v1 import AppTest

from config import ROOT
from utils.dataset import CLASSES


class EndToEndTests(unittest.TestCase):
    def test_incomplete_artifacts_do_not_crash(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            results = root / "results"
            run = results / "training_runs" / "old-run"
            run.mkdir(parents=True)
            (run / "training.json").write_text('{"status": "Completed", "test": {}}', encoding="utf-8")
            (run / "history.csv").write_text("epoch,split,metric,value\n1,train,9,0.5\n", encoding="utf-8")
            trial = results / "evaluations" / "old-ablation" / "training_runs" / "trial"
            trial.mkdir(parents=True)
            (trial / "training.json").write_text('{"status": "Completed"}', encoding="utf-8")
            (trial.parents[1] / "experiment.json").write_text(
                '{"status": "Completed", "protocol": "ablation"}', encoding="utf-8"
            )
            with patch("config.RESULTS", results), patch("utils.dashboard.RESULTS", results):
                app = AppTest.from_file(str(ROOT / "app.py")).run()
                for page in ("dataset", "evaluation"):
                    app.switch_page(f"pages/{page}.py").run()
                    self.assertFalse(app.exception)

    @unittest.skipUnless(importlib.util.find_spec("tensorflow"), "Optional TensorFlow dependencies are absent")
    def test_fast_demo_saves_actual_artifacts_and_classifies(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            data, results = root / "images", root / "results"
            for label, name in enumerate(CLASSES):
                folder = data / name
                folder.mkdir(parents=True)
                for index in range(3):
                    Image.new("RGB", (12, 12), (label * 60, index * 60, 90)).save(folder / f"{index}.png")
            png = (data / CLASSES[0] / "0.png").read_bytes()
            with (
                patch.dict(os.environ, {"COTTONLEAF_DATASET_ROOT": str(data)}),
                patch("config.ROOT", root),
                patch("config.RESULTS", results),
                patch("utils.dashboard.RESULTS", results),
            ):
                app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60).run()
                app.switch_page("pages/dataset.py").run()
                self.assertEqual(app.metric[0].value, "12")
                self.assertTrue(any("Not trained yet" in value.value for value in app.markdown))
                app.number_input[0].set_value(1)
                next(button for button in app.button if button.label == "Start training").click().run()
                self.assertFalse(app.error or app.exception)

                run, = (results / "training_runs").iterdir()
                state = json.loads((run / "training.json").read_text(encoding="utf-8"))
                self.assertEqual(state["status"], "Completed")
                self.assertEqual(state["completed"], 1)
                for filename in ("classifier.keras", "inference.json", "history.csv", "metrics.csv",
                                 "predictions.csv", "confusion.csv", "roc.csv", "complexity.csv"):
                    self.assertGreater((run / filename).stat().st_size, 0)

                with patch("streamlit.file_uploader", side_effect=lambda *args, **kwargs: BytesIO(png)):
                    app.switch_page("pages/inspect_image.py").run()
                    self.assertFalse(app.button[0].disabled)
                    app.button[0].click().run()
                    self.assertFalse(app.error or app.exception)
                    self.assertIn(app.metric[0].value, CLASSES)
                    self.assertEqual(len(app.get("plotly_chart")), 1)

                    manual = root / "checkpoints" / "classifier.keras"
                    manual.parent.mkdir()
                    manual.write_bytes(b"Invalid Keras archive")
                    (results / "inference.json").write_bytes((run / "inference.json").read_bytes())
                    app.run()
                    self.assertFalse(app.exception)
                    self.assertTrue(app.button[0].disabled)
                    self.assertTrue(any("Could not use this checkpoint" in warning.value for warning in app.error))

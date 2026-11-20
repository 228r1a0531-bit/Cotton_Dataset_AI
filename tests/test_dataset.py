"""Validate dataset integrity, split reproducibility, and dashboard counts."""

import json
import os
import unittest
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from PIL import Image
from streamlit.testing.v1 import AppTest

from config import ROOT
from utils.dataset import CLASSES, SPLITS, discover, save_splits, scan, stratify

FOLDERS = ("Bacterial Blight", "Leaf CURL Virus", "Fussarium_Wilt", "HEALTHY")


def make_dataset(root: Path) -> None:
    for label, folder in enumerate(FOLDERS):
        directory = root / folder
        directory.mkdir(parents=True)
        for index in range(20):
            Image.new("RGB", (8, 8), (label * 60, index * 9, 30)).save(directory / f"{index}.PNG")


class DatasetTests(unittest.TestCase):
    def test_scan_and_split_duplicate_groups(self):
        with TemporaryDirectory() as directory:
            root = Path(directory) / "Cotton Leaves"
            make_dataset(root)
            source = root / FOLDERS[0] / "0.PNG"
            with Image.open(source) as image:
                image.save(root / FOLDERS[0] / "renamed.bmp")
                image.save(root / FOLDERS[1] / "mislabeled.bmp")
            with Image.open(root / FOLDERS[2] / "0.PNG") as image:
                image.save(root / FOLDERS[2] / "same-label.bmp")
            (root / FOLDERS[3] / "broken.jpg").write_bytes(b"bad file")

            with patch.dict(os.environ, {"COTTONLEAF_DATASET_ROOT": str(root)}):
                self.assertEqual(discover(), root)
            dataset = scan(root)
            self.assertEqual(len(dataset.images), 83)
            self.assertEqual(dataset.matched_classes, set(CLASSES))
            self.assertEqual(set(dataset.unreadable), {"HEALTHY/broken.jpg"})
            self.assertEqual(sorted(map(len, dataset.duplicates)), [2, 3])
            splits = stratify(dataset)
            self.assertEqual(splits, stratify(scan(root)))
            self.assertEqual([len(splits[name]) for name in SPLITS], [55, 12, 12])
            excluded_digest = next(group[0].digest for group in dataset.duplicates if len(group) == 3)
            digests = [sample.digest for images in splits.values() for sample in images]
            self.assertEqual(len(digests), len(set(digests)))
            self.assertNotIn(excluded_digest, digests)
            for name, samples in splits.items():
                counts = Counter(sample.label for sample in samples)
                self.assertEqual(set(counts), set(CLASSES))
                if name != "train":
                    self.assertEqual(set(counts.values()), {3})

            results = Path(directory) / "results"
            save_splits(results, dataset, splits)
            metadata = json.loads((results / "splits.json").read_text())
            self.assertEqual(metadata["seed"], 42)
            self.assertEqual([len(metadata["splits"][name]) for name in SPLITS], [55, 12, 12])
            self.assertEqual(len((results / "dataset.csv").read_text().splitlines()), 13)

    def test_dataset_page_renders_scan_and_empty_training_status(self):
        with TemporaryDirectory() as directory:
            root = Path(directory) / "Cotton Leaves"
            make_dataset(root)
            with patch.dict(os.environ, {"COTTONLEAF_DATASET_ROOT": str(root)}):
                app = AppTest.from_file(str(ROOT / "app.py")).run()
                app.switch_page("pages/dataset.py").run()

            self.assertFalse(app.exception)
            self.assertEqual(app.metric[0].value, "80")
            self.assertEqual(len(app.get("plotly_chart")), 2)
            self.assertTrue(any("Not trained yet" in item.value for item in app.markdown))

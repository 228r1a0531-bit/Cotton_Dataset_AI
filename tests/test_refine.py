"""Exercise the audit, same-split comparison, and read-only checkpoint contract."""

import hashlib
import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase, skipUnless

from PIL import Image

from models.inference import available_checkpoints
from training.refine import load_verified_split, run
from training.run import train
from utils.dataset import CLASSES, scan, stratify


class RefinementTests(TestCase):
    def test_saved_split_rejects_changed_pixels(self):
        from utils.dataset import save_splits

        with TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = scan(make_images(root))
            splits = stratify(dataset)
            source = root / "source"
            save_splits(source, dataset, splits)
            verified, _ = load_verified_split(dataset, source)
            self.assertEqual(verified, splits)
            changed = dataset.root / splits["test"][0].path
            Image.new("RGB", (12, 12), (7, 8, 9)).save(changed)
            with self.assertRaisesRegex(ValueError, "missing or differs"):
                load_verified_split(scan(dataset.root), source)

    @skipUnless(importlib.util.find_spec("tensorflow"), "Optional TensorFlow dependencies are absent")
    def test_audit_refines_without_changing_source(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = scan(make_images(root))
            results = root / "results"
            source = train(dataset, stratify(dataset), 1, results, lambda *_: None,
                           architecture="hybrid", loss_name="hybrid")
            original = {name: hashlib.sha256((source / name).read_bytes()).hexdigest()
                        for name in ("classifier.keras", "history.csv", "training.json", "splits.json")}
            audit = run(dataset, source, results, 1, audit_only=True)
            self.assertTrue((audit / "class_metrics.csv").is_file())
            self.assertTrue((audit / "confusion.csv").is_file())
            refined = run(dataset, source, results, 1)
            state = json.loads((refined / "experiment.json").read_text())
            self.assertEqual(state["status"], "Completed")
            self.assertTrue((refined / "comparisons.csv").is_file())
            candidate = refined / state["candidate_run"]
            self.assertEqual(original, {name: hashlib.sha256((source / name).read_bytes()).hexdigest()
                                        for name in original})
            self.assertEqual(json.loads((source / "splits.json").read_text())["splits"],
                             json.loads((candidate / "splits.json").read_text())["splits"])
            self.assertEqual(available_checkpoints(root, results)[0][0].parent,
                             candidate if state["prefer_candidate"] else source)


def make_images(root):
    data = root / "images"
    for label, name in enumerate(CLASSES):
        folder = data / name
        folder.mkdir(parents=True)
        for number in range(3):
            Image.new("RGB", (12, 12), (label * 60, number * 60, 90)).save(folder / f"{number}.png")
    return data

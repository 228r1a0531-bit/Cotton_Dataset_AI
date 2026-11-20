"""Exercise model builders and loss serialization on synthetic tensors only."""

import importlib.util
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory


@unittest.skipUnless(importlib.util.find_spec("tensorflow"), "Optional TensorFlow dependencies are absent")
class ModelTests(unittest.TestCase):
    def test_all_architectures_return_four_probabilities(self):
        import numpy as np
        from tensorflow import keras

        from models.cnn import build_cnn
        from models.hybrid import build_hybrid
        from models.inception import build_inception_v3, build_modified_inception

        builders = {
            "CNN": build_cnn,
            "CN2-SE": lambda: build_cnn(use_se=True),
            "standard InceptionV3 (offline)": build_inception_v3,
            "modified InceptionV3": build_modified_inception,
            "hybrid": build_hybrid,
        }
        image = np.full((1, 224, 224, 3), 127, dtype=np.float32)
        for name, build in builders.items():
            with self.subTest(architecture=name):
                keras.backend.clear_session()
                model = build()
                scores = model(image, training=False).numpy()
                self.assertEqual(scores.shape, (1, 4))
                self.assertTrue(np.isfinite(scores).all())
                self.assertTrue(((scores >= 0) & (scores <= 1)).all())
                np.testing.assert_allclose(scores.sum(axis=1), [1], atol=1e-6)

    def test_hybrid_loss_weights_and_checkpoint_round_trip(self):
        import numpy as np
        from tensorflow import keras

        from models.hybrid import build_hybrid
        from models.keras_io import load_classifier
        from models.losses import build_loss

        targets = keras.ops.convert_to_tensor([[1.0, 0.0, 0.0, 0.0]])
        estimates = keras.ops.convert_to_tensor([[0.7, 0.1, 0.1, 0.1]])
        ce = build_loss("cross_entropy")(targets, estimates)
        focal = build_loss("focal", gamma=1.5)(targets, estimates)
        loss = build_loss("hybrid", ce_weight=0.3, focal_weight=0.7, gamma=1.5)
        np.testing.assert_allclose(loss(targets, estimates), 0.3 * ce + 0.7 * focal)
        restored_loss = keras.losses.deserialize(keras.losses.serialize(loss))
        np.testing.assert_allclose(restored_loss(targets, estimates), loss(targets, estimates))

        model = build_hybrid((32, 32, 3))
        model.compile(optimizer="adam", loss=loss)
        image = np.full((1, 32, 32, 3), 127, dtype=np.float32)
        with TemporaryDirectory() as directory:
            path = Path(directory) / "hybrid.keras"
            model.save(path)
            restored_model = load_classifier(path)
            np.testing.assert_allclose(restored_model(image, training=False), model(image, training=False))

"""Inference for local classifiers that declare their input and output contract."""

from pathlib import Path

import numpy as np
from PIL import Image, ImageOps


def model_settings(manifest: dict) -> tuple[list[str], tuple[int, int], str]:
    labels = manifest.get("classes")
    size = manifest.get("image_size")
    scale = manifest.get("pixel_scale")
    if (
        not isinstance(labels, list)
        or len(labels) != 4
        or not all(isinstance(label, str) and label.strip() for label in labels)
        or len(set(labels)) != 4
    ):
        raise ValueError("Specify four unique class labels in checkpoint output order.")
    if (
        not isinstance(size, list)
        or len(size) != 2
        or not all(type(dimension) is int and 1 <= dimension <= 2048 for dimension in size)
    ):
        raise ValueError("Specify image_size as [width, height] with dimensions from 1 to 2048.")
    if scale not in ("0_255", "0_1", "minus1_1"):
        raise ValueError("Specify a supported pixel_scale.")
    return labels, (size[0], size[1]), scale


def predict(model, source, size: tuple[int, int], scale: str) -> np.ndarray:
    source.seek(0)
    with Image.open(source) as image:
        pixels = np.asarray(
            ImageOps.exif_transpose(image).convert("RGB").resize(size, Image.Resampling.BILINEAR),
            dtype=np.float32,
        )
    if scale == "0_1":
        pixels /= 255
    elif scale == "minus1_1":
        pixels = pixels / 127.5 - 1

    scores = np.asarray(model.predict(pixels[None, ...], verbose=0))
    if (
        scores.shape != (1, 4)
        or not np.isfinite(scores).all()
        or (scores < 0).any()
        or (scores > 1).any()
        or not np.isclose(scores.sum(), 1, atol=1e-4)
    ):
        raise ValueError("The checkpoint must output four softmax probabilities.")
    return scores[0]


CHECKPOINT_NAME = Path("checkpoints") / "classifier.keras"

"""Decode uploaded images and summarize their color channels."""

from typing import BinaryIO

import numpy as np
import pandas as pd
from PIL import Image, ImageOps

from config import MAX_PREVIEW_SIZE


def load_preview(source: BinaryIO) -> Image.Image:
    """Decode an image, apply its EXIF orientation, and bound the preview size."""
    with Image.open(source) as image:
        # JPEG decoders can reduce the image before allocating full RGB pixels.
        image.draft("RGB", MAX_PREVIEW_SIZE)
        preview = ImageOps.exif_transpose(image).convert("RGB")
        preview.thumbnail(MAX_PREVIEW_SIZE)
    return preview


def rgb_histogram(image: Image.Image) -> pd.DataFrame:
    """Count each intensity in the preview's red, green, and blue channels."""
    pixels = np.asarray(image).reshape(-1, 3)
    counts = {
        channel: np.bincount(pixels[:, index], minlength=256)
        for index, channel in enumerate(("Red", "Green", "Blue"))
    }
    return pd.DataFrame(counts).rename_axis("Intensity")

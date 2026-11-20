"""Shared settings for the CottonLeaf AI demo."""

from pathlib import Path

APP_TITLE = "CottonLeaf AI"
IMAGE_TYPES = ("jpg", "jpeg", "png")
MAX_PREVIEW_SIZE = (1024, 1024)


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"

"""Inference for local classifiers that declare their input and output contract."""

import json
from pathlib import Path

import numpy as np
from utils.images import model_pixels


def model_settings(manifest: dict) -> tuple[list[str], tuple[int, int], str]:
    """Validate class order, resize dimensions, and any external pixel scaling."""
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
    """Run a single image through an actual model and validate its probabilities."""
    source.seek(0)
    pixels = model_pixels(source, size, scale)

    scores = np.asarray(model(pixels[None, ...], training=False))
    if (
        scores.shape != (1, 4)
        or not np.isfinite(scores).all()
        or (scores < 0).any()
        or (scores > 1).any()
        or not np.isclose(scores.sum(), 1, atol=1e-4)
    ):
        raise ValueError("The checkpoint must output four softmax probabilities.")
    return scores[0]


def validate_model_shape(model, size: tuple[int, int]) -> None:
    """Check that the checkpoint accepts one RGB input and emits four scores."""
    expected = (size[1], size[0], 3)
    shape = getattr(model, "input_shape", None)
    output = getattr(model, "output_shape", None)
    if not isinstance(shape, tuple) or len(shape) != 4 or shape[0] not in (1, None) or any(
        actual is not None and actual != target for actual, target in zip(shape[1:], expected)
    ):
        raise ValueError("Checkpoint input does not match the configured RGB image size.")
    if not isinstance(output, tuple) or len(output) != 2 or output[-1] != 4:
        raise ValueError("Checkpoint output does not match the four configured classes.")


CHECKPOINT_NAME = Path("checkpoints") / "classifier.keras"


def available_checkpoints(root: Path, results: Path) -> list[tuple[Path, Path]]:
    """Prefer a manual override, then the latest hybrid and its validated refinements."""
    manual = root / CHECKPOINT_NAME, results / "inference.json"
    checkpoints = [manual] if all(path.is_file() for path in manual) else []
    hybrids, other = [], []
    for state in sorted((results / "training_runs").glob("*/training.json"), reverse=True):
        candidate = state.parent / "classifier.keras", state.parent / "inference.json"
        if not all(path.is_file() for path in candidate):
            continue
        try:
            metadata = json.loads(state.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if isinstance(metadata, dict) and metadata.get("status") == "Completed":
            configuration = metadata.get("configuration")
            is_hybrid = metadata.get("architecture") == "hybrid" or (
                isinstance(configuration, dict) and configuration.get("Model") == "Hybrid feature fusion"
            )
            (hybrids if is_hybrid else other).append(candidate)
    refinements = []
    preferred = []
    for path in sorted((results / "evaluations").glob("*/experiment.json"), reverse=True):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
            if record.get("status") != "Completed" or record.get("protocol") != "hybrid_refinement":
                continue
            directory = (path.parent / record["candidate_run"]).resolve()
            if not directory.is_relative_to(path.parent.resolve()):
                continue
            candidate = directory / "classifier.keras", directory / "inference.json"
            if not all(part.is_file() for part in candidate):
                continue
            if (hybrids and Path(record["source_run"]).resolve() == hybrids[0][0].parent.resolve()
                    and record.get("prefer_candidate") is True
                    and 0 <= record["source_validation_accuracy"] < record["candidate_validation_accuracy"] <= 1):
                preferred.append((record["candidate_validation_accuracy"], candidate))
            else:
                refinements.append(candidate)
        except (AttributeError, KeyError, OSError, TypeError, ValueError):
            continue
    return [*checkpoints, *(candidate for _, candidate in sorted(preferred, key=lambda item: -item[0])),
            *hybrids, *refinements, *other]


def find_checkpoint(root: Path, results: Path) -> tuple[Path, Path] | None:
    """Return the preferred saved checkpoint when one exists."""
    return next(iter(available_checkpoints(root, results)), None)

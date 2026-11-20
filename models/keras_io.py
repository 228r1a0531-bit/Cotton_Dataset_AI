"""Optional inference support for trained Keras checkpoints."""

from pathlib import Path


def load_classifier(checkpoint: str | Path):
    """Load an existing Keras model for inference without compiling it."""
    path = Path(checkpoint)
    if not path.is_file():
        raise FileNotFoundError(path)

    from tensorflow import keras

    return keras.models.load_model(path, compile=False)

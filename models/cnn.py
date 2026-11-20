"""Compact convolutional feature branches and matched four-class classifiers.

Both CNN variants consume raw RGB pixels and share the same convolutional
layout. The SE variant adds channel gates before pooling at each stage.
"""

from tensorflow import keras
from tensorflow.keras import layers


def se_block(features, reduction=8, name="se"):
    """Squeeze spatial dimensions, learn channel gates, and rescale the input."""
    channels = int(features.shape[-1])
    gates = layers.GlobalAveragePooling2D(name=f"{name}_squeeze")(features)
    gates = layers.Dense(max(1, channels // reduction), activation="relu", name=f"{name}_reduce")(gates)
    gates = layers.Dense(channels, activation="sigmoid", name=f"{name}_expand")(gates)
    gates = layers.Reshape((1, 1, channels), name=f"{name}_broadcast")(gates)
    return layers.Multiply(name=f"{name}_recalibrate")([features, gates])


def cnn_features(inputs, *, use_se=False):
    """Convert an image batch into a compact globally pooled feature vector."""
    prefix = "cn2_se" if use_se else "cnn"
    x = layers.Rescaling(1 / 255, name=f"{prefix}_scale")(inputs)
    for stage, filters in enumerate((16, 32, 64), start=1):
        x = layers.Conv2D(filters, 3, strides=2 if stage == 1 else 1, padding="same",
                          activation="relu", name=f"{prefix}_conv_{stage}")(x)
        if use_se:
            x = se_block(x, name=f"{prefix}_attention_{stage}")
        if stage < 3:
            x = layers.MaxPooling2D(name=f"{prefix}_pool_{stage}")(x)
    return layers.GlobalAveragePooling2D(name=f"{prefix}_features")(x)


def build_cnn(input_shape=(224, 224, 3), *, use_se=False, dropout=0.5):
    """Build the baseline CNN or the CN2-SE classifier with a matched head."""
    inputs = keras.Input(input_shape, name="image")
    features = cnn_features(inputs, use_se=use_se)
    x = layers.Dense(64, activation="relu")(features)
    x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(4, activation="softmax", name="class_probabilities")(x)
    return keras.Model(inputs, outputs, name="cn2_se" if use_se else "baseline_cnn")

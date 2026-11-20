"""Standard InceptionV3 and a compact research-inspired Inception variant.

Standard InceptionV3 uses the Keras Applications topology and its expected
pixel scaling. The modified branch is trained from scratch with a shorter stem,
fewer multi-scale blocks, and smaller widths. It is a conceptual variant.
"""

from pathlib import Path

from tensorflow import keras
from tensorflow.keras import layers


def build_inception_v3(input_shape=(224, 224, 3), *, weights=None, train_backbone=None, dropout=0.5):
    """Build a four-class model using random, explicit ImageNet, or local backbone weights.

    With no weights the backbone is trainable. With explicit weights it is
    frozen by default. Passing 'imagenet' lets Keras use its cache or download
    weights, and download or file errors propagate to the caller.
    """
    if weights is not None and weights != "imagenet":
        weights = str(Path(weights).expanduser().resolve(strict=True))
    backbone = keras.applications.InceptionV3(
        include_top=False, weights=weights, input_shape=input_shape, pooling="avg"
    )
    backbone.trainable = (weights is None) if train_backbone is None else train_backbone
    inputs = keras.Input(input_shape, name="image")
    scaled = layers.Rescaling(1 / 127.5, offset=-1)(inputs)
    features = backbone(scaled) if backbone.trainable else backbone(scaled, training=False)
    outputs = layers.Dense(4, activation="softmax", name="class_probabilities")(
        layers.Dropout(dropout)(features)
    )
    return keras.Model(inputs, outputs, name="standard_inception_v3")


def conv(features, filters, kernel, name, strides=1):
    """Apply a same-padded convolution, batch normalization, and ReLU."""
    x = layers.Conv2D(filters, kernel, strides=strides, padding="same",
                      use_bias=False, name=f"{name}_conv")(features)
    x = layers.BatchNormalization(name=f"{name}_bn")(x)
    return layers.ReLU(name=f"{name}_relu")(x)


def modified_stem(inputs):
    """Reduce spatial resolution with a strided convolution and max pooling."""
    x = layers.Rescaling(1 / 255, name="modified_scale")(inputs)
    x = conv(x, 24, 3, "stem_entry", strides=2)
    x = conv(x, 32, 3, "stem_refine")
    x = layers.MaxPooling2D(3, strides=2, padding="same", name="stem_pool")(x)
    return conv(x, 48, 3, "stem_output")


def inception_block(inputs, width, name):
    """Fuse pointwise, 3×3, stacked 3×3, and pooled spatial responses."""
    point = conv(inputs, width, 1, f"{name}_point")
    spatial = conv(inputs, width, 1, f"{name}_spatial_reduce")
    spatial = conv(spatial, width, 3, f"{name}_spatial")
    wide = conv(inputs, width, 1, f"{name}_wide_reduce")
    wide = conv(wide, width, 3, f"{name}_wide_first")
    wide = conv(wide, width, 3, f"{name}_wide_second")
    pooled = layers.AveragePooling2D(3, strides=1, padding="same", name=f"{name}_pool")(inputs)
    pooled = conv(pooled, width, 1, f"{name}_pool_project")
    return layers.Concatenate(name=f"{name}_concat")([point, spatial, wide, pooled])


def modified_features(inputs):
    """Extract a feature vector using a short stem and two multi-scale blocks."""
    x = modified_stem(inputs)
    x = inception_block(x, 24, "mixed_small")
    x = layers.MaxPooling2D(3, strides=2, padding="same", name="mixed_downsample")(x)
    x = inception_block(x, 32, "mixed_large")
    return layers.GlobalAveragePooling2D(name="modified_inception_features")(x)


def build_modified_inception(input_shape=(224, 224, 3), *, dropout=0.5):
    """Attach a four-class head to the research-inspired feature extractor."""
    inputs = keras.Input(input_shape, name="image")
    features = modified_features(inputs)
    outputs = layers.Dense(4, activation="softmax", name="class_probabilities")(
        layers.Dropout(dropout)(features)
    )
    return keras.Model(inputs, outputs, name="modified_inception_v3")

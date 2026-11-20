"""Fuse the CN2-SE and modified Inception branches for leaf classification."""

from tensorflow import keras
from tensorflow.keras import layers

from models.cnn import cnn_features
from models.inception import modified_features


def build_hybrid(input_shape=(224, 224, 3), *, dropout=0.5):
    """Feed one image into both feature branches, then classify their concatenation."""
    inputs = keras.Input(input_shape, name="image")
    attended = cnn_features(inputs, use_se=True)
    multiscale = modified_features(inputs)
    x = layers.Concatenate(name="feature_fusion")([attended, multiscale])
    x = layers.Dense(128, activation="relu", name="fusion_dense")(x)
    x = layers.Dropout(dropout, name="fusion_dropout")(x)
    x = layers.Dense(64, activation="relu", name="classifier_dense")(x)
    x = layers.Dropout(dropout, name="classifier_dropout")(x)
    outputs = layers.Dense(4, activation="softmax", name="class_probabilities")(x)
    return keras.Model(inputs, outputs, name="hybrid_feature_fusion")

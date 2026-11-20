"""Fuse the CN2-SE and modified Inception branches for leaf classification."""

from tensorflow import keras
from tensorflow.keras import layers

from models.cnn import cnn_features
from models.inception import modified_features


def build_hybrid(input_shape=(224, 224, 3), *, dropout=0.5, head="mlp"):
    """Compare classifier heads using the same two feature extractors."""
    inputs = keras.Input(input_shape, name="image")
    attended = cnn_features(inputs, use_se=True)
    multiscale = modified_features(inputs)
    if head == "lstm":
        # Treat branch embeddings as two feature tokens in a fixed order.
        tokens = [layers.Reshape((1, 64))(layers.Dense(64)(branch))
                  for branch in (attended, multiscale)]
        x = layers.Concatenate(axis=1, name="feature_sequence")(tokens)
        x = layers.Dropout(dropout)(layers.LSTM(64)(x))
    elif head in ("ann", "mlp"):
        x = layers.Concatenate(name="feature_fusion")([attended, multiscale])
        x = layers.Dense(128, activation="relu", name="fusion_dense")(x)
        x = layers.Dropout(dropout, name="fusion_dropout")(x)
        if head == "mlp":
            x = layers.Dense(64, activation="relu", name="classifier_dense")(x)
            x = layers.Dropout(dropout, name="classifier_dropout")(x)
    else:
        raise ValueError(f"Unknown hybrid head: {head}")
    outputs = layers.Dense(4, activation="softmax", name="class_probabilities")(x)
    name = "hybrid_feature_fusion" if head == "mlp" else f"hybrid_{head}"
    return keras.Model(inputs, outputs, name=name)

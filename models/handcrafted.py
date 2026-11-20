"""Fixed color and edge descriptors with a trained four-class classifier."""

from tensorflow import keras
from tensorflow.keras import layers


def build_handcrafted(input_shape=(224, 224, 3), *, dropout=0.5):
    """Summarize regional color and fixed Sobel responses before classification."""
    inputs = keras.Input(input_shape, name="image")
    rgb = layers.Rescaling(1 / 255)(layers.Resizing(64, 64)(inputs))
    color = layers.AveragePooling2D(16)(rgb)
    horizontal = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
    kernel = [[[[horizontal[i][j] / 3, horizontal[j][i] / 3] for _ in range(3)]
               for j in range(3)] for i in range(3)]
    edges = layers.Conv2D(
        2, 3, padding="same", use_bias=False, trainable=False,
        kernel_initializer=keras.initializers.Constant(kernel), name="fixed_sobel",
    )(rgb)
    magnitude = keras.ops.abs(edges)
    texture = layers.AveragePooling2D(16)(magnitude)
    features = layers.Concatenate()([layers.Flatten()(color), layers.Flatten()(texture)])
    outputs = layers.Dense(4, activation="softmax", name="class_probabilities")(layers.Dropout(dropout)(features))
    return keras.Model(inputs, outputs, name="handcrafted_color_edge")

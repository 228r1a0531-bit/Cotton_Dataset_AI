"""Losses for one-hot class targets and four-class softmax probabilities."""

from math import isfinite

from tensorflow import keras


@keras.utils.register_keras_serializable(package="CottonLeaf")
class HybridLoss(keras.losses.Loss):
    """Combine per-sample categorical cross-entropy and categorical focal loss."""

    def __init__(self, ce_weight=0.5, focal_weight=0.5, gamma=2.0, alpha=0.25,
                 name="hybrid_ce_focal", **kwargs):
        super().__init__(name=name, **kwargs)
        values = (ce_weight, focal_weight, gamma, alpha)
        if any(not isfinite(value) or value < 0 for value in values):
            raise ValueError("Loss parameters must be finite and nonnegative.")
        if ce_weight + focal_weight == 0 or not 0 < alpha <= 1:
            raise ValueError("Use at least one positive weight and an alpha in (0, 1].")
        self.ce_weight = ce_weight
        self.focal_weight = focal_weight
        self.gamma = gamma
        self.alpha = alpha
        self.focal = keras.losses.CategoricalFocalCrossentropy(alpha=alpha, gamma=gamma, reduction=None)

    def call(self, y_true, y_pred):
        ce = keras.losses.categorical_crossentropy(y_true, y_pred)
        return self.ce_weight * ce + self.focal_weight * self.focal(y_true, y_pred)

    def get_config(self):
        return {
            **super().get_config(),
            "ce_weight": self.ce_weight,
            "focal_weight": self.focal_weight,
            "gamma": self.gamma,
            "alpha": self.alpha,
        }


def build_loss(name, *, ce_weight=0.5, focal_weight=0.5, gamma=2.0, alpha=0.25):
    """Select a Keras loss using the same one-hot labels for every experiment."""
    if name == "cross_entropy":
        return keras.losses.CategoricalCrossentropy()
    if name == "focal":
        return keras.losses.CategoricalFocalCrossentropy(gamma=gamma, alpha=alpha)
    if name == "hybrid":
        return HybridLoss(ce_weight, focal_weight, gamma, alpha)
    raise ValueError(f"Unknown loss: {name}")

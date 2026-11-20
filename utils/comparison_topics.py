"""Architecture descriptions and matching labels for the comparison dashboard."""

TOPICS = {
    "CNN without SE vs CNN with SE": (
        ("Baseline CNN", "Convolutions → pooling → global average pooling → dense head."),
        ("CNN+SE", "The same CNN stages add learned channel gates before pooling."),
    ),
    "Standard InceptionV3 vs Modified InceptionV3": (
        ("InceptionV3", "Keras Applications stem and mixed blocks with a four-class head."),
        ("Modified InceptionV3", "Compact learned stem and two parallel multi-scale blocks."),
    ),
    "With STEM vs without STEM": (
        ("Modified InceptionV3", "Learned convolutional STEM feeds the multi-scale feature blocks."),
        ("Modified InceptionV3 without STEM", "Fixed average pooling replaces the learned STEM. The multi-scale blocks and head remain."),
    ),
    "Cross-Entropy vs Hybrid CE + Focal Loss": (
        ("Categorical cross-entropy", "Uses the same fused architecture and penalizes low probability on the true class."),
        ("Hybrid CE + focal loss", "Uses the same fused architecture with a focal term that emphasizes harder examples."),
    ),
    "Single branch vs hybrid feature fusion": (
        ("CNN+SE", "Channel-attention CNN feature vector with a dense head."),
        ("Modified InceptionV3", "Multi-scale feature vector with a four-class head."),
        ("Hybrid feature fusion", "Concatenates both feature vectors before a dropout-regularized MLP."),
    ),
    "ANN vs LSTM vs MLP": (
        ("Hybrid + ANN", "The paper's ANN classification head on the hybrid features."),
        ("Hybrid + LSTM", "The paper's recurrent LSTM head on the hybrid features."),
        ("Hybrid + MLP", "The paper's MLP head on the hybrid features. The local implementation has a research-inspired MLP."),
    ),
}

ALIASES = {
    "CNN": "Baseline CNN",
    "Standard InceptionV3": "InceptionV3",
}


def matched_results(scores, names):
    """Select a complete accuracy pairing from a single saved protocol."""
    rows = scores.loc[scores["Metric"] == "Accuracy"].copy()
    rows["Model"] = rows["Model"].replace(ALIASES)
    rows = rows.loc[rows["Model"].isin(names)]
    return rows if set(rows["Model"]) == set(names) and not rows["Model"].duplicated().any() else rows.iloc[:0]

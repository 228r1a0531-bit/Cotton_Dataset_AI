"""Read recorded model evaluation outputs."""

import plotly.express as px
import streamlit as st

from utils.dashboard import PALETTE, chart, empty, heading, table

heading("Evaluation Metrics", "Review measured classification performance and computational cost.")
summary, confusion, roc, folds = st.tabs(
    ["Summary", "Confusion matrix", "ROC / AUC", "Five-fold cross-validation"]
)
with summary:
    st.subheader("Accuracy, precision, recall, F1-score, and specificity")
    metrics = table("metrics.csv", ("metric", "value"))
    if metrics is None:
        empty("No held-out evaluation metrics have been recorded.")
    else:
        st.dataframe(metrics, hide_index=True)
    st.subheader("Training time and model complexity")
    complexity = table("complexity.csv", ("model", "training_seconds", "parameters"))
    if complexity is None:
        empty("Training duration and parameter counts are unavailable.")
    else:
        st.dataframe(complexity, hide_index=True)
with confusion:
    matrix = table("confusion.csv", ("actual", "predicted", "count"))
    if matrix is None:
        empty("A confusion matrix requires recorded true and predicted labels.")
    else:
        grid = matrix.pivot_table(index="actual", columns="predicted", values="count", aggfunc="sum")
        chart(px.imshow(grid, text_auto=True, color_continuous_scale="Greens", aspect="auto",
                        labels={"x": "Predicted", "y": "Actual", "color": "Count"}))
with roc:
    points = table("roc.csv", ("class", "fpr", "tpr", "auc"))
    if points is None:
        empty("ROC curves and AUC require saved evaluation scores.")
    else:
        chart(px.line(points.sort_values("fpr"), x="fpr", y="tpr", color="class",
                      color_discrete_sequence=PALETTE,
                      labels={"fpr": "False positive rate", "tpr": "True positive rate"}))
        st.dataframe(points[["class", "auc"]].drop_duplicates(), hide_index=True)
with folds:
    scores = table("folds.csv", ("fold", "metric", "value"))
    if scores is None:
        empty("No cross-validation fold scores are available.")
    else:
        chart(px.bar(scores, x="fold", y="value", color="metric", barmode="group",
                     color_discrete_sequence=PALETTE))
        st.dataframe(scores, hide_index=True)

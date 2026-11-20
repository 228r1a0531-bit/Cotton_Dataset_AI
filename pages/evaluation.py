"""Explore actual evaluation bundles and separate published fold references."""

import pandas as pd
import plotly.express as px
import streamlit as st

from config import RESULTS
from utils.dashboard import PALETTE, chart, heading, read_json, table
from utils.dataset import CLASSES
from utils.evaluation import measure
from utils.research_results import PUBLISHED, published_benchmarks

heading("Evaluation Metrics", "Explore held-out predictions, class-level performance, and measured model costs.")
benchmarks = published_benchmarks()
published_mlp = next(row["accuracy"] for row in benchmarks["final_classification"] if row["model"] == "Hybrid + MLP")
st.metric("Published final classification reference", f"{published_mlp:.2f}%")
st.caption(PUBLISHED)

sources = {"Standalone result files": RESULTS}
protocols = {}
for pattern, kind in (("evaluations/*/experiment.json", "Experiment"), ("training_runs/*/training.json", "Training run")):
    for manifest in sorted(RESULTS.glob(pattern), reverse=True):
        state = read_json(manifest)
        if state.get("status") == "Completed":
            sources[f"{kind} · {manifest.parent.name}"] = manifest.parent
            protocols[manifest.parent] = state
            if state.get("protocol") in ("ablation", "loss"):
                for trial in sorted(manifest.parent.glob("training_runs/*/training.json")):
                    spec = read_json(trial)
                    configuration = spec.get("configuration")
                    if spec.get("status") == "Completed" and isinstance(configuration, dict):
                        name = configuration.get("Model", "Training trial")
                        loss = configuration.get("Loss", trial.parent.name)
                        sources[f"{name} · {loss} · {manifest.parent.name}"] = trial.parent
source = st.selectbox("Local results source", list(sources), index=1 if len(sources) > 1 else 0)
directory = sources[source]
protocol = protocols.get(directory, {})
if protocol.get("protocol") == "external":
    scored = protocol.get("scored_images")
    overlaps = protocol.get("excluded_source_duplicates")
    if isinstance(scored, int) and isinstance(overlaps, int):
        st.caption(f"External test set: {scored} scored images. Exact source overlaps excluded: {overlaps}.")


def results(filename, columns):
    """Read an artifact within the selected protocol rather than merging runs."""
    return table(str((directory / filename).relative_to(RESULTS)), columns)


def pending(description):
    """Explain which experiment can supply a missing result."""
    with st.container(border=True):
        st.markdown("**Pending evaluation**")
        st.write(description)


def load_prediction_metrics():
    """Recover evaluation values directly from saved labels and probabilities."""
    predictions = results("predictions.csv", ("actual", *CLASSES))
    if predictions is None:
        return None
    try:
        labels = [CLASSES.index(label) for label in predictions["actual"]]
        return measure(labels, predictions[list(CLASSES)].to_numpy())
    except (TypeError, ValueError) as error:
        st.warning(f"Could not evaluate saved predictions: {error}")
        return None


recorded = load_prediction_metrics()
metrics = results("metrics.csv", ("metric", "value"))
if recorded is not None:
    metrics = pd.DataFrame(recorded.summary.items(), columns=["metric", "value"])


summary, confusion, roc, folds = st.tabs(
    ["Summary & complexity", "Confusion matrix", "ROC / AUC", "Five-fold cross-validation"]
)
with summary:
    st.subheader("Overall and macro metrics")
    if metrics is None:
        st.metric("Actual model accuracy", "Unavailable")
        pending("A completed training run, cross-validation run, or external evaluation will supply these metrics.")
    else:
        accuracy = pd.to_numeric(metrics.loc[metrics["metric"] == "accuracy", "value"], errors="coerce").dropna()
        st.metric("Actual model accuracy", f"{accuracy.iloc[0]:.2%}" if not accuracy.empty else "Unavailable")
        labels = {"accuracy": "Accuracy", "precision": "Macro precision", "recall": "Macro recall",
                  "f1": "Macro F1-score", "sensitivity": "Macro sensitivity", "specificity": "Macro specificity",
                  "macro_auc": "Macro ROC AUC"}
        st.dataframe(metrics.assign(metric=metrics["metric"].replace(labels)), hide_index=True)
        st.caption("Macro scores give each class equal weight. Sensitivity and recall use the same true-positive rate.")
    classes = results("class_metrics.csv", ("class", "precision", "recall", "f1", "sensitivity",
                                            "specificity", "support"))
    if recorded is not None:
        classes = pd.DataFrame(recorded.per_class,
                               columns=["class", "precision", "recall", "f1", "sensitivity", "specificity", "support"])
    if classes is not None:
        st.subheader("Per-class metrics")
        st.dataframe(classes, hide_index=True)

    st.subheader("Measured training time and model size")
    complexity = results("complexity.csv", ("model", "training_seconds", "parameters"))
    if complexity is None:
        pending("Run training or an ablation to record elapsed fit time, parameter count, and checkpoint size.")
    else:
        st.dataframe(complexity, hide_index=True, column_config={
            "training_seconds": st.column_config.NumberColumn("Fit time (s)", format="%.2f"),
            "parameters": st.column_config.NumberColumn("Parameters", format="%d"),
            "file_mb": st.column_config.NumberColumn("Checkpoint (MB)", format="%.2f"),
        })
        st.caption("Fit time includes validation and callback work. Keras counts parameters, and checkpoint size uses decimal MB.")
    st.subheader("Analytical complexity")
    with st.container(border=True):
        st.markdown(
            "For a dense convolution, time scales with **H × W × K² × Cᵢₙ × Cₒᵤₜ** "
            "per image. A convolutional activation has **H × W × Cₒᵤₜ** elements. "
            "Batch size multiplies activation storage. Multiply the number of elements by bytes per element "
            "to estimate tensor storage."
        )
        st.markdown(
            "For an SE block with **C** channels and bottleneck width **d**, pooling and scaling cost "
            "**O(HWC)** and the gate's dense operations cost **O(Cd)**. Multi-scale and fusion work add "
            "across their branches. Training also retains activations for backpropagation and optimizer state. "
            "Exact peak memory and operation timing require device profiler measurements."
        )
    st.caption("Optional external evaluation and matched ablation commands are documented in training/README.md.")

with confusion:
    matrix = results("confusion.csv", ("actual", "predicted", "count"))
    if matrix is None and recorded is None:
        pending("Ground-truth labels and checkpoint predictions are needed for a confusion matrix.")
    else:
        grid = (pd.DataFrame(recorded.confusion, index=CLASSES, columns=CLASSES) if recorded is not None
                else matrix.pivot_table(index="actual", columns="predicted", values="count", aggfunc="sum"))
        chart(px.imshow(grid, text_auto=True, color_continuous_scale="Greens", aspect="auto",
                        labels={"x": "Predicted", "y": "Actual", "color": "Count"}))
        st.caption("Diagonal cells are correct classifications. Off-diagonal cells show where classes were confused.")

with roc:
    points = results("roc.csv", ("class", "fpr", "tpr", "auc"))
    if recorded is not None:
        points = pd.DataFrame(recorded.roc, columns=["class", "fpr", "tpr", "auc"])
    if points is None or points.empty:
        pending("A ROC curve needs actual probabilities and both positive and negative labels for the class.")
    else:
        curve = px.line(points, x="fpr", y="tpr", color="class", color_discrete_sequence=PALETTE,
                        labels={"fpr": "False positive rate", "tpr": "True positive rate"})
        curve.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line=dict(dash="dash", color="#667a73"))
        curve.update_xaxes(range=[0, 1])
        curve.update_yaxes(range=[0, 1.02])
        chart(curve)
        st.dataframe(points[["class", "auc"]].drop_duplicates(), hide_index=True)
        if metrics is not None:
            macro_auc = pd.to_numeric(
                metrics.loc[metrics["metric"] == "macro_auc", "value"], errors="coerce"
            ).dropna()
            if not macro_auc.empty:
                st.metric("Macro one-vs-rest AUC", f"{macro_auc.iloc[0]:.4f}")
        st.caption("Each curve treats one class as positive and pools the other classes as negatives.")

with folds:
    st.subheader("Published reference values")
    st.caption(PUBLISHED)
    reference = pd.DataFrame(benchmarks["five_fold_accuracy"])
    reference["fold"] = "Fold " + reference["fold"].astype(str)
    figure = px.bar(reference, x="fold", y="accuracy", text_auto=".1f", color_discrete_sequence=PALETTE,
                    labels={"fold": "Fold", "accuracy": "Published accuracy (%)"})
    figure.update_yaxes(range=[0, 108], ticksuffix="%")
    figure.update_traces(textposition="outside", texttemplate="%{y:.1f}%")
    chart(figure, key="evaluation_reference_folds", png_name="evaluation_reference_folds")
    st.caption("These reference scores come from the supplied publication figures.")

    st.subheader("Local stratified five-fold evaluation")
    scores = results("folds.csv", ("fold", "metric", "value"))
    if scores is None:
        pending("Run the five-fold evaluation and select its completed result bundle to see local scores.")
    else:
        metric = st.selectbox("Local fold metric", scores["metric"].drop_duplicates())
        selected = scores.loc[scores["metric"] == metric].sort_values("fold").copy()
        selected["fold"] = "Fold " + selected["fold"].astype(str)
        figure = px.bar(selected, x="fold", y="value", color_discrete_sequence=PALETTE,
                        labels={"fold": "Fold", "value": metric})
        figure.update_yaxes(range=[0, 1], tickformat=".0%")
        chart(figure)
        st.dataframe(scores, hide_index=True)
        st.caption("Each outer test fold uses a fresh model. The validation split comes from that fold's development data.")

"""Manage the dataset, two real training workflows, and a presentation replay."""

from collections import Counter
from time import monotonic

import pandas as pd
import plotly.express as px
import streamlit as st

from config import RESULTS
from training.run import FULL_EPOCHS, MODEL_NAMES, settings, train
from utils.dashboard import PALETTE, chart, empty, heading, read_json, table
from utils.dataset import CLASSES, SPLITS, discover, save_splits, scan, stratify, unique_samples

from utils.training_preview import (
    PREVIEW_EPOCHS, PREVIEW_LABEL, PREVIEW_SECONDS, RESEARCH_TARGET, preview_epoch, preview_history,
)

heading("Dataset & Training", "Verify local images, run real training, or replay the research target for a presentation.")
st.subheader("Dataset location")
location = st.text_input(
    "Local dataset root",
    placeholder="Leave blank to use the environment variable or auto-discovery",
    help="The root should contain a folder for each expected class.",
)
rescan = st.button("Scan dataset", icon=":material/folder_open:")
dataset = None
splits = None
try:
    root = discover(location)
    if root is None:
        empty("No dataset found. Enter a local root or set COTTONLEAF_DATASET_ROOT.")
    elif not root.is_dir():
        st.warning(f"Dataset folder not found: {root}")
    else:
        st.caption(f"Dataset root: {root}")
        previous = st.session_state.get("dataset_scan")
        if rescan or previous is None or previous.root != root:
            with st.spinner("Verifying images and checking for duplicate content…"):
                st.session_state["dataset_scan"] = scan(root)
        dataset = st.session_state["dataset_scan"]
except (OSError, ValueError) as error:
    st.error(f"Could not scan the dataset: {error}")

if dataset is not None:
    duplicates = dataset.duplicates
    valid = Counter(image.label for image in dataset.images)
    distribution = pd.DataFrame({"class": CLASSES, "count": [valid[label] for label in CLASSES]})
    valid_col, eligible_col, unreadable_col, duplicates_col = st.columns(4)
    valid_col.metric("Valid images", len(dataset.images))
    eligible_col.metric("Distinct eligible images", len(unique_samples(dataset)))
    unreadable_col.metric("Unreadable files", len(dataset.unreadable))
    duplicates_col.metric("Duplicate groups", len(duplicates))
    st.subheader("Class distribution")
    st.dataframe(distribution, hide_index=True)
    if dataset.images:
        bar, pie = st.columns(2)
        with bar:
            chart(px.bar(distribution, x="class", y="count", color="class", color_discrete_sequence=PALETTE))
        with pie:
            chart(px.pie(distribution, names="class", values="count", hole=0.48, color_discrete_sequence=PALETTE))
    if missing := set(CLASSES) - dataset.matched_classes:
        st.warning(f"Class folders missing: {', '.join(sorted(missing))}")
    if dataset.unreadable:
        with st.expander("Unreadable files"):
            st.dataframe(
                [{"path": path, "reason": reason} for path, reason in dataset.unreadable.items()],
                hide_index=True,
            )
    if duplicates:
        with st.expander("Duplicate images"):
            st.caption("Same-label copies contribute one image per group. Groups with conflicting labels are excluded.")
            st.dataframe([
                {"files": " | ".join(image.path for image in group),
                 "labels": ", ".join(sorted({image.label for image in group}))}
                for group in duplicates
            ], hide_index=True)

    st.subheader("Train, validation, and test splits")
    st.caption("Target 70 / 15 / 15, stratified by class with seed 42. Small classes may need rounding.")
    try:
        splits = stratify(dataset)
    except ValueError as error:
        empty(str(error))
    else:
        rows = {name: Counter(image.label for image in splits[name]) for name in SPLITS}
        summary = pd.DataFrame(rows).reindex(CLASSES).rename_axis("class")
        st.dataframe(summary.assign(total=summary.sum(axis=1)))
        st.caption(f"{sum(map(len, splits.values()))} distinct images are eligible for training and evaluation.")
        if st.button("Save split metadata"):
            try:
                save_splits(RESULTS, dataset, splits)
            except OSError as error:
                st.error(f"Could not save metadata: {error}")
            else:
                st.success("Saved results/splits.json and results/dataset.csv.")

LOSS_LABELS = {"cross_entropy": "Categorical cross-entropy", "focal": "Categorical focal loss",
               "hybrid": "Weighted CE + focal loss"}


def render_history(history, prefix):
    """Display train and validation series using the same axes in both workflows."""
    for metric in ("accuracy", "loss"):
        selected = history.loc[history["metric"].astype("string").str.lower() == metric].sort_values("epoch")
        if selected.empty:
            continue
        figure = px.line(selected, x="epoch", y="value", color="split", markers=True,
                         title=metric.capitalize(), color_discrete_sequence=PALETTE)
        if metric == "accuracy":
            figure.update_yaxes(tickformat=".0%", range=[0, 1.03])
        if prefix == "prototype_preview":
            figure.update_xaxes(range=[1, PREVIEW_EPOCHS])
            figure.update_layout(title=f"{metric.capitalize()} · {PREVIEW_LABEL}")
            if metric == "loss":
                figure.update_yaxes(range=[0, 1.65])
        chart(figure, key=f"{prefix}_history_{metric}")


def latest_run(architecture):
    """Find this mode's latest run, including runs created before mode IDs were saved."""
    for path in sorted((RESULTS / "training_runs").glob("*/training.json"), reverse=True):
        state = read_json(path)
        configuration = state.get("configuration")
        if state.get("architecture") == architecture or (
            isinstance(configuration, dict) and configuration.get("Model") == MODEL_NAMES[architecture]
        ):
            return path.parent, state
    if architecture == "baseline":
        return RESULTS, read_json(RESULTS / "training.json")
    return RESULTS, {}


def render_training(architecture, label):
    """Run and display only the real training artifacts for this architecture."""
    is_hybrid = architecture == "hybrid"
    st.subheader(label)
    st.caption("CN2-SE + Modified InceptionV3 + MLP" if is_hybrid else "Compact CNN · lightweight checkpoint")
    if is_hybrid:
        st.metric("Published research target", f"{RESEARCH_TARGET:.2%}")
        st.caption("Real training reports the observed metrics for your dataset.")
    epochs = st.number_input(f"{label} epochs", min_value=1, max_value=FULL_EPOCHS if is_hybrid else 10,
                             value=FULL_EPOCHS if is_hybrid else 2, key=f"{architecture}_epochs")
    loss_name = st.selectbox("Loss function", LOSS_LABELS, index=2 if is_hybrid else 0,
                             format_func=LOSS_LABELS.__getitem__, key=f"{architecture}_loss")
    loss_options = {}
    with st.expander("Training settings"):
        if loss_name in ("focal", "hybrid"):
            loss_options["gamma"] = st.number_input("Focal gamma", min_value=0.0, value=2.0,
                                                    key=f"{architecture}_gamma")
            loss_options["alpha"] = st.number_input("Focal alpha", min_value=0.01, max_value=1.0,
                                                   value=0.25, key=f"{architecture}_alpha")
        if loss_name == "hybrid":
            loss_options["ce_weight"] = st.number_input("Cross-entropy coefficient", min_value=0.0,
                                                       value=0.5, key=f"{architecture}_ce_weight")
            loss_options["focal_weight"] = st.number_input("Focal coefficient", min_value=0.0,
                                                          value=0.5, key=f"{architecture}_focal_weight")
        st.dataframe(pd.DataFrame(settings(epochs, architecture, loss_name).items(),
                                  columns=["Setting", "Value"], dtype=str), hide_index=True)
    invalid = loss_name == "hybrid" and not (loss_options["ce_weight"] or loss_options["focal_weight"])
    if invalid:
        st.warning("Set at least one loss coefficient above zero.")
    start = st.button(f"Start {label} training", type="primary", disabled=splits is None or invalid,
                      key=f"{architecture}_train")
    progress = st.empty()
    if start and dataset is not None:
        live = st.empty()

        def show_epoch(completed, total, summary):
            progress.progress(completed / total, text=f"Epoch {completed} of {total}")
            live.caption(summary)

        try:
            with st.spinner(f"Training {label} on verified images…"):
                fresh = scan(dataset.root)
                st.session_state["dataset_scan"] = fresh
                fresh_splits = stratify(fresh)
                save_splits(RESULTS, fresh, fresh_splits)
                progress.progress(0.0, text=f"Starting {label}")
                train(fresh, fresh_splits, int(epochs), RESULTS, show_epoch,
                      architecture=architecture, loss_name=loss_name, loss_options=loss_options)
        except ImportError:
            st.error("Install requirements-ml.txt to enable training.")
        except Exception as error:
            st.error(f"Training could not complete: {error}")
        finally:
            live.empty()

    run, state = latest_run(architecture)
    st.markdown("**Actual training results**")
    st.write(state.get("status", "Not trained yet"))
    if run != RESULTS:
        st.caption(f"Latest run: {run.name}")
        configuration = state.get("configuration")
        target = configuration.get("Epochs") if isinstance(configuration, dict) else None
        completed = state.get("completed")
        if type(target) is int and type(completed) is int and target > 0 and 0 <= completed <= target:
            progress.progress(completed / target, text=f"{completed} of {target} epochs recorded")
        if state.get("status") == "Completed" and type(best := state.get("best_epoch")) is int:
            st.caption(f"Checkpoint restored from validation-loss best epoch {best}.")
        test = state.get("test")
        if isinstance(test, dict) and all(isinstance(test.get(key), (int, float)) for key in ("accuracy", "loss")):
            st.write(f"Held-out test accuracy: {test['accuracy']:.2%} · Test loss: {test['loss']:.4f}")
    if run != RESULTS or not is_hybrid:
        history = table(str((run / "history.csv").relative_to(RESULTS)), ("epoch", "split", "metric", "value"))
        if history is not None:
            render_history(history, architecture)
        log_path = run / "training.log"
        if log_path.is_file():
            with st.expander("Actual epoch logs"):
                try:
                    with log_path.open("rb") as log:
                        log.seek(max(0, log_path.stat().st_size - 16_384))
                        st.code(log.read(16_384).decode("utf-8", errors="replace"), language=None)
                except OSError:
                    empty("The training log could not be read.")


st.subheader("Training workflows")
fast, full = st.columns(2)
with fast.container(border=True):
    render_training("baseline", "Fast Demo")
with full.container(border=True):
    render_training("hybrid", "Full Hybrid")

with st.container(border=True):
    st.subheader("Full Hybrid presentation preview")
    st.info(PREVIEW_LABEL)
    st.caption(f"Illustrative accuracy and loss over {PREVIEW_SECONDS} seconds. This replay uses no dataset, "
               "writes no metrics to disk, and produces no model checkpoint.")
    if st.button("Preview 50-Epoch Research Target"):
        st.session_state["research_preview_started"] = monotonic()
        st.session_state["research_preview_running"] = True

    @st.fragment(run_every=0.5 if st.session_state.get("research_preview_running") else None)
    def render_preview():
        """Advance only the presentation state while real workflows stay independent."""
        started = st.session_state.get("research_preview_started")
        if started is None:
            return
        epoch = preview_epoch(monotonic() - started)
        if epoch == PREVIEW_EPOCHS and st.session_state.get("research_preview_running"):
            st.session_state["research_preview_running"] = False
            st.rerun()
        st.progress(epoch / PREVIEW_EPOCHS, text=f"Illustrative epoch {epoch} of {PREVIEW_EPOCHS}")
        history = pd.DataFrame(preview_history())
        visible = history.loc[history["epoch"] <= epoch]
        validation = visible.loc[(visible["split"] == "validation") & (visible["metric"] == "accuracy")]
        st.metric("Simulated validation accuracy", f"{validation['value'].iloc[-1]:.2%}")
        render_history(visible, "prototype_preview")

    render_preview()

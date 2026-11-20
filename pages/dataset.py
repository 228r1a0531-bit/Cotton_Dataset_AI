"""Manage a local dataset and run or review baseline training."""

from collections import Counter

import pandas as pd
import plotly.express as px
import streamlit as st

from config import RESULTS
from training.run import FULL_EPOCHS, settings, train
from utils.dashboard import PALETTE, chart, empty, heading, read_json, table
from utils.dataset import CLASSES, SPLITS, discover, save_splits, scan, stratify

heading("Dataset & Training", "Verify local images, prepare stratified splits, and train a baseline classifier.")
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
    valid_col, unreadable_col, duplicates_col = st.columns(3)
    valid_col.metric("Valid images", len(dataset.images))
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

st.subheader("Training configuration")
mode = st.radio("Training mode", ("Fast Demo", "Full training"), horizontal=True)
epochs = (
    st.number_input("Fast Demo epochs", min_value=1, max_value=10, value=2)
    if mode == "Fast Demo" else FULL_EPOCHS
)
st.dataframe(pd.DataFrame(settings(epochs).items(), columns=["Setting", "Value"], dtype=str), hide_index=True)
st.caption("Full training uses 50 epochs. This trains a baseline CNN. Validation and test images use deterministic preprocessing.")
start = st.button("Start training", type="primary", disabled=splits is None)
progress = st.empty()
if start and dataset is not None:
    latest_epoch = st.empty()

    def show_epoch(completed, total, summary):
        progress.progress(completed / total, text=f"Epoch {completed} of {total}")
        latest_epoch.caption(summary)

    try:
        with st.spinner("Preparing verified splits and training the baseline CNN…"):
            fresh = scan(dataset.root)
            st.session_state["dataset_scan"] = fresh
            fresh_splits = stratify(fresh)
            save_splits(RESULTS, fresh, fresh_splits)
            progress.progress(0.0, text=f"Starting {mode.lower()}")
            train(fresh, fresh_splits, int(epochs), RESULTS, show_epoch)
    except ImportError:
        st.error("Install requirements-ml.txt to enable training.")
    except Exception as error:
        st.error(f"Training could not complete: {error}")
    finally:
        latest_epoch.empty()

st.subheader("Training status")
runs = sorted((RESULTS / "training_runs").glob("*/training.json"), reverse=True)
run = runs[0].parent if runs else RESULTS
state = read_json(run / "training.json")
st.write(state.get("status", "Not trained yet"))
if runs and state:
    target = state["configuration"]["Epochs"]
    completed = state["completed"]
    progress.progress(completed / target, text=f"{completed} of {target} epochs recorded")
    st.caption(f"Latest run: {run.name}")
    if test := state.get("test"):
        st.write(f"Held-out test accuracy: {test['accuracy']:.2%} · Test loss: {test['loss']:.4f}")

st.subheader("Training history")
history = table(str((run / "history.csv").relative_to(RESULTS)), ("epoch", "split", "metric", "value"))
if history is None:
    empty("No epoch metrics have been recorded.")
else:
    for metric, column in zip(("accuracy", "loss"), st.columns(2)):
        series = history.loc[history["metric"].str.lower() == metric].sort_values("epoch")
        if not series.empty:
            with column:
                chart(px.line(series, x="epoch", y="value", color="split", markers=True,
                              title=metric.capitalize(), color_discrete_sequence=PALETTE))

st.subheader("Epoch logs")
log_path = run / "training.log"
if log_path.is_file():
    try:
        with log_path.open("rb") as log:
            log.seek(max(0, log_path.stat().st_size - 16_384))
            st.code(log.read(16_384).decode("utf-8", errors="replace"), language=None)
    except OSError:
        empty("The training log could not be read.")
else:
    empty("No training log is available.")

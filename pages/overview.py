"""Research objective and available evidence."""

import streamlit as st

from config import APP_TITLE, RESULTS
from utils.dashboard import empty, heading, read_json, table

heading(APP_TITLE, "Hybrid deep learning for cotton leaf disease classification")
st.subheader("Research objective")
st.write(
    "Evaluate whether channel attention, feature fusion, and a hybrid loss improve "
    "cotton leaf disease classification against matched baselines."
)

metadata = read_json(RESULTS / "research.json")
counts = table("dataset.csv", ("class", "split", "count"))
benchmark = metadata.get("benchmark")
if not isinstance(benchmark, dict) or not all(
    isinstance(benchmark.get(key), str) and benchmark[key].strip()
    for key in ("accuracy", "citation")
):
    benchmark = {}
images, classes, published = st.columns(3)
images.metric("Dataset images", f"{counts['count'].sum():,}" if counts is not None else "Unavailable")
classes.metric("Target classes", "4")
published.metric("Published accuracy", benchmark.get("accuracy", "Unavailable"))
st.caption(benchmark.get("citation", "A source citation and published accuracy are needed to show a benchmark."))

labels = metadata.get("classes")
st.subheader("Cotton leaf classes")
if not isinstance(labels, list) or len(labels) != 4 or not all(
    isinstance(label, str) and label.strip() for label in labels
):
    empty("Four class labels are expected. The dataset's class names have not been provided.")
else:
    for column, label in zip(st.columns(4), labels):
        with column.container(border=True):
            st.write(label)

st.subheader("Research architecture")
st.write(
    "The proposed pipeline combines a channel-attention CNN (CN2-SE) with "
    "a Modified InceptionV3 branch. Their extracted features feed an MLP classifier."
)
st.page_link("pages/architecture.py", label="Explore the model architecture", icon=":material/arrow_forward:")

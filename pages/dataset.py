"""Dataset distribution and training artifacts."""

import plotly.express as px
import streamlit as st

from config import RESULTS
from utils.dashboard import PALETTE, chart, empty, heading, read_json, table

heading("Dataset & Training", "Inspect class balance, dataset splits, and recorded training runs.")
st.subheader("Dataset distribution and splits")
counts = table("dataset.csv", ("class", "split", "count"))
if counts is None:
    empty("Dataset counts are unavailable. Add the recorded class and split counts in results/dataset.csv.")
else:
    totals = counts.groupby("split", as_index=False)["count"].sum()
    total_images = totals["count"].sum()
    if total_images > 0:
        totals["share"] = totals["count"].map(lambda count: f"{count / total_images:.1%}")
    st.dataframe(totals, hide_index=True)
    chart(px.bar(counts, x="class", y="count", color="split", color_discrete_sequence=PALETTE))

configuration, status = st.columns(2)
training = read_json(RESULTS / "training.json")
with configuration:
    st.subheader("Training configuration")
    if training.get("configuration"):
        st.json(training["configuration"])
    else:
        empty("No training configuration has been recorded.")
with status:
    st.subheader("Training status")
    st.write(training.get("status", "No run status recorded."))

st.subheader("Training history")
history = table("history.csv", ("epoch", "split", "metric", "value"))
if history is None:
    empty("Training curves will appear when recorded epoch metrics are available.")
else:
    metric = st.selectbox("Metric", history["metric"].unique())
    series = history.loc[history["metric"] == metric].sort_values("epoch")
    chart(px.line(series, x="epoch", y="value", color="split", color_discrete_sequence=PALETTE))

st.subheader("Training logs")
log_path = RESULTS / "training.log"
if log_path.is_file():
    try:
        with log_path.open("rb") as log:
            log.seek(max(0, log_path.stat().st_size - 16_384))
            st.code(log.read(16_384).decode("utf-8", errors="replace"), language=None)
    except OSError:
        empty("The training log could not be read.")
else:
    empty("No training log is available.")

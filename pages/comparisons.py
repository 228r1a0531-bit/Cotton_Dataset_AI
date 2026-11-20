"""Compare only recorded experimental metrics."""

import plotly.express as px
import streamlit as st

from utils.dashboard import PALETTE, chart, empty, heading, table

heading("Model Comparisons", "Explore measured changes across model variants and loss functions.")
comparisons = {
    "CNN vs CNN+SE": "Tests the contribution of squeeze-and-excitation channel attention.",
    "InceptionV3 vs Modified InceptionV3": "Tests the contribution of changes to the Inception branch.",
    "Feature fusion": "Compares standalone feature branches with the combined representation.",
    "Cross-Entropy vs Hybrid Loss": "Compares loss functions under the same evaluation protocol.",
}
results = table("comparisons.csv", ("comparison", "variant", "metric", "value"))
for (name, description), tab in zip(comparisons.items(), st.tabs(list(comparisons))):
    with tab:
        st.write(description)
        data = results.loc[results["comparison"] == name] if results is not None else None
        if data is None or data.empty:
            empty("No measured comparison is available for these variants.")
        else:
            chart(px.bar(data, x="variant", y="value", color="metric", barmode="group",
                         color_discrete_sequence=PALETTE))
            st.dataframe(data[["variant", "metric", "value"]], hide_index=True)

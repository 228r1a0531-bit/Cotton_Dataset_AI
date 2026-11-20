"""Present supplied research benchmarks and independently recorded local results."""

import pandas as pd
import plotly.express as px
import streamlit as st

from config import RESULTS
from utils.dashboard import PALETTE, chart, empty, heading, table
from utils.research_results import (
    LOCAL, METRICS, PUBLISHED, QUALITY_METRICS, STAGES, citation, latest_training_accuracy, local_percentages,
    published_benchmarks, published_metrics,
)

PANELS = {
    "CNN vs CNN+SE": (
        ("CNN", "Convolution, ReLU, pooling, and global average pooling."),
        ("CNN+SE", "The same convolution stages add learned channel gates before pooling."),
    ),
    "InceptionV3 vs Modified InceptionV3": (
        ("Standard InceptionV3", "Keras Applications backbone with the standard stem and mixed blocks."),
        ("Modified InceptionV3", "Shorter stem and compact parallel multi-scale blocks in the local design."),
    ),
    "Cross-Entropy vs Hybrid Loss": (
        ("Categorical cross-entropy", "Penalizes low probability assigned to the true class."),
        ("Hybrid CE + focal loss", "Adds a configurable focal term that emphasizes lower-confidence examples."),
    ),
}


def percentage_chart(data, key, interpretation, *, color="Model", x="Model", grouped=False):
    """Render percentages on a common scale and enable Plotly's PNG export."""
    fig = px.bar(
        data, x=x, y="Percent", color=color, text_auto=".2f",
        barmode="group" if grouped else "relative", color_discrete_sequence=PALETTE,
        hover_name=x, hover_data={"Source": True, "Context": True, "Percent": ":.2f"},
        labels={"Percent": "Score (%)"},
    )
    fig.update_yaxes(range=[0, 108], tickvals=[0, 20, 40, 60, 80, 100], ticksuffix="%")
    fig.update_traces(textposition="outside", texttemplate="%{y:.2f}%", cliponaxis=False)
    if color == x:
        fig.update_layout(showlegend=False)
    chart(fig, key=key, png_name=key)
    st.caption(interpretation)


def export_table(data, name):
    """Display and export the source-tagged rows underlying the view."""
    st.dataframe(data, hide_index=True, column_config={
        "Percent": st.column_config.NumberColumn("Score (%)", format="%.2f"),
    })
    st.download_button("Download CSV", data.to_csv(index=False, float_format="%.2f"),
                       f"{name}.csv", mime="text/csv", key=f"csv_{name}")


heading("Model Comparisons", "Compare the paper's feature extractors and classifiers, then inspect local experiments.")
benchmarks = published_benchmarks()
published = published_metrics(benchmarks)
research_tab, experiment_tab = st.tabs(["Published Research", "Our Experiment"])

with research_tab:
    st.info(PUBLISHED)
    st.caption(citation(benchmarks))
    interpretations = {
        "feature_extraction": "The combined CN2-SE and Modified InceptionV3 feature extractor has the highest reported accuracy in this feature-extraction comparison.",
        "final_classification": "Among the classifiers evaluated on hybrid features, the MLP has the highest reported final classification accuracy.",
    }
    for key, stage in STAGES.items():
        st.subheader(f"{stage} accuracy")
        accuracy = published.loc[(published["Context"] == stage) & (published["Metric"] == "Accuracy")]
        percentage_chart(accuracy, key, interpretations[key])

    st.subheader("Before vs after feature extraction")
    before, after = st.columns(2)
    with before.container(border=True):
        st.markdown("**Before · Standalone CNN**")
        st.write("Convolutional features from a single branch.")
    with after.container(border=True):
        st.markdown("**After · Combined feature extractor**")
        st.write("CN2-SE channel attention and Modified InceptionV3 multi-scale features.")
    feature = published.loc[
        (published["Context"] == STAGES["feature_extraction"]) & (published["Metric"] == "Accuracy")
        & published["Model"].isin(("CNN", "CN2-SE + Modified InceptionV3"))
    ]
    percentage_chart(feature, "before_after", (
        "The published scores compare the standalone CNN with the complete proposed extractor. "
        "A matched ablation would be needed to isolate each component's contribution."
    ))

    st.subheader("Reported sensitivity, recall, and specificity")
    quality = published.loc[published["Metric"].isin([METRICS[key] for key in QUALITY_METRICS])]
    percentage_chart(quality, "published_quality", (
        "The authors reported these metrics for the proposed extractor and the final MLP classifier. "
        "Precision and F1 values were not supplied."
    ), color="Context", x="Metric", grouped=True)

    st.subheader("Five-fold research accuracy")
    if benchmarks["five_fold_accuracy"]:
        folds = pd.DataFrame(benchmarks["five_fold_accuracy"]).rename(columns={"fold": "Fold", "accuracy": "Percent"})
        folds["Fold"] = "Fold " + folds["Fold"].astype(str)
        folds = folds.assign(Source=PUBLISHED, Context="Five-fold cross-validation", Model="Published model")
        percentage_chart(folds, "published_folds", "Each bar represents one reported research fold.", color="Fold", x="Fold")
        export_table(folds, "published_folds")
    else:
        empty("Per-fold research accuracies were not supplied. Add the five sourced values to display this chart.")

    st.subheader("Architecture and loss ablations")
    for name, variants in PANELS.items():
        st.markdown(f"**{name}**")
        for column, (variant, description) in zip(st.columns(2), variants):
            with column.container(border=True):
                st.markdown(f"**{variant}**")
                st.write(description)
        st.caption("Experimental result pending")

    st.subheader("Published benchmark table")
    export_table(published, "published_benchmarks")

with experiment_tab:
    st.subheader("Locally measured results")
    local_tables = []
    run_accuracy = latest_training_accuracy(RESULTS)
    if not run_accuracy.empty:
        local_tables.append(run_accuracy)
        st.markdown("**Latest completed training run**")
        percentage_chart(run_accuracy, "local_training_accuracy", (
            "This is the accuracy measured on the saved test split for the indicated run."
        ))
    comparisons = table("comparisons.csv", ("comparison", "variant", "metric", "value"))
    if comparisons is not None:
        for name, rows in comparisons.groupby("comparison", sort=False):
            scores = local_percentages(rows, str(name), "variant")
            if scores.empty:
                continue
            local_tables.append(scores)
            st.markdown(f"**{name}**")
            percentage_chart(scores, f"local_comparison_{len(local_tables)}", (
                "These values come from the saved local experiment. Compare variants evaluated under the same split and protocol."
            ), color="Metric", grouped=True)

    metrics = table("metrics.csv", ("metric", "value"))
    if metrics is not None:
        scores = local_percentages(metrics, "Held-out evaluation", "model" if "model" in metrics else None)
        if not scores.empty:
            local_tables.append(scores)
            available = scores.loc[scores["Metric"].isin([METRICS[key] for key in QUALITY_METRICS])]
            if not available.empty:
                st.markdown("**Held-out precision, recall, F1, and specificity**")
                percentage_chart(available, "local_quality", (
                    "These are the saved metrics for the local held-out evaluation. Available metrics depend on what was recorded."
                ), x="Metric", grouped=True)

    folds = table("folds.csv", ("fold", "metric", "value"))
    if folds is not None:
        scores = local_percentages(folds, "Five-fold cross-validation", "fold")
        scores["Model"] = "Fold " + scores["Model"].astype(str)
        scores = scores.rename(columns={"Model": "Fold"})
        accuracy = scores.loc[scores["Metric"] == "Accuracy"]
        if not accuracy.empty:
            st.markdown("**Five-fold local accuracy**")
            percentage_chart(accuracy, "local_folds", "Each bar shows the recorded accuracy of one local fold.", color="Fold", x="Fold")
        if not scores.empty:
            local_tables.append(scores.rename(columns={"Fold": "Model"}))

    if not local_tables:
        empty("Experimental result pending. Add recorded metrics, comparisons, or fold scores under results/.")

    st.subheader("Published and local metric provenance")
    st.caption(f"Source identifies each row. Published stages and {LOCAL.lower()} protocols are labeled in Context.")
    export_table(pd.concat([published, *local_tables], ignore_index=True), "research_metric_provenance")

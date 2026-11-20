"""Present supplied research benchmarks and independently recorded local results."""

import pandas as pd
import plotly.express as px
import streamlit as st

from config import RESULTS
from utils.comparison_topics import TOPICS, matched_results
from utils.dashboard import PALETTE, chart, empty, heading, read_json, table
from utils.research_results import (
    LOCAL, METRICS, PUBLISHED, QUALITY_METRICS, STAGES, citation, latest_training_accuracy, local_percentages,
    published_benchmarks, published_metrics,
)

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
                       f"{name}.csv", mime="text/csv", key=f"csv_{name}", on_click="ignore")


heading("Model Comparisons", "Compare the paper's feature extractors and classifiers, then inspect local experiments.")
benchmarks = published_benchmarks()
published = published_metrics(benchmarks)
research_tab, experiment_tab = st.tabs(["Published Research", "Our Experiment"])
completed = [
    path.parent.relative_to(RESULTS)
    for path in sorted(RESULTS.glob("evaluations/*/experiment.json"), reverse=True)
    if read_json(path).get("status") == "Completed"
]
comparison_data = []
for directory in [None, *completed]:
    filename = str(directory / "comparisons.csv") if directory else "comparisons.csv"
    comparisons = table(filename, ("comparison", "variant", "metric", "value"))
    if comparisons is not None:
        for name, rows in comparisons.groupby("comparison", sort=False):
            context = f"{name} · {directory.name}" if directory else str(name)
            scores = local_percentages(rows, context, "variant")
            if not scores.empty:
                comparison_data.append((context, scores))
trained = set()
for path in RESULTS.glob("training_runs/*/training.json"):
    state = read_json(path)
    configuration = state.get("configuration")
    if state.get("status") == "Completed" and isinstance(configuration, dict):
        trained.add(configuration.get("Model"))
        if configuration.get("Model") == "Hybrid feature fusion":
            losses = {"cross_entropy": "Categorical cross-entropy", "hybrid": "Hybrid CE + focal loss"}
            trained.add(losses.get(configuration.get("Loss")))

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
        percentage_chart(folds, "published_folds", "Published reference values for the five research folds.", color="Fold", x="Fold")
        export_table(folds, "published_folds")
    else:
        empty("Per-fold research accuracies were not supplied. Add the five sourced values to display this chart.")

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
    st.subheader("Architecture and loss comparisons")
    topic = st.selectbox("Comparison", TOPICS)
    variants = TOPICS[topic]
    names = [name for name, _ in variants]
    reported_names = {model for _, scores in comparison_data for model in scores["Model"]}
    for column, (name, description) in zip(st.columns(len(variants)), variants):
        with column.container(border=True):
            st.markdown(f"**{name}**")
            st.write(description)
            local_name = "Hybrid feature fusion" if name == "Hybrid + MLP" else name
            alias = {"Baseline CNN": "CNN", "InceptionV3": "Standard InceptionV3"}.get(local_name, local_name)
            if local_name in trained or reported_names.intersection({name, local_name, alias}):
                st.caption("Local training recorded")
            else:
                st.caption("No completed local training recorded")
    candidates = [matched_results(scores, names) for _, scores in comparison_data]
    matched = next((rows for rows in candidates if not rows.empty), None)
    if matched is not None:
        percentage_chart(matched, "paired_comparison", "Accuracy from a single matched local protocol.")
    else:
        empty("Matched local evaluation pending.")
        reference_names = {"Baseline CNN": "CNN", "Hybrid feature fusion": "CN2-SE + Modified InceptionV3"}
        available = published.loc[(published["Metric"] == "Accuracy")
                                  & published["Model"].isin([reference_names.get(name, name) for name in names])]
        if not available.empty:
            percentage_chart(available, "topic_reference", "Published references for documented variants. "
                             "Missing variants require a matched experiment.")

    for context, scores in comparison_data:
        local_tables.append(scores)
        st.markdown(f"**{context}**")
        percentage_chart(scores, f"local_comparison_{len(local_tables)}", (
            "These values come from the saved local experiment. Compare variants evaluated under the same split and protocol."
        ), color="Metric", grouped=True)

    st.subheader("Training time and model complexity")
    st.caption("Fit time, parameter count, and checkpoint size are read from saved local runs.")
    cost_frames = []
    cost_files = [RESULTS / "complexity.csv", *[RESULTS / path / "complexity.csv" for path in completed]]
    for manifest in sorted(RESULTS.glob("training_runs/*/training.json"), reverse=True):
        if read_json(manifest).get("status") == "Completed":
            cost_files.append(manifest.parent / "complexity.csv")
    for path in cost_files:
        costs = table(str(path.relative_to(RESULTS)), ("model", "training_seconds", "parameters"))
        if costs is not None:
            cost_frames.append(costs.assign(Run=path.parent.name))
    if cost_frames:
        costs = pd.concat(cost_frames, ignore_index=True)
        fields = {"training_seconds": "Fit time (s)", "parameters": "Parameters", "file_mb": "Checkpoint (MB)"}
        field = st.selectbox("Complexity measure", [key for key in fields if key in costs],
                             format_func=fields.__getitem__)
        costs[field] = pd.to_numeric(costs[field], errors="coerce")
        values = costs.dropna(subset=[field])
        if not values.empty:
            figure = px.bar(values, x="model", y=field, color="Run", barmode="group",
                            labels={"model": "Model", field: fields[field]}, color_discrete_sequence=PALETTE)
            chart(figure, key="measured_model_cost", png_name="measured_model_cost")
        else:
            empty("No local values were recorded for this measure.")
        st.dataframe(costs, hide_index=True)
    else:
        empty("Measured fit time, parameter count, and checkpoint size appear after real training.")

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

    for directory in [None, *completed]:
        filename = str(directory / "folds.csv") if directory else "folds.csv"
        folds = table(filename, ("fold", "metric", "value"))
        if folds is None:
            continue
        context = f"Five-fold cross-validation · {directory.name}" if directory else "Five-fold cross-validation"
        scores = local_percentages(folds, context, "fold")
        scores["Model"] = "Fold " + scores["Model"].astype(str)
        scores = scores.rename(columns={"Model": "Fold"})
        accuracy = scores.loc[scores["Metric"] == "Accuracy"]
        if not accuracy.empty:
            st.markdown("**Five-fold local accuracy**")
            st.caption("Completed local cross-validation. Each fold fits a new copy of the chosen architecture.")
            percentage_chart(accuracy, f"local_folds_{len(local_tables)}",
                             "Each bar shows the recorded accuracy of one local fold.", color="Fold", x="Fold")
        if not scores.empty:
            local_tables.append(scores.rename(columns={"Fold": "Model"}))

    if not local_tables:
        empty("Experimental result pending. Add recorded metrics, comparisons, or fold scores under results/.")

    st.subheader("Published and local metric provenance")
    st.caption(f"Source identifies each row. Published stages and {LOCAL.lower()} protocols are labeled in Context.")
    export_table(pd.concat([published, *local_tables], ignore_index=True), "research_metric_provenance")

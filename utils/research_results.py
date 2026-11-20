"""Load published benchmark values and prepare chart-ready metric tables.

Published values are stored as percentages. Local CSVs declare a unit when
they store percentages; otherwise their values are interpreted as fractions.
The two sources carry a provenance label through tables and downloads.
"""

import json

import pandas as pd

from config import ROOT
from utils.dashboard import read_json

PUBLISHED = "Published research results — not reproduced locally."
LOCAL = "Locally measured"
STAGES = {"feature_extraction": "Feature extraction", "final_classification": "Final classification"}
METRICS = {
    "accuracy": "Accuracy",
    "precision": "Precision",
    "recall": "Recall",
    "f1": "F1-score",
    "specificity": "Specificity",
    "sensitivity": "Sensitivity",
}
QUALITY_METRICS = ("precision", "recall", "f1", "specificity", "sensitivity")
PUBLISHED_PATH = ROOT / "research" / "published_benchmarks.json"


def published_benchmarks() -> dict:
    """Read the version-controlled transcription without consulting local runs."""
    return json.loads(PUBLISHED_PATH.read_text(encoding="utf-8"))


def citation(benchmarks: dict) -> str:
    """Format the title and authors associated with the transcription."""
    source = benchmarks["source"]
    return f'{", ".join(source["authors"])}. “{source["title"]}.”'


def published_metrics(benchmarks: dict) -> pd.DataFrame:
    """Flatten the paper's two result stages without filling unreported metrics."""
    return pd.DataFrame([
        {"Source": PUBLISHED, "Context": stage, "Model": entry["model"],
         "Metric": METRICS[metric], "Percent": value}
        for key, stage in STAGES.items()
        for entry in benchmarks[key]
        for metric, value in entry.items()
        if metric in METRICS
    ])


def local_percentages(data: pd.DataFrame, context: str, model_column: str | None = None) -> pd.DataFrame:
    """Keep numeric percentage metrics with a documented unit and valid range."""
    metric = data["metric"].astype("string").str.lower().str.strip().str.replace(r"[^a-z0-9]", "", regex=True)
    metric = metric.replace({"f1score": "f1"})
    values = pd.to_numeric(data["value"], errors="coerce")
    units = data["unit"].astype("string").str.strip().str.lower() if "unit" in data else "fraction"
    if isinstance(units, str):
        units = pd.Series(units, index=data.index)
    units = units.fillna("fraction").replace("", "fraction")
    percent = values.where(units == "percent", values * 100)
    valid = metric.isin(METRICS) & units.isin(("percent", "fraction")) & percent.between(0, 100)
    if model_column is not None:
        valid &= data[model_column].notna()
    return pd.DataFrame({
        "Source": LOCAL,
        "Context": context,
        "Model": data[model_column] if model_column is not None else "Local evaluation",
        "Metric": metric.map(METRICS),
        "Percent": percent,
    }, index=data.index).loc[valid]


def latest_training_accuracy(directory) -> pd.DataFrame:
    """Read held-out accuracy from the newest completed dashboard training run."""
    for path in sorted((directory / "training_runs").glob("*/training.json"), reverse=True):
        run = read_json(path)
        test = run.get("test")
        configuration = run.get("configuration")
        if run.get("status") != "Completed" or not isinstance(test, dict):
            continue
        model = configuration.get("Model") if isinstance(configuration, dict) else None
        data = pd.DataFrame([{"model": model or "Training run", "metric": "accuracy",
                              "value": test.get("accuracy")}])
        scores = local_percentages(data, f"Held-out test · {path.parent.name}", "model")
        if not scores.empty:
            return scores
    return pd.DataFrame()

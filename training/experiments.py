"""Run local holdout ablations, five-fold CV, or external-checkpoint evaluation.

Run from the repository root with python -m training.experiments --help.
Each protocol writes an independent result bundle. Only completed bundles
are presented by the dashboard.
"""

import argparse
import csv
import json
import random
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import numpy as np
from sklearn.model_selection import StratifiedKFold

from config import RESULTS
from models.inference import model_settings, validate_model_shape
from models.keras_io import load_classifier
from training.run import BATCH_SIZE, FULL_EPOCHS, MODEL_NAMES, train
from utils.dataset import CLASSES, SEED, Sample, atomic_write, check_split_leakage, scan, stratify, unique_samples
from utils.evaluation import save_evaluation, score_samples, write_csv

FOLD_COUNT = 5


def five_fold_splits(dataset, seed=SEED):
    """Generate outer stratified test folds with disjoint inner validation splits."""
    samples = unique_samples(dataset)
    counts = Counter(sample.label for sample in samples)
    if any(counts[label] < FOLD_COUNT for label in CLASSES):
        raise ValueError("Five-fold evaluation requires at least five eligible images per class.")
    splitter = StratifiedKFold(n_splits=FOLD_COUNT, shuffle=True, random_state=seed)
    tested = set()
    all_folds = []
    for fold, (development, test) in enumerate(splitter.split(samples, [sample.label for sample in samples]), 1):
        rng = random.Random(seed + fold)
        splits = {"train": [], "validation": [], "test": [samples[index] for index in test]}
        for label in CLASSES:
            group = [samples[index] for index in development if samples[index].label == label]
            rng.shuffle(group)
            size = max(1, round(len(group) * 0.15))
            splits["validation"].extend(group[:size])
            splits["train"].extend(group[size:])
        check_split_leakage(splits)
        digests = {sample.digest for sample in splits["test"]}
        if tested & digests:
            raise ValueError("Test images overlap between outer folds.")
        tested.update(digests)
        all_folds.append(splits)
    if tested != {sample.digest for sample in samples}:
        raise ValueError("Outer test folds must cover every eligible image exactly once.")
    return all_folds


def read_rows(path):
    """Read one of the trainer's CSV artifacts as dictionaries."""
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def run_fold_evaluation(dataset, directory, epochs, architecture, loss):
    """Fit a fresh model per fold and summarize out-of-fold predictions."""
    fold_rows, predictions, complexity = [], [], []
    for fold, splits in enumerate(five_fold_splits(dataset), 1):
        print(f"Fold {fold}/{FOLD_COUNT}", flush=True)
        run = train(dataset, splits, epochs, directory, report_epoch, architecture=architecture,
                    loss_name=loss, protocol=f"cross-validation fold {fold}")
        fold_rows.extend((fold, row["metric"], row["value"]) for row in read_rows(run / "metrics.csv"))
        predictions.extend(read_rows(run / "predictions.csv"))
        complexity.extend((f"Fold {fold} · {row['model']}", row["training_seconds"],
                           row["parameters"], row["file_mb"]) for row in read_rows(run / "complexity.csv"))
    samples = [Sample(row["path"], row["actual"], row["digest"]) for row in predictions]
    check_split_leakage({"out-of-fold": samples})
    labels = [CLASSES.index(row["actual"]) for row in predictions]
    scores = np.array([[float(row[name]) for name in CLASSES] for row in predictions])
    save_evaluation(directory, samples, labels, scores)
    write_csv(directory / "folds.csv", ("fold", "metric", "value"), fold_rows)
    write_csv(directory / "complexity.csv", ("model", "training_seconds", "parameters", "file_mb"), complexity)


def run_comparison(dataset, directory, epochs, mode):
    """Run matched holdout trials with a common split and seed."""
    splits = stratify(dataset)
    if mode == "loss":
        comparison = "Cross-Entropy vs Hybrid Loss"
        cases = [("Categorical cross-entropy", "hybrid", "cross_entropy"),
                 ("Hybrid CE + focal loss", "hybrid", "hybrid")]
    elif mode == "stem":
        comparison = "With STEM vs without STEM"
        cases = [(MODEL_NAMES[key], key, "cross_entropy") for key in ("modified", "modified_no_stem")]
    else:
        comparison = "Architecture ablation"
        cases = [(name, model, "cross_entropy") for model, name in MODEL_NAMES.items() if model != "modified_no_stem"]
    metrics, complexity = [], []
    for variant, architecture, loss in cases:
        print(variant, flush=True)
        run = train(dataset, splits, epochs, directory, report_epoch, architecture=architecture, loss_name=loss)
        metrics.extend((comparison, variant, row["metric"], row["value"]) for row in read_rows(run / "metrics.csv"))
        complexity.extend((variant, row["training_seconds"], row["parameters"], row["file_mb"])
                          for row in read_rows(run / "complexity.csv"))
    write_csv(directory / "comparisons.csv", ("comparison", "variant", "metric", "value"), metrics)
    write_csv(directory / "complexity.csv", ("model", "training_seconds", "parameters", "file_mb"), complexity)


def run_external(dataset, source_run, directory):
    """Score a second compatible dataset after excluding source dataset overlaps."""
    state = json.loads((source_run / "training.json").read_text(encoding="utf-8"))
    if state.get("status") != "Completed":
        raise ValueError("External evaluation requires a completed source training run.")
    source_splits = json.loads((source_run / "splits.json").read_text(encoding="utf-8"))
    if dataset.root == Path(source_splits["root"]).resolve():
        raise ValueError("Choose a second dataset rather than the source training root.")
    source_digests = {record["digest"] for part in source_splits["splits"].values() for record in part}
    manifest = json.loads((source_run / "inference.json").read_text(encoding="utf-8"))
    classes, size, scale = model_settings(manifest)
    if set(classes) != set(CLASSES):
        raise ValueError("The checkpoint class mapping must match the four cotton leaf classes.")
    eligible = unique_samples(dataset)
    samples = [sample for sample in eligible if sample.digest not in source_digests]
    if {sample.label for sample in samples} != set(CLASSES):
        raise ValueError("The second dataset must retain all four classes after overlap removal.")
    checkpoint = source_run / "classifier.keras"
    model = load_classifier(checkpoint)
    validate_model_shape(model, size)
    labels, scores = score_samples(model, dataset.root, samples, size, BATCH_SIZE, classes, scale)
    save_evaluation(directory, samples, labels, scores, classes)
    write_csv(directory / "complexity.csv", ("model", "training_seconds", "parameters", "file_mb"),
              [(manifest.get("model_name", model.name), None, model.count_params(), checkpoint.stat().st_size / 1_000_000)])
    return {"excluded_source_duplicates": len(eligible) - len(samples), "scored_images": len(samples)}


def report_epoch(number, epochs, message):
    """Report actual Keras epoch metrics to the terminal."""
    print(message, flush=True)


def main(argv=None):
    """Parse and execute one local experiment protocol."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("cv", "ablation", "stem", "loss", "external"))
    parser.add_argument("dataset", type=Path, help="Local four-class dataset directory")
    parser.add_argument("--epochs", type=int, default=FULL_EPOCHS)
    parser.add_argument("--model", choices=MODEL_NAMES, default="baseline", help="Architecture for cv")
    parser.add_argument("--loss", choices=("cross_entropy", "hybrid"), default="cross_entropy", help="Loss for cv")
    parser.add_argument("--run", type=Path, help="Completed source training run directory for external mode")
    parser.add_argument("--results", type=Path, default=RESULTS)
    args = parser.parse_args(argv)
    if args.epochs < 1 or (args.mode == "external" and args.run is None):
        parser.error("Use positive epochs and supply --run for external evaluation.")
    dataset = scan(args.dataset)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
    directory = args.results / "evaluations" / f"{stamp}-{args.mode}-{uuid4().hex[:8]}"
    state_file = directory / "experiment.json"
    state = {"status": "Running", "protocol": args.mode, "dataset": str(dataset.root), "seed": SEED}
    if args.mode != "external":
        state["epochs"] = args.epochs
    if args.mode == "cv":
        state.update(model=args.model, loss=args.loss)
    if args.run is not None:
        state["source_run"] = str(args.run.resolve())

    def save_state():
        atomic_write(state_file, lambda handle: json.dump(state, handle, indent=2))

    save_state()
    try:
        if args.mode == "cv":
            run_fold_evaluation(dataset, directory, args.epochs, args.model, args.loss)
        elif args.mode == "external":
            state.update(run_external(dataset, args.run, directory))
        else:
            run_comparison(dataset, directory, args.epochs, args.mode)
    except Exception as error:
        state.update(status="Failed", error=str(error))
        save_state()
        raise
    state["status"] = "Completed"
    save_state()
    print(f"Saved evaluation to {directory}")


if __name__ == "__main__":
    main()

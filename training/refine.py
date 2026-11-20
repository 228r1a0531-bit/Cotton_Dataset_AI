"""Audit or fine-tune a saved hybrid on its original verified split."""

import argparse
import json
from collections import Counter
from pathlib import Path
from uuid import uuid4

from config import RESULTS
from models.inference import model_settings, validate_model_shape
from models.keras_io import load_classifier
from training.run import BATCH_SIZE, INPUT_SIZE, train
from utils.dataset import CLASSES, SPLITS, Sample, atomic_write, check_split_leakage, scan
from utils.evaluation import measure, save_evaluation, score_samples, write_csv


def load_verified_split(dataset, source):
    """Require the exact original images, class labels and decoded pixel hashes."""
    snapshot = json.loads((source / "splits.json").read_text(encoding="utf-8"))
    saved = snapshot["splits"]
    if set(saved) != set(SPLITS):
        raise ValueError("The original run must contain train, validation, and test splits.")
    splits = {name: [Sample(**row) for row in saved[name]] for name in SPLITS}
    current = set(dataset.images)
    check_split_leakage(splits)
    for name, samples in splits.items():
        if {sample.label for sample in samples} != set(CLASSES):
            raise ValueError(f"The original {name} split must represent all four classes.")
        for sample in samples:
            if sample not in current:
                raise ValueError(f"Original {name} image is missing or differs: {sample.path}")
    return splits, snapshot.get("seed", 42)


def run(dataset, source, results, epochs, *, audit_only=False):
    """Save a fresh audit bundle and optionally fit a candidate without altering the incumbent."""
    source = source.resolve()
    config = json.loads((source / "training.json").read_text(encoding="utf-8"))
    if config.get("status") != "Completed" or config["configuration"].get("Model") != "Hybrid feature fusion":
        raise ValueError("Select a completed Full Hybrid run.")
    manifest = json.loads((source / "inference.json").read_text(encoding="utf-8"))
    labels, size, scale = model_settings(manifest)
    if tuple(labels) != CLASSES or size != INPUT_SIZE[:2][::-1] or scale != "0_255":
        raise ValueError("The original manifest must match the trainer's class order and input contract.")
    splits, seed = load_verified_split(dataset, source)
    checkpoint = source / "classifier.keras"
    incumbent = load_classifier(checkpoint)
    validate_model_shape(incumbent, size)
    for layer in ("cn2_se_features", "modified_inception_features", "feature_fusion", "fusion_dropout",
                  "classifier_dense", "classifier_dropout"):
        incumbent.get_layer(layer)

    directory = results / "evaluations" / f"{source.name}-{'audit' if audit_only else 'refine'}-{uuid4().hex[:8]}"
    state = {"status": "Running", "protocol": "hybrid_audit" if audit_only else "hybrid_refinement",
             "source_run": str(source), "seed": seed}

    def save_state():
        atomic_write(directory / "experiment.json", lambda handle: json.dump(state, handle, indent=2))

    def score(model, name, report=None):
        samples = splits[name]
        truth, probabilities = score_samples(model, dataset.root, samples, size, BATCH_SIZE)
        if report is not None:
            return save_evaluation(report, samples, truth, probabilities).summary
        return measure(truth, probabilities).summary

    save_state()
    try:
        support = {name: Counter(sample.label for sample in splits[name]) for name in SPLITS}
        write_csv(directory / "split_counts.csv", ("class", *SPLITS),
                  ((label, *(support[name][label] for name in SPLITS)) for label in CLASSES))
        before = score(incumbent, "test", directory if audit_only else directory / "before")
        train_accuracy = score(incumbent, "train")["accuracy"]
        validation_accuracy = score(incumbent, "validation")["accuracy"]
        state.update(source_clean_train_accuracy=train_accuracy, source_validation_accuracy=validation_accuracy,
                     source_test_accuracy=before["accuracy"])
        logged_test = config.get("test")
        if isinstance(logged_test, dict) and isinstance(logged_test.get("accuracy"), (float, int)):
            state["source_logged_test_accuracy"] = logged_test["accuracy"]
            if abs(logged_test["accuracy"] - before["accuracy"]) > 1e-6:
                print("The saved checkpoint's recomputed test accuracy differs from the historical training log.")
        print(f"Original checkpoint inference-mode train accuracy: {train_accuracy:.2%}")
        print(f"Original checkpoint inference-mode validation accuracy: {validation_accuracy:.2%}")
        if not audit_only:
            loss_name = config["configuration"].get("Loss", "hybrid")
            candidate = train(
                dataset, splits, epochs, directory, lambda _, __, message: print(message, flush=True),
                architecture="hybrid", loss_name=loss_name, loss_options=config.get("loss_options"),
                seed=seed, protocol="hybrid refinement on original holdout", warm_start=checkpoint,
            )
            candidate_model = load_classifier(candidate / "classifier.keras")
            candidate_validation = score(candidate_model, "validation")["accuracy"]
            after = score(candidate_model, "test", directory)
            state.update(candidate_run=str(candidate.relative_to(directory)),
                         candidate_validation_accuracy=candidate_validation, candidate_test_accuracy=after["accuracy"],
                         prefer_candidate=candidate_validation > validation_accuracy)
            metrics = [(label, metric, value) for label, summary in
                       (("Original Hybrid", before), ("Fine-tuned Hybrid", after))
                       for metric, value in summary.items() if metric != "macro_auc"]
            write_csv(directory / "comparisons.csv", ("comparison", "variant", "metric", "value"),
                      (("Hybrid before vs after fine-tuning", label, metric, value) for label, metric, value in metrics))
    except Exception as error:
        state.update(status="Failed", error=str(error))
        save_state()
        raise
    state["status"] = "Completed"
    save_state()
    print(f"Saved reports to {directory}")
    return directory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("source_run", type=Path)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs must be positive.")
    run(scan(args.dataset), args.source_run, RESULTS, args.epochs, audit_only=args.audit_only)


if __name__ == "__main__":
    main()

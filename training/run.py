"""Train a baseline or hybrid model on verified, stratified local images.

The pipeline decodes images with inference's preprocessing, fits on the
training split, chooses weights by validation loss, and evaluates test data
once. Each run saves its own configuration, class order, history and model.
"""

import csv
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from uuid import uuid4

from utils.dataset import CLASSES, Scan, Sample, SEED, atomic_write, check_split_leakage, save_splits
from utils.evaluation import save_evaluation, score_samples, write_csv
from utils.images import model_pixels

INPUT_SIZE = (224, 224, 3)
BATCH_SIZE = 32
LEARNING_RATE = 0.0001
FULL_EPOCHS = 50
DROPOUT = 0.5
EARLY_STOP_PATIENCE = 7
LR_PATIENCE = 3
LR_FACTOR = 0.5
MIN_LEARNING_RATE = 1e-6
MODEL_NAMES = {
    "baseline": "Baseline CNN",
    "cn2_se": "CNN+SE",
    "inception": "InceptionV3",
    "modified": "Modified InceptionV3",
    "modified_no_stem": "Modified InceptionV3 without STEM",
    "hybrid": "Hybrid feature fusion",
}


def settings(epochs: int = FULL_EPOCHS, architecture="baseline", loss_name="cross_entropy") -> dict:
    """Describe the selected architecture and optimizer settings for the dashboard."""
    return {
        "Model": MODEL_NAMES[architecture],
        "Input size": "224 × 224 × 3",
        "Batch size": BATCH_SIZE,
        "Optimizer": "Adam",
        "Learning rate": LEARNING_RATE,
        "Epochs": epochs,
        "Dropout": DROPOUT,
        "Loss": loss_name,
        "Early stopping patience": EARLY_STOP_PATIENCE,
        "Learning rate reduction patience": LR_PATIENCE,
        "Learning rate reduction factor": LR_FACTOR,
        "Minimum learning rate": MIN_LEARNING_RATE,
    }


def train(dataset: Scan, splits: dict[str, list[Sample]], epochs: int, results: Path, on_epoch,
          *, architecture="baseline", loss_name="cross_entropy", loss_options=None,
          seed=SEED, protocol="holdout", warm_start: Path | None = None) -> Path:
    """Fit train samples, monitor validation samples, and evaluate the held-out test split."""
    configuration = settings(epochs, architecture, loss_name)
    configuration["Seed"] = seed
    if warm_start is not None:
        if architecture != "hybrid":
            raise ValueError("Checkpoint fine-tuning is supported for the existing hybrid architecture.")
        configuration.update({
            "Learning rate": 3e-5, "Dropout": 0.25, "Initialization": str(warm_start),
            "Augmentation": "Train-only flips and small rotations",
            "Class weighting": "Normalized inverse square root of training frequency",
        })
    check_split_leakage(splits)
    loss_options = {} if loss_options is None else dict(loss_options)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
    run = results / "training_runs" / f"{stamp}-{uuid4().hex[:8]}"
    run.mkdir(parents=True)
    state = {"status": "Training", "architecture": architecture, "configuration": configuration, "completed": 0,
             "loss_options": loss_options, "class_indices": dict(zip(CLASSES, range(len(CLASSES))))}
    history = []

    def save_state():
        atomic_write(run / "training.json", lambda handle: json.dump(state, handle, indent=2))

    def log(message):
        with (run / "training.log").open("a", encoding="utf-8") as handle:
            handle.write(message + "\n")

    save_state()
    log(f"Started {MODEL_NAMES[architecture]} training for up to {epochs} epoch(s).")
    try:
        import numpy as np
        import tensorflow as tf
        from tensorflow import keras

        from models.cnn import build_cnn
        from models.hybrid import build_hybrid
        from models.inception import build_inception_v3, build_modified_inception
        from models.keras_io import load_classifier
        from models.losses import build_loss

        save_splits(run, dataset, splits, seed=seed, protocol=protocol)
        keras.utils.set_random_seed(seed)
        targets = dict(zip(CLASSES, np.eye(len(CLASSES), dtype=np.float32)))
        if warm_start is not None:
            counts = Counter(item.label for item in splits["train"])
            if any(counts[label] == 0 for label in CLASSES):
                raise ValueError("Checkpoint fine-tuning requires training images for every class.")
            weights = np.array([1 / np.sqrt(counts[label]) for label in CLASSES])
            mean_weight = np.average(weights, weights=[counts[label] for label in CLASSES])
            weights = tf.constant(weights / mean_weight, dtype=tf.float32)
            state["class_weights"] = dict(zip(CLASSES, weights.numpy().tolist()))
            augment = keras.Sequential([
                keras.layers.RandomFlip("horizontal_and_vertical", seed=seed),
                keras.layers.RandomRotation(0.04, fill_mode="reflect", seed=seed + 1),
            ])

            def augment_training(pixels, labels):
                return augment(pixels, training=True), labels, tf.gather(weights, tf.argmax(labels, axis=1))

        def batches(samples: list[Sample], shuffle: bool):
            def read_images():
                for item in samples:
                    pixels = model_pixels(dataset.root / item.path, INPUT_SIZE[:2][::-1])
                    yield pixels, targets[item.label]

            data = tf.data.Dataset.from_generator(
                read_images,
                output_signature=(tf.TensorSpec(INPUT_SIZE, tf.float32),
                                  tf.TensorSpec((len(CLASSES),), tf.float32)),
            )
            data = data.apply(tf.data.experimental.assert_cardinality(len(samples)))
            if shuffle:
                data = data.shuffle(min(len(samples), BATCH_SIZE * 2), seed=seed)
            options = tf.data.Options()
            options.threading.private_threadpool_size = 1
            data = data.with_options(options).batch(BATCH_SIZE)
            if warm_start is not None and shuffle:
                data = data.map(augment_training)
            return data.prefetch(1)

        class EpochRecorder(keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                metrics = logs or {}
                number = epoch + 1
                for split, prefix in (("train", ""), ("validation", "val_")):
                    history.extend((number, split, metric, float(metrics[prefix + metric]))
                                   for metric in ("accuracy", "loss"))

                def write_history(handle):
                    writer = csv.writer(handle)
                    writer.writerow(("epoch", "split", "metric", "value"))
                    writer.writerows(history)

                atomic_write(run / "history.csv", write_history)
                summary = f"Epoch {number}/{epochs} · " + " · ".join(
                    f"{key}={value:.4f}" for key, value in metrics.items()
                )
                log(summary)
                state["completed"] = number
                save_state()
                on_epoch(number, epochs, summary)

        builder = {
            "baseline": build_cnn,
            "cn2_se": lambda shape, **kwargs: build_cnn(shape, use_se=True, **kwargs),
            "inception": build_inception_v3,
            "modified": build_modified_inception,
            "modified_no_stem": lambda shape, **kwargs: build_modified_inception(shape, use_stem=False, **kwargs),
            "hybrid": build_hybrid,
        }[architecture]
        if warm_start is None:
            model = builder(INPUT_SIZE, dropout=DROPOUT)
        else:
            model = load_classifier(warm_start)
            for layer_name in ("fusion_dropout", "classifier_dropout"):
                model.get_layer(layer_name).rate = configuration["Dropout"]
        loss = build_loss(loss_name, **loss_options)
        state["loss_configuration"] = loss.get_config()
        save_state()
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=configuration["Learning rate"]),
            loss=loss,
            metrics=[keras.metrics.CategoricalAccuracy(name="accuracy")],
        )
        early_stop = keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=EARLY_STOP_PATIENCE, restore_best_weights=True
        )
        reduce_lr = keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=LR_FACTOR, patience=LR_PATIENCE, min_lr=MIN_LEARNING_RATE
        )
        training_batches = batches(splits["train"], shuffle=True)
        validation_batches = batches(splits["validation"], shuffle=False)
        started = perf_counter()
        model.fit(training_batches, validation_data=validation_batches,
                  epochs=epochs, shuffle=False, callbacks=[early_stop, reduce_lr, EpochRecorder()], verbose=0)
        seconds = perf_counter() - started
        state["best_epoch"] = early_stop.best_epoch + 1
        test = splits["test"]
        labels, scores = score_samples(model, dataset.root, test, INPUT_SIZE[:2][::-1], BATCH_SIZE)
        evaluation = save_evaluation(run, test, labels, scores)
        state["test"] = {"accuracy": evaluation.summary["accuracy"],
                         "loss": float(loss(np.eye(len(CLASSES), dtype=np.float32)[labels], scores))}
        checkpoint = run / "classifier.keras"
        model.save(checkpoint)
        complexity = (MODEL_NAMES[architecture], seconds, model.count_params(),
                      checkpoint.stat().st_size / 1_000_000)
        write_csv(run / "complexity.csv", ("model", "training_seconds", "parameters", "file_mb"), [complexity])
        manifest = {
            "classes": list(CLASSES),
            "image_size": list(INPUT_SIZE[:2][::-1]),
            "pixel_scale": "0_255",
            "model_name": MODEL_NAMES[architecture],
        }
        atomic_write(run / "inference.json", lambda handle: json.dump(manifest, handle, indent=2))
    except Exception as error:
        state["status"] = "Failed"
        log(f"Training failed: {error}")
        save_state()
        raise
    else:
        state["status"] = "Completed"
        log("Training and held-out test evaluation completed.")
        save_state()
    return run

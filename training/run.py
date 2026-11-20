"""Train a small baseline CNN on verified, stratified local images."""

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from utils.dataset import CLASSES, Scan, Sample, SEED, atomic_write, save_splits

INPUT_SIZE = (224, 224, 3)
BATCH_SIZE = 32
LEARNING_RATE = 0.0001
FULL_EPOCHS = 50
DROPOUT = 0.5


def settings(epochs: int = FULL_EPOCHS) -> dict:
    return {
        "Model": "Baseline CNN",
        "Input size": "224 × 224 × 3",
        "Batch size": BATCH_SIZE,
        "Optimizer": "Adam",
        "Learning rate": LEARNING_RATE,
        "Epochs": epochs,
        "Dropout": DROPOUT,
    }


def train(dataset: Scan, splits: dict[str, list[Sample]], epochs: int, results: Path, on_epoch) -> Path:
    """Fit train samples, monitor validation samples, and evaluate the held-out test split."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
    run = results / "training_runs" / f"{stamp}-{uuid4().hex[:8]}"
    run.mkdir(parents=True)
    state = {"status": "Training", "configuration": settings(epochs), "completed": 0}
    history = []

    def save_state():
        atomic_write(run / "training.json", lambda handle: json.dump(state, handle, indent=2))

    def log(message):
        with (run / "training.log").open("a", encoding="utf-8") as handle:
            handle.write(message + "\n")

    save_state()
    log(f"Started baseline CNN training for {epochs} epoch(s).")
    try:
        import numpy as np
        import tensorflow as tf
        from PIL import Image, ImageOps
        from tensorflow import keras

        save_splits(run, dataset, splits)
        keras.utils.set_random_seed(SEED)

        def batches(samples: list[Sample], shuffle: bool):
            def read_images():
                for item in samples:
                    with Image.open(dataset.root / item.path) as image:
                        rgb = ImageOps.exif_transpose(image).convert("RGB")
                        resized = rgb.resize(INPUT_SIZE[:2][::-1], Image.Resampling.BILINEAR)
                        yield np.asarray(resized, dtype=np.float32), CLASSES.index(item.label)

            data = tf.data.Dataset.from_generator(
                read_images,
                output_signature=(tf.TensorSpec(INPUT_SIZE, tf.float32), tf.TensorSpec((), tf.int32)),
            )
            data = data.apply(tf.data.experimental.assert_cardinality(len(samples)))
            if shuffle:
                data = data.shuffle(min(len(samples), BATCH_SIZE * 2), seed=SEED)
            options = tf.data.Options()
            options.threading.private_threadpool_size = 1
            return data.with_options(options).batch(BATCH_SIZE).prefetch(1)

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

        model = keras.Sequential([
            keras.Input(INPUT_SIZE),
            keras.layers.Rescaling(1 / 255),
            keras.layers.Conv2D(16, 3, strides=2, activation="relu"),
            keras.layers.MaxPooling2D(),
            keras.layers.Conv2D(32, 3, activation="relu"),
            keras.layers.MaxPooling2D(),
            keras.layers.Conv2D(64, 3, activation="relu"),
            keras.layers.GlobalAveragePooling2D(),
            keras.layers.Dropout(DROPOUT),
            keras.layers.Dense(len(CLASSES), activation="softmax"),
        ])
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        model.fit(batches(splits["train"], shuffle=True),
                  validation_data=batches(splits["validation"], shuffle=False),
                  epochs=epochs, shuffle=False, callbacks=[EpochRecorder()], verbose=0)
        test = model.evaluate(batches(splits["test"], shuffle=False), return_dict=True, verbose=0)
        state["test"] = {key: float(value) for key, value in test.items()}
        model.save(run / "baseline.keras")
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

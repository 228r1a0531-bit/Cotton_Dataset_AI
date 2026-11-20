"""Deterministic presentation-only metrics for the 50-epoch research target."""

from math import exp

PREVIEW_EPOCHS = 50
PREVIEW_SECONDS = 40
RESEARCH_TARGET = 0.9941
PREVIEW_LABEL = "Simulated prototype training preview"


def preview_epoch(elapsed_seconds: float) -> int:
    """Map replay time to the revealed epoch without requiring a dataset."""
    return min(PREVIEW_EPOCHS, 1 + int(max(0, elapsed_seconds) * (PREVIEW_EPOCHS - 1) / PREVIEW_SECONDS))


def preview_history() -> list[dict]:
    """Return illustrative values only. The validation curve ends at the research target."""
    rows = []
    for epoch in range(1, PREVIEW_EPOCHS + 1):
        fraction = (epoch - 1) / (PREVIEW_EPOCHS - 1)
        growth = (1 - exp(-4.1 * fraction)) / (1 - exp(-4.1))
        for split, start, end, loss_start, loss_end in (
            ("train", 0.36, 0.997, 1.46, 0.031),
            ("validation", 0.31, RESEARCH_TARGET, 1.53, 0.058),
        ):
            rows.extend((
                {"epoch": epoch, "split": split, "metric": "accuracy",
                 "value": end if epoch == PREVIEW_EPOCHS else start + (end - start) * growth},
                {"epoch": epoch, "split": split, "metric": "loss",
                 "value": loss_end if epoch == PREVIEW_EPOCHS else loss_start + (loss_end - loss_start) * growth},
            ))
    return rows

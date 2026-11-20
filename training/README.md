# Training

Install `requirements-ml.txt` and open **Dataset & Training** in the Streamlit
dashboard. Fast Demo fits the compact baseline CNN for a small configurable
number of epochs. Full hybrid training fits the CN2-SE and modified Inception
fusion model for up to 50 epochs. Use suitable GPU resources and a verified
dataset for a full hybrid experiment.

The data pipeline loads the saved split's images as EXIF-oriented RGB, resizes
them to 224 × 224 using bilinear sampling, and passes float32 pixels to the
model. The model applies its own pixel scaling. Only training is shuffled.
Labels use the saved four-class order and are encoded as one-hot vectors.
Choose categorical cross-entropy, focal loss, or weighted CE + focal loss.

Adam starts at 0.0001. The learning rate can drop on a validation loss plateau.
Early stopping restores the weights from the best validation epoch. The
trainer evaluates the held-out test split once and saves the resulting
checkpoint, configuration, class mapping, and history. Inference automatically
finds the newest completed run unless a manual checkpoint is configured.

See the repository README for local dataset setup and [the artifact contract](../results/README.md)
for the run outputs.

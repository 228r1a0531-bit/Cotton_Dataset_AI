"""Preview an uploaded image and optionally classify it with a local checkpoint."""

import pandas as pd
import plotly.express as px
import streamlit as st
from PIL import Image

from config import IMAGE_TYPES, RESULTS, ROOT
from models.inference import available_checkpoints, model_settings, predict, validate_model_shape
from models.keras_io import load_classifier
from utils.dashboard import PALETTE, chart, empty, heading, read_json
from utils.images import load_preview

heading("Upload & Classification", "Inspect a cotton leaf image and run inference when a trained model is available.")
checkpoints = available_checkpoints(ROOT, RESULTS)
candidate = checkpoints[0] if checkpoints else None
if len(checkpoints) > 1:
    names = {}
    for paths in checkpoints:
        checkpoint, manifest = paths
        model_name = read_json(manifest).get("model_name", "Classifier")
        names[f"{model_name} · {checkpoint.parent.name}"] = paths
    candidate = names[st.selectbox("Saved classifier", names)]


@st.cache_resource
def classifier(path, modified_ns):
    return load_classifier(path)


upload = st.file_uploader("Upload a JPEG or PNG image", type=IMAGE_TYPES)
preview, result = st.columns(2)
with preview:
    st.subheader("Leaf preview")
    image = None
    if upload is None:
        empty("Upload a leaf photo to see the preview.")
    else:
        try:
            image = load_preview(upload)
            st.image(image, width="stretch")
        except (OSError, Image.DecompressionBombError, ValueError):
            st.error("Could not open the image. Try a smaller, valid JPEG or PNG.")

with result:
    st.subheader("Classification result")
    manifest = {}
    contract = None
    model = None
    if candidate is None:
        empty("Inference unavailable. A trained checkpoint and its inference configuration are required.")
        st.caption("Complete a training run or configure a trusted checkpoint using the instructions in results/README.md.")
    else:
        checkpoint, manifest_path = candidate
        manifest = read_json(manifest_path)
        try:
            contract = model_settings(manifest)
        except ValueError as error:
            st.error(f"Incompatible inference configuration. {error}")
        else:
            st.caption(f"Model: {manifest.get('model_name') or checkpoint.stem}")
    if contract is not None and image is None:
        empty("Upload a valid image to run inference.")
    if contract is not None and image is not None:
        labels, size, scale = contract
        try:
            loaded = classifier(checkpoint, checkpoint.stat().st_mtime_ns)
            validate_model_shape(loaded, size)
        except ImportError:
            st.error("Inference needs the optional dependencies in requirements-ml.txt.")
        except (OSError, ValueError, TypeError, RuntimeError, Image.DecompressionBombError) as error:
            st.error(f"Could not use this checkpoint. {error}")
        else:
            model = loaded
    if st.button("Classify image", type="primary", disabled=model is None) and model is not None:
        try:
            with st.spinner("Running inference…"):
                scores = predict(model, upload, size, scale)
        except (OSError, ValueError, TypeError, RuntimeError, Image.DecompressionBombError) as error:
            st.error(f"Inference failed. {error}")
        else:
            winner = scores.argmax()
            st.success("Inference complete.")
            st.caption(f"Checkpoint model: {manifest.get('model_name') or model.name}")
            st.metric("Predicted class", labels[winner])
            probabilities = pd.DataFrame({"Class": labels, "Probability": scores})
            figure = px.bar(probabilities, x="Probability", y="Class", orientation="h",
                            range_x=[0, 1], color_discrete_sequence=PALETTE)
            figure.update_xaxes(tickformat=".0%")
            chart(figure)

st.subheader("Grad-CAM")
empty("Unavailable. Grad-CAM has not been implemented.")

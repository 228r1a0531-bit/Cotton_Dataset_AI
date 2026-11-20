"""Preview an uploaded image and optionally classify it with a local checkpoint."""

from io import BytesIO

import pandas as pd
import plotly.express as px
import streamlit as st
from PIL import Image

from config import IMAGE_TYPES, RESULTS, ROOT
from models.inference import CHECKPOINT_NAME, model_settings, predict
from models.keras_io import load_classifier
from utils.dashboard import PALETTE, chart, empty, heading, read_json
from utils.images import load_preview

heading("Upload & Classification", "Inspect a cotton leaf image and run inference when a trained model is available.")
checkpoint = ROOT / CHECKPOINT_NAME
manifest_path = RESULTS / "inference.json"


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
    if not checkpoint.is_file() or not manifest_path.is_file():
        empty("Inference unavailable. A trained checkpoint and its inference configuration are required.")
        st.caption("Class probabilities will appear after a successful inference run.")
    elif image is None:
        empty("Upload a valid image to run inference.")
    else:
        try:
            labels, size, scale = model_settings(read_json(manifest_path))
            with st.spinner("Running inference…"):
                model = classifier(checkpoint, checkpoint.stat().st_mtime_ns)
                scores = predict(model, BytesIO(upload.getvalue()), size, scale)
        except ImportError:
            st.error("Inference needs the optional dependencies in requirements-ml.txt.")
        except (OSError, ValueError, TypeError, RuntimeError, Image.DecompressionBombError) as error:
            st.error(f"Inference failed. {error}")
        else:
            winner = scores.argmax()
            st.success("Inference complete.")
            st.metric("Predicted class", labels[winner])
            probabilities = pd.DataFrame({"Class": labels, "Probability": scores})
            figure = px.bar(probabilities, x="Probability", y="Class", orientation="h",
                            range_x=[0, 1], color_discrete_sequence=PALETTE)
            figure.update_xaxes(tickformat=".0%")
            chart(figure)

st.subheader("Grad-CAM")
empty("Unavailable. Grad-CAM has not been implemented.")

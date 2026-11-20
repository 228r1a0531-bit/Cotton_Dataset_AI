"""Describe the research demo and its current capabilities."""

import streamlit as st

from config import APP_TITLE

st.title(APP_TITLE)
st.subheader("Hybrid deep learning for cotton leaf disease classification")
st.write(
    "Upload a cotton leaf photo to preview the image and inspect its RGB "
    "pixel distribution. The demo runs without model weights or training."
)
st.page_link("pages/inspect_image.py", label="Inspect a leaf image")

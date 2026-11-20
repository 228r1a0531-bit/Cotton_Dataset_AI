"""Preview an uploaded cotton leaf image and plot its RGB distribution."""

import plotly.express as px
import streamlit as st
from PIL import Image

from config import IMAGE_TYPES
from utils.images import load_preview, rgb_histogram


def render() -> None:
    st.title("Inspect a leaf image")
    upload = st.file_uploader("Upload a JPEG or PNG image", type=IMAGE_TYPES)
    if upload is None:
        return

    try:
        preview = load_preview(upload)
    except (OSError, Image.DecompressionBombError):
        st.error("This image could not be opened. Try a smaller, valid JPEG or PNG.")
        return

    image_column, chart_column = st.columns(2)
    image_column.image(preview, caption="Leaf preview", width="stretch")
    histogram = rgb_histogram(preview).reset_index()
    chart = px.line(
        histogram,
        x="Intensity",
        y=["Red", "Green", "Blue"],
        title="RGB distribution of the preview",
        labels={"value": "Pixel count", "variable": "Channel"},
        color_discrete_map={"Red": "#d33b36", "Green": "#278750", "Blue": "#386bc5"},
    )
    chart_column.plotly_chart(chart, width="stretch")


render()

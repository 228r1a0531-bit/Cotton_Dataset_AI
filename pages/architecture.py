"""Explain the data flow and research-inspired model components."""

import streamlit as st

from utils.dashboard import heading

heading("Model Architecture", "How the implemented hybrid classifier transforms a leaf image into class probabilities.")
st.caption("The diagram summarizes the implemented research-inspired model.")
st.graphviz_chart("""
digraph Hybrid {
    graph [rankdir=TB, bgcolor="transparent"];
    node [shape=box, style="rounded,filled", fillcolor="#e8f3ec", color="#54856b",
          fontname="sans-serif", fontcolor="#173b2f", margin="0.18,0.10"];
    edge [color="#54856b"];
    Input [label="Cotton leaf image"];
    CNN [label="CN2-SE"];
    Inception [label="Modified InceptionV3"];
    Fusion [label="Feature fusion"];
    Classifier [label="MLP classifier"];
    Output [label="Four class scores"];
    Input -> {CNN Inception};
    {CNN Inception} -> Fusion;
    Fusion -> Classifier -> Output;
}
""")

components = [
    ("SE block", "Feature map → global pooling → channel gates → channel-wise rescaling",
     "Squeeze spatial dimensions into channel descriptors. Learned gates rescale each input channel."),
    ("STEM", "Input image → early convolutions → spatial reduction → feature map",
     "A strided 24-channel convolution, a 32-channel convolution, max pooling, and a 48-channel convolution form the modified stem."),
    ("CN2-SE", "Image → convolutional feature extractor → SE attention → feature vector",
     "The CNN branch uses channel attention to recalibrate its learned feature maps."),
    ("Modified InceptionV3", "Image → stem → parallel multi-scale operations → feature vector",
     "Two compact multi-scale blocks combine pointwise, 3×3, stacked 3×3, and pooled responses. The shorter stem and narrower blocks differ from standard Keras InceptionV3."),
    ("Feature fusion and MLP", "Branch feature vectors → fusion → dense classifier → four class scores",
     "The fusion stage combines representations before the MLP maps them to class scores."),
]
for name, flow, description in components:
    with st.container(border=True):
        st.subheader(name)
        st.markdown(f"**{flow}**")
        st.write(description)

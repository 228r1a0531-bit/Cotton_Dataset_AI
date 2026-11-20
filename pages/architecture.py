"""Present the hybrid, squeeze-and-excitation, and stem data flows."""

import streamlit as st

from utils.dashboard import heading


def diagram(name, body):
    """Render a vertical Graphviz flow with a shared presentation theme."""
    st.graphviz_chart(f"""
    digraph {name} {{
        graph [rankdir=TB, bgcolor="transparent"];
        node [shape=box, style="rounded,filled", fillcolor="#e8f3ec", color="#54856b",
              fontname="sans-serif", fontcolor="#173b2f", margin="0.18,0.10"];
        edge [color="#54856b", fontname="sans-serif"];
        {body}
    }}
    """)


def explain(name, description):
    """Show one component explanation alongside the flow."""
    with st.container(border=True):
        st.markdown(f"**{name}**")
        st.write(description)


heading("Model Architecture", "Follow the leaf image from preprocessing to the four-class prediction.")
hybrid, se, stem = st.tabs(["Hybrid feature fusion", "Squeeze-and-excitation", "Modified stem"])
with hybrid:
    diagram("Hybrid", """
        Image [label="Cotton image"];
        Prepare [label="Preprocessing"];
        CNN [label="CN2-SE branch"];
        Inception [label="Modified InceptionV3 branch"];
        Fuse [label="Concatenation"];
        MLP [label="MLP"];
        Output [label="Four-class softmax", fillcolor="#cce8d9"];
        Loss [label="Hybrid CE + focal loss", fillcolor="#f5ecd7"];
        Labels [label="Ground-truth label"];
        Image -> Prepare;
        Prepare -> {CNN Inception};
        {CNN Inception} -> Fuse;
        Fuse -> MLP -> Output;
        Output -> Loss [style=dashed, label="training only"];
        Labels -> Loss [style=dashed];
    """)
    explain("Preprocessing", "The pipeline corrects image orientation, converts to RGB, and resizes to 224 × 224. The model scales the pixels internally.")
    explain("CN2-SE and Modified InceptionV3", "The CN2-SE branch learns which channels to emphasize. The other branch processes several receptive-field sizes in parallel to capture both small spots and wider leaf patterns.")
    explain("Concatenation and MLP", "Concatenation joins the two pooled feature vectors. The dense layers learn how to combine them, and dropout regularizes training.")
    explain("Softmax and training loss", "Softmax converts the four logits to probabilities that sum to one. During training, optional hybrid loss compares those probabilities with the true label using cross-entropy and a focal term that emphasizes harder examples.")

with se:
    diagram("SqueezeExcitation", """
        Input [label="Input feature map"];
        Average [label="Global average pooling"];
        Relu [label="Dense / ReLU"];
        Sigmoid [label="Dense / sigmoid"];
        Scale [label="Channel scaling"];
        Output [label="Recalibrated output", fillcolor="#cce8d9"];
        Input -> Average -> Relu -> Sigmoid -> Scale -> Output;
        Input -> Scale [style=dashed, label="original feature map"];
    """)
    explain("Squeeze", "Global average pooling summarizes each feature channel with one number, giving the gate a view of the entire feature map.")
    explain("Excitation", "The first dense layer compresses the channel vector with ReLU. The second expands it and uses sigmoid to assign a weight between zero and one to each channel.")
    explain("Channel scaling", "Each gate multiplies every pixel of its original channel. The output keeps the original feature map dimensions.")

with stem:
    diagram("Stem", """
        Image [label="Image"];
        Conv [label="Convolution"];
        Activation [label="Activation"];
        Pool [label="Pooling"];
        Maps [label="Feature maps", fillcolor="#cce8d9"];
        Image -> Conv -> Activation -> Pool -> Maps;
    """)
    explain("Convolution and activation", "The opening convolution finds local patterns such as edges and changes in color. Batch normalization and ReLU help later layers build useful responses from them.")
    explain("Pooling and feature maps", "Max pooling reduces spatial resolution while keeping strong local responses. A final convolution refines the feature maps before the multi-scale blocks.")
    explain("What is modified", "The implementation uses a shorter 24/32/48-channel stem and two compact multi-scale blocks. This design is inspired by the research and differs from the standard Keras InceptionV3 topology.")

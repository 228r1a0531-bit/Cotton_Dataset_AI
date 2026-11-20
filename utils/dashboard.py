"""Shared presentation and local research artifact readers."""

import json
from html import escape

import pandas as pd
import streamlit as st

from config import RESULTS

PALETTE = ["#267956", "#5277a1", "#b07834", "#8a659e"]


def apply_style() -> None:
    st.markdown(
        """
        <style>
        .stMainBlockContainer {max-width: 74rem; padding-top: 4rem;}
        .research-banner {
            background: linear-gradient(110deg, #12392d, #245d46);
            color: #eaf8ef; border-radius: 1rem;
            padding: clamp(1.1rem, 3.5vw, 2.4rem); margin-bottom: 1rem;
        }
        .research-banner h1 {color: white; padding: .2rem 0 .5rem; line-height: 1.1;}
        .research-banner p {margin: 0; max-width: 65ch;}
        [data-testid="stMetric"] {
            background: #fff; border: 1px solid #dce8e0; border-radius: .85rem;
            padding: .85rem 1rem;
        }
        [data-testid="stFileUploaderDropzone"] {min-height: 8rem;}
        @media (max-width: 640px) {
            .stMainBlockContainer {padding-top: 3.4rem;}
            .research-banner h1 {font-size: 1.85rem;}
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def heading(title: str, description: str) -> None:
    st.markdown(
        f'<div class="research-banner"><h1>{escape(title)}</h1>'
        f"<p>{escape(description)}</p></div>",
        unsafe_allow_html=True,
    )


def empty(message: str) -> None:
    st.info(message, icon=":material/info:")


def read_json(path):
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("Expected a JSON object.")
        return data
    except (OSError, ValueError) as error:
        st.warning(f"Could not read {path.name}. {error}")
        return {}


def table(filename: str, required: tuple[str, ...]):
    path = RESULTS / filename
    if not path.is_file():
        return None
    try:
        data = pd.read_csv(path)
        if missing := set(required) - set(data.columns):
            raise ValueError(f"Missing columns: {', '.join(sorted(missing))}")
        return data if not data.empty else None
    except (OSError, ValueError) as error:
        st.warning(f"Could not read {filename}. {error}")
        return None


def chart(figure, *, key=None, png_name=None) -> None:
    figure.update_layout(
        margin=dict(l=15, r=15, t=45, b=15),
        legend_title_text="",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#f8faf8",
    )
    config = {"displaylogo": False}
    if png_name:
        config["toImageButtonOptions"] = {"format": "png", "filename": png_name, "scale": 2}
    st.plotly_chart(figure, key=key, config=config)

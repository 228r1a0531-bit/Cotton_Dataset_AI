"""Streamlit entry point for CottonLeaf AI."""

import streamlit as st

from config import APP_TITLE


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    navigation = st.navigation(
        [
            st.Page("pages/overview.py", title="Overview", default=True),
            st.Page("pages/inspect_image.py", title="Inspect a leaf image"),
        ]
    )
    navigation.run()


if __name__ == "__main__":
    main()

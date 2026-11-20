"""CottonLeaf AI research dashboard."""

import streamlit as st

from config import APP_TITLE
from utils.dashboard import apply_style


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, page_icon=":material/eco:", layout="wide")
    apply_style()
    navigation = st.navigation(
        [
            st.Page("pages/overview.py", title="Overview", icon=":material/dashboard:", default=True),
            st.Page("pages/dataset.py", title="Dataset & Training", icon=":material/database:"),
            st.Page("pages/inspect_image.py", title="Upload & Classification", icon=":material/upload:"),
            st.Page("pages/comparisons.py", title="Model Comparisons", icon=":material/compare_arrows:"),
            st.Page("pages/evaluation.py", title="Evaluation Metrics", icon=":material/monitoring:"),
            st.Page("pages/architecture.py", title="Model Architecture", icon=":material/account_tree:"),
        ]
    )
    with st.sidebar:
        st.header("CottonLeaf AI")
        st.caption("Cotton disease research")
    navigation.run()


if __name__ == "__main__":
    main()

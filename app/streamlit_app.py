from pathlib import Path

import streamlit as st


ROOT = Path(__file__).resolve().parent

st.set_page_config(
    page_title="SkinSight research workspace",
    page_icon=":material/health_and_safety:",
    layout="wide",
    initial_sidebar_state="expanded",
)

page = st.navigation(
    [
        st.Page(
            str(ROOT / "app.py"),
            title="Analyze",
            icon=":material/health_and_safety:",
        ),
        st.Page(
            str(ROOT / "app_pages" / "model_performance.py"),
            title="Model performance",
            icon=":material/monitoring:",
        ),
        st.Page(
            str(ROOT / "app_pages" / "dataset_explorer.py"),
            title="Dataset explorer",
            icon=":material/dataset:",
        ),
        st.Page(
            str(ROOT / "app_pages" / "about.py"),
            title="About the research",
            icon=":material/menu_book:",
        ),
    ],
    position="top",
)

page.run()

from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parents[1]
METADATA_PATH = ROOT.parent / "data" / "raw" / "HAM10000_metadata.csv"

CLASS_LABELS = {
    "akiec": "Actinic keratoses",
    "bcc": "Basal cell carcinoma",
    "bkl": "Benign keratosis-like lesions",
    "df": "Dermatofibroma",
    "mel": "Melanoma",
    "nv": "Melanocytic nevi",
    "vasc": "Vascular lesions",
}

st.markdown("<div class='app-kicker'>Dataset workspace</div>", unsafe_allow_html=True)
st.title("Dataset explorer", icon=":material/dataset:")
st.markdown(
    "Inspect the HAM10000 metadata used to organize the training and evaluation split."
)


@st.cache_data
def load_metadata(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


if not METADATA_PATH.exists():
    with st.container(border=True):
        st.subheader("Metadata not found", divider="gray")
        st.write(
            "The explorer is ready, but HAM10000_metadata.csv is not present in data/raw. "
            "Add the dataset metadata file to populate these views."
        )
    st.stop()

metadata = load_metadata(str(METADATA_PATH))
class_column = "dx" if "dx" in metadata.columns else None

with st.container(horizontal=True):
    st.metric("Metadata rows", f"{len(metadata):,}", border=True)
    st.metric("Metadata fields", f"{len(metadata.columns)}", border=True)
    st.metric(
        "Classes",
        f"{metadata[class_column].nunique()}" if class_column else "Not detected",
        border=True,
    )

if class_column:
    distribution = (
        metadata[class_column]
        .value_counts()
        .rename_axis("Class")
        .reset_index(name="Images")
    )
    distribution["Label"] = distribution["Class"].map(CLASS_LABELS).fillna(
        distribution["Class"]
    )

    st.space("medium")
    st.subheader("Class balance", divider="gray")
    st.caption(
        "The dataset is unevenly distributed. This matters when interpreting accuracy and macro metrics."
    )
    chart = (
        alt.Chart(distribution)
        .mark_bar(size=28, cornerRadiusEnd=4)
        .encode(
            x=alt.X("Images:Q", title="Images"),
            y=alt.Y("Label:N", title=None, sort="-x"),
            color=alt.value("#0f766e"),
            tooltip=[
                alt.Tooltip("Label:N", title="Class"),
                alt.Tooltip("Images:Q", title="Images"),
            ],
        )
        .properties(height=300)
        .configure_view(stroke=None)
    )
    st.altair_chart(
        chart,
        width="stretch",
        alt="Horizontal bar chart of images per lesion class",
    )

    st.dataframe(
        distribution[["Class", "Label", "Images"]],
        hide_index=True,
        width="stretch",
        alt="Dataset image counts by lesion class",
    )
else:
    st.warning(
        "The metadata file does not contain the expected dx class column.",
        icon=":material/warning:",
    )

with st.expander("Metadata preview", icon=":material/table_view:"):
    st.dataframe(
        metadata.head(100),
        hide_index=True,
        width="stretch",
        alt="Preview of HAM10000 metadata",
    )

import streamlit as st


st.markdown("<div class='app-kicker'>Project notes</div>", unsafe_allow_html=True)
st.title("About the research", icon=":material/menu_book:")
st.markdown(
    "SkinSight is a research and educational prototype for multi-class classification of dermatoscopic images."
)

with st.container(border=True):
    st.subheader("What the workspace contains", divider="gray")
    st.write(
        "Analyze runs the trained EfficientNetB0 model and generates a Grad-CAM explanation. "
        "Model performance collects training and held-out evaluation artifacts. Dataset explorer "
        "summarizes HAM10000 metadata when it is available."
    )

st.space("medium")
left, right = st.columns(2)
with left:
    st.subheader("Method", divider="gray")
    st.markdown(
        "- EfficientNetB0 transfer learning\n"
        "- Two-stage training with fine-tuning\n"
        "- Seven lesion categories\n"
        "- Grad-CAM visual explanation\n"
        "- Lesion-aware dataset split"
    )
with right:
    st.subheader("Responsible use", divider="gray")
    st.markdown(
        "- Predictions can be incorrect\n"
        "- Confidence is not diagnostic certainty\n"
        "- Heatmaps do not prove causality\n"
        "- Outputs require qualified human review\n"
        "- Never use this tool for clinical decisions"
    )

st.space("medium")
st.warning(
    "SkinSight is not a medical diagnostic system. It must not be used as a substitute for professional examination, diagnosis, or treatment.",
    icon=":material/medical_information:",
)

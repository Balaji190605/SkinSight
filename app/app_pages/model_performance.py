from pathlib import Path

import streamlit as st


ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT.parent / "outputs" / "figures"

st.markdown(
    "<div class='app-kicker'>Evaluation workspace</div>",
    unsafe_allow_html=True,
)
st.title("Model performance", icon=":material/monitoring:")
st.markdown(
    "Review the model's behavior across training and held-out evaluation artifacts."
)

with st.container(horizontal=True):
    st.metric("Test accuracy", "67.55%", border=True)
    st.metric("Macro F1", "50.92%", border=True)
    st.metric("Macro ROC-AUC", "90.18%", border=True)

st.space("medium")

training_accuracy = FIGURES / "training_accuracy.png"
training_loss = FIGURES / "training_loss.png"
if training_accuracy.exists() and training_loss.exists():
    st.subheader("Training behavior", divider="gray")
    st.caption(
        "The two-stage training history shows optimization of the classification head followed by fine-tuning."
    )
    accuracy_col, loss_col = st.columns(2)
    with accuracy_col:
        st.image(
            training_accuracy,
            caption="Accuracy across training epochs",
            width="stretch",
            alt="Training and validation accuracy across epochs",
        )
    with loss_col:
        st.image(
            training_loss,
            caption="Loss across training epochs",
            width="stretch",
            alt="Training and validation loss across epochs",
        )
else:
    st.info("Training history figures are not available yet.", icon=":material/insights:")

st.space("medium")
st.subheader("Held-out evaluation", divider="gray")
st.caption(
    "These figures summarize model discrimination, class balance, and the types of mistakes made on evaluation data."
)

evaluation_figures = [
    ("ROC curves", "roc_curves.png", "ROC curves across the seven lesion classes"),
    (
        "Confusion matrix",
        "confusion_matrix.png",
        "Confusion matrix for the held-out evaluation set",
    ),
    (
        "Class distribution",
        "class_distribution.png",
        "Class distribution in the dataset",
    ),
]
available = [
    (label, FIGURES / filename, image_alt)
    for label, filename, image_alt in evaluation_figures
    if (FIGURES / filename).exists()
]
if available:
    columns = st.columns(len(available))
    for column, (label, path, image_alt) in zip(columns, available):
        with column:
            st.image(path, caption=label, width="stretch", alt=image_alt)
else:
    st.info("Evaluation figures are not available yet.", icon=":material/insights:")

with st.container(border=True):
    st.subheader("How to read this", divider="gray")
    st.write(
        "Accuracy measures overall correctness, while macro F1 gives every class equal weight. "
        "ROC-AUC measures ranking quality and does not mean the model is clinically reliable. "
        "Use this page to understand model behavior, not to make a diagnosis."
    )

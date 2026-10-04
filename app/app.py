import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
import altair as alt
from PIL import Image


# ============================================================
# PATHS AND CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = ROOT / "models" / "skinsight_efficientnetb0.keras"
CLASS_NAMES_PATH = ROOT / "models" / "class_names.json"

IMG_SIZE = (224, 224)


# Human-readable class names
CLASS_LABELS = {
    "akiec": "Actinic keratoses",
    "bcc": "Basal cell carcinoma",
    "bkl": "Benign keratosis-like lesions",
    "df": "Dermatofibroma",
    "mel": "Melanoma",
    "nv": "Melanocytic nevi",
    "vasc": "Vascular lesions",
}


ACTIVE_THEME = getattr(st.context.theme, "type", None) or "light"
THEME_COLORS = {
    "light": {
        "ink": "#17211f",
        "muted": "#53635f",
        "accent": "#0f766e",
        "accent_soft": "#e6f1ee",
        "accent_border": "#a8cfc5",
        "accent_deep": "#123d38",
        "surface": "#ffffff",
        "surface_soft": "#edf2ef",
        "border": "#d6dfdb",
        "warning": "#5c4c37",
    },
    "dark": {
        "ink": "#e8f0ed",
        "muted": "#a8bab5",
        "accent": "#61b8aa",
        "accent_soft": "#183d3a",
        "accent_border": "#397169",
        "accent_deep": "#d2eee8",
        "surface": "#182522",
        "surface_soft": "#20312d",
        "border": "#354b45",
        "warning": "#e7c995",
    },
}
COLORS = THEME_COLORS[ACTIVE_THEME if ACTIVE_THEME in THEME_COLORS else "light"]


# ============================================================
# CUSTOM STYLING
# ============================================================

st.markdown(
    f"""
    <style>
    :root {{
        --ss-ink: {COLORS['ink']};
        --ss-muted: {COLORS['muted']};
        --ss-accent: {COLORS['accent']};
        --ss-accent-soft: {COLORS['accent_soft']};
        --ss-accent-border: {COLORS['accent_border']};
        --ss-accent-deep: {COLORS['accent_deep']};
        --ss-surface: {COLORS['surface']};
        --ss-surface-soft: {COLORS['surface_soft']};
        --ss-border: {COLORS['border']};
        --ss-warning: {COLORS['warning']};
    }}
    </style>
    """
    + """
    <style>
    .app-kicker {
        color: var(--ss-accent);
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        margin-bottom: 0.7rem;
    }

    .app-title {
        color: var(--ss-ink);
        font-size: clamp(2.4rem, 5vw, 4.5rem);
        font-weight: 760;
        letter-spacing: -0.055em;
        line-height: 0.96;
        margin: 0;
    }

    .app-subtitle {
        color: var(--ss-muted);
        font-size: 1.08rem;
        line-height: 1.55;
        margin: 1rem 0 0;
        max-width: 44rem;
    }

    .header-meta {
        border-left: 2px solid var(--ss-accent-border);
        color: var(--ss-muted);
        font-size: 0.9rem;
        line-height: 1.55;
        padding: 0.4rem 0 0.4rem 1.1rem;
    }

    .section-heading {
        color: var(--ss-ink);
        font-size: 1.7rem;
        font-weight: 700;
        letter-spacing: -0.025em;
        margin: 0;
    }

    .section-copy {
        color: var(--ss-muted);
        font-size: 1.05rem;
        line-height: 1.55;
        margin: 0.35rem 0 0;
    }

    .upload-panel {
        background: var(--ss-surface);
        border: 1px solid var(--ss-border);
        border-radius: 0.75rem;
        padding: 1.25rem 1.35rem 1.05rem;
    }

    .prediction-box {
        background: var(--ss-accent-soft);
        border: 1px solid var(--ss-accent-border);
        border-radius: 0.75rem;
        padding: 2rem 1.7rem;
    }

    .prediction-label {
        color: var(--ss-accent);
        font-size: 0.85rem;
        font-weight: 700;
        letter-spacing: 0.12em;
        text-transform: uppercase;
    }

    .prediction-class {
        color: var(--ss-accent-deep);
        font-size: clamp(2rem, 3.4vw, 3rem);
        font-weight: 760;
        letter-spacing: -0.045em;
        line-height: 1.05;
        margin-top: 0.45rem;
    }

    .confidence {
        color: var(--ss-accent-deep);
        font-size: 1.25rem;
        margin-top: 0.8rem;
    }

    .result-note {
        color: var(--ss-muted);
        font-size: 1rem;
        line-height: 1.5;
        margin-top: 1rem;
    }

    .empty-panel {
        background: var(--ss-surface-soft);
        border: 1px dashed var(--ss-accent-border);
        border-radius: 0.75rem;
        padding: 1.4rem;
    }

    .disclaimer-copy {
        color: var(--ss-warning);
        font-size: 0.9rem;
        line-height: 1.5;
    }

    @media (max-width: 768px) {
        .app-title { font-size: 3rem; }
        .header-meta { border-left: 0; border-top: 2px solid var(--ss-accent-border); margin-top: 1.5rem; padding: 0.85rem 0 0; }
    }
    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title("SkinSight", icon=":material/health_and_safety:")

    st.caption(
        f"{ACTIVE_THEME.title()} theme · Change appearance from Streamlit settings"
    )

    st.markdown(
        """
        **Explainable Skin Lesion Classification**

        SkinSight is a research and educational prototype
        that uses transfer learning with EfficientNetB0 to
        classify dermatoscopic skin-lesion images.
        """
    )

    st.subheader("Model", divider="gray")

    st.write("**Architecture:** EfficientNetB0")
    st.write("**Learning:** Transfer Learning + Fine-tuning")
    st.write("**Input:** 224 × 224 pixels")
    st.write("**Classes:** 7")

    st.subheader("Dataset", divider="gray")

    st.write("**Dataset:** HAM10000")
    st.write("**Images:** 10,015")
    st.write("**Lesion-aware split:** Yes")

    st.subheader("Evaluation", divider="gray")

    st.write("**Test Accuracy:** 67.55%")
    st.write("**Macro F1:** 50.92%")
    st.write("**Weighted F1:** 70.29%")
    st.write("**Macro ROC-AUC:** 90.18%")

    st.markdown("---")

    st.caption(
        "SkinSight is intended for research and educational "
        "demonstration only."
    )


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="app-kicker">Explainable image analysis</div>',
    unsafe_allow_html=True
)

hero_copy, hero_meta = st.columns([1.55, 0.8], vertical_alignment="bottom")
with hero_copy:
    st.markdown('<h1 class="app-title">A clearer view of the model\'s answer.</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p class="app-subtitle">Upload a dermatoscopic image to see the predicted lesion class, probability distribution, and Grad-CAM explanation in one place.</p>',
        unsafe_allow_html=True,
    )
with hero_meta:
    st.markdown(
        '<div class="header-meta"><strong>Research prototype</strong><br>Built with EfficientNetB0 and HAM10000 data.<br>For education only, not diagnosis.</div>',
        unsafe_allow_html=True,
    )

st.space("large")


# ============================================================
# MODEL LOADING
# ============================================================

@st.cache_resource
def load_model():

    return tf.keras.models.load_model(
        MODEL_PATH
    )


# ============================================================
# CLASS NAME LOADING
# ============================================================

@st.cache_data
def load_class_names():

    with open(
        CLASS_NAMES_PATH,
        "r"
    ) as f:

        return json.load(f)


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

def preprocess(image):

    image = image.convert(
        "RGB"
    ).resize(
        IMG_SIZE,
        resample=Image.Resampling.NEAREST
    )

    array = np.asarray(
        image,
        dtype=np.float32
    )

    tensor = np.expand_dims(
        array,
        axis=0
    )

    return tensor


# ============================================================
# GRAD-CAM LAYER
# ============================================================

def find_gradcam_layer(
    base_model
):

    for layer in reversed(
        base_model.layers
    ):

        try:

            if len(
                layer.output.shape
            ) == 4:

                return layer

        except Exception:

            pass

    raise RuntimeError(
        "Could not find a suitable Grad-CAM layer."
    )


# ============================================================
# GRAD-CAM GENERATION
# ============================================================

def generate_gradcam(
    model,
    image_tensor,
    class_index
):

    base_model = model.get_layer(
        "efficientnetb0"
    )

    target_layer = find_gradcam_layer(
        base_model
    )

    feature_model = tf.keras.Model(
        inputs=base_model.input,
        outputs=target_layer.output
    )

    with tf.GradientTape() as tape:

        conv_outputs = feature_model(
            image_tensor,
            training=False
        )

        x = model.get_layer(
            "global_average_pooling"
        )(conv_outputs)

        x = model.get_layer(
            "dropout"
        )(
            x,
            training=False
        )

        predictions = model.get_layer(
            "predictions"
        )(x)

        class_score = predictions[
            :,
            class_index
        ]

    gradients = tape.gradient(
        class_score,
        conv_outputs
    )

    if gradients is None:

        raise RuntimeError(
            "Gradients could not be calculated."
        )

    weights = tf.reduce_mean(
        gradients,
        axis=(1, 2)
    )

    cam = tf.reduce_sum(
        conv_outputs *
        weights[:, None, None, :],
        axis=-1
    )

    cam = tf.maximum(
        cam,
        0
    )

    cam = cam / (
        tf.reduce_max(cam) + 1e-8
    )

    return (
        cam[0].numpy(),
        target_layer.name
    )


# ============================================================
# CREATE GRAD-CAM OVERLAY
# ============================================================

def create_gradcam_overlay(
    image,
    heatmap
):

    original = np.asarray(
        image.convert("RGB")
    )

    original = cv2.cvtColor(
        original,
        cv2.COLOR_RGB2BGR
    )

    heatmap = cv2.resize(
        heatmap,
        (
            original.shape[1],
            original.shape[0]
        )
    )

    heatmap_uint8 = np.uint8(
        255 * heatmap
    )

    colored_heatmap = cv2.applyColorMap(
        heatmap_uint8,
        cv2.COLORMAP_JET
    )

    overlay = cv2.addWeighted(
        original,
        0.55,
        colored_heatmap,
        0.45,
        0
    )

    overlay = cv2.cvtColor(
        overlay,
        cv2.COLOR_BGR2RGB
    )

    return overlay


# ============================================================
# CHECK REQUIRED FILES
# ============================================================

if not MODEL_PATH.exists():

    st.error(
        f"Trained model not found:\n\n{MODEL_PATH}"
    )

    st.stop()


if not CLASS_NAMES_PATH.exists():

    st.error(
        f"Class names file not found:\n\n{CLASS_NAMES_PATH}"
    )

    st.stop()


class_names = load_class_names()


# ============================================================
# IMAGE UPLOAD
# ============================================================

with st.container(border=True):
    st.markdown('<h2 class="section-heading">Start with an image</h2>', unsafe_allow_html=True)
    st.markdown(
        '<p class="section-copy">Use a clear JPG or PNG dermatoscopic image. The model resizes it to 224 × 224 pixels before inference.</p>',
        unsafe_allow_html=True,
    )
    uploaded = st.file_uploader(
        "Choose a dermatoscopic image",
        type=["jpg", "jpeg", "png"],
        help="Upload a skin-lesion image for model prediction.",
    )


# ============================================================
# PROCESS IMAGE
# ============================================================

if uploaded:

    image = Image.open(
        uploaded
    ).convert(
        "RGB"
    )

    st.caption(f"Loaded image: {uploaded.name}")


    with st.spinner("Running image analysis"):
        model = load_model()
        tensor = preprocess(image)
        probabilities = model.predict(tensor, verbose=0)[0]
    st.caption(":material/check_circle: Analysis ready")


    # ========================================================
    # PREDICTION
    # ========================================================

    predicted_index = int(
        np.argmax(probabilities)
    )

    predicted_class = class_names[
        predicted_index
    ]

    confidence = float(
        probabilities[
            predicted_index
        ]
    )

    readable_prediction = CLASS_LABELS.get(
        predicted_class,
        predicted_class
    )


    # ========================================================
    # IMAGE + PREDICTION
    # ========================================================

    st.space("large")
    st.markdown('<h2 class="section-heading">Analysis result</h2>', unsafe_allow_html=True)
    st.markdown(
        '<p class="section-copy">The result below reflects the model output for this image. It is not a clinical assessment.</p>',
        unsafe_allow_html=True,
    )

    image_col, prediction_col = st.columns(
        [1.1, 1]
    )


    with image_col:

        st.image(
            image,
            caption="Uploaded dermatoscopic image",
            width="stretch",
            alt="Uploaded dermatoscopic image",
        )

        with st.container(border=True):
            st.caption("Input image")
            st.write("The image is used only for this analysis session.")


    with prediction_col:

        st.markdown(
            f"""
            <div class="prediction-box">
                <div class="prediction-label">Most likely class</div>
                <div class="prediction-class">{readable_prediction}</div>
                <div class="confidence">Model confidence: <strong>{confidence * 100:.2f}%</strong></div>
                <div class="result-note">Confidence is the model's probability estimate, not a measure of diagnostic certainty.</div>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.write("")

        st.caption(f"Internal class label: `{predicted_class}`")

        if confidence < 0.60:
            st.warning(
                "The leading class is below 60% probability. Treat this output as especially uncertain.",
                icon=":material/priority_high:",
            )
        else:
            st.caption("The leading class has the strongest model probability for this image.")


    # ========================================================
    # PROBABILITY TABLE
    # ========================================================

    st.space("large")
    st.markdown('<h2 class="section-heading">Probability profile</h2>', unsafe_allow_html=True)
    st.markdown(
        '<p class="section-copy">A ranked view of how the model distributed its probability across all seven classes.</p>',
        unsafe_allow_html=True,
    )


    probability_df = pd.DataFrame(
        {
            "Class": class_names,
            "Description": [
                CLASS_LABELS.get(
                    name,
                    name
                )
                for name in class_names
            ],
            "Probability (%)": [
                round(
                    float(
                        probability
                    ) * 100,
                    2
                )
                for probability in probabilities
            ]
        }
    )


    probability_df = probability_df.sort_values(
        "Probability (%)",
        ascending=False
    ).reset_index(
        drop=True
    )

    st.markdown("#### Leading signals")
    with st.container(horizontal=True):
        for _, row in probability_df.head(3).iterrows():
            st.metric(
                row["Description"],
                f"{row['Probability (%)']:.2f}%",
                border=True,
            )

    st.space("small")

    st.dataframe(
        probability_df,
        width="stretch",
        hide_index=True,
        alt="Ranked probability table for the seven lesion classes",
    )


    # ========================================================
    # PROBABILITY CHART
    # ========================================================

    chart_df = probability_df[["Description", "Probability (%)"]]
    chart_max = max(100, float(chart_df["Probability (%)"].max()) + 8)
    secondary_bar = "#395950" if ACTIVE_THEME == "dark" else "#8ab2aa"
    probability_chart = (
        alt.Chart(chart_df)
        .mark_bar(size=24, cornerRadiusEnd=4)
        .encode(
            x=alt.X(
                "Probability (%):Q",
                title="Model probability",
                scale=alt.Scale(domain=[0, chart_max]),
            ),
            y=alt.Y(
                "Description:N",
                title=None,
                sort="-x",
            ),
            color=alt.condition(
                alt.datum.Description == readable_prediction,
                alt.value(COLORS["accent"]),
                alt.value(secondary_bar),
            ),
            tooltip=[
                alt.Tooltip("Description:N", title="Class"),
                alt.Tooltip("Probability (%):Q", title="Probability", format=".2f"),
            ],
        )
        .properties(height=300)
        .configure_view(stroke=None)
        .configure_axis(
            labelColor=COLORS["muted"],
            titleColor=COLORS["muted"],
            domainColor=COLORS["border"],
            tickColor=COLORS["border"],
            gridColor=COLORS["border"],
        )
    )
    st.altair_chart(
        probability_chart,
        width="stretch",
        alt="Horizontal probability chart ranking the seven lesion classes",
    )

    report_payload = {
        "application": "SkinSight",
        "file_name": uploaded.name,
        "predicted_class": predicted_class,
        "predicted_class_label": readable_prediction,
        "model_confidence": round(confidence, 6),
        "probabilities": {
            row["Class"]: round(float(row["Probability (%)"]) / 100, 6)
            for _, row in probability_df.iterrows()
        },
        "disclaimer": "Research and educational output only. Not a medical diagnosis.",
    }

    with st.container(horizontal=True, vertical_alignment="center"):
        st.caption("Save the current model output for later review.")
        st.download_button(
            "Download analysis",
            data=json.dumps(report_payload, indent=2),
            file_name=f"skinsight-{Path(uploaded.name).stem}.json",
            mime="application/json",
            icon=":material/download:",
        )


    # ========================================================
    # GRAD-CAM
    # ========================================================

    st.space("large")
    st.markdown('<h2 class="section-heading">Why the model looked there</h2>', unsafe_allow_html=True)


    st.write(
        "Grad-CAM highlights image regions that contributed "
        "to the model's prediction. Warmer regions indicate "
        "areas with stronger influence on the selected class."
    )


    try:

        heatmap, layer_name = generate_gradcam(
            model,
            tensor,
            predicted_index
        )

        overlay = create_gradcam_overlay(
            image,
            heatmap
        )


        grad_col1, grad_col2 = st.columns(
            2
        )


        with grad_col1:

            st.markdown("#### Original image")

            st.image(
                image,
                width="stretch",
                alt="Original uploaded dermatoscopic image",
            )


        with grad_col2:

            st.markdown("#### Grad-CAM visualization")

            st.image(
                overlay,
                width="stretch",
                alt="Grad-CAM heatmap overlay showing regions influencing the prediction",
            )


        st.caption(
            f"Feature layer used for Grad-CAM: "
            f"{layer_name}"
        )


        st.caption("Warmer regions indicate areas with stronger influence on the selected class. This visualization does not establish causality.")


    except Exception as e:

        st.error(
            f"Grad-CAM generation failed: {e}"
        )


    # ========================================================
    # MODEL INFORMATION
    # ========================================================

    st.space("large")
    st.markdown('<h2 class="section-heading">About the model</h2>', unsafe_allow_html=True)


    info_col1, info_col2, info_col3 = st.columns(
        3
    )


    with info_col1:

        st.metric(
            "Architecture",
            "EfficientNetB0"
        )


    with info_col2:

        st.metric(
            "Input Size",
            "224 × 224"
        )


    with info_col3:

        st.metric(
            "Classes",
            "7"
        )


    st.caption(
        """
        SkinSight uses EfficientNetB0 as a transfer-learning
        backbone. The ImageNet-pretrained network is adapted
        to classify seven categories from the HAM10000 dataset,
        followed by fine-tuning of the later convolutional layers.
        """
    )

    training_accuracy_path = ROOT / "outputs" / "figures" / "training_accuracy.png"
    training_loss_path = ROOT / "outputs" / "figures" / "training_loss.png"
    if training_accuracy_path.exists() and training_loss_path.exists():
        with st.expander("Training history", icon=":material/monitoring:"):
            st.caption("These curves show how the model changed during its two training stages.")
            history_col1, history_col2 = st.columns(2)
            with history_col1:
                st.image(
                    training_accuracy_path,
                    caption="Training and validation accuracy",
                    width="stretch",
                    alt="Training and validation accuracy across epochs",
                )
            with history_col2:
                st.image(
                    training_loss_path,
                    caption="Training and validation loss",
                    width="stretch",
                    alt="Training and validation loss across epochs",
                )

    evidence_paths = {
        "Class distribution": ROOT / "outputs" / "figures" / "class_distribution.png",
        "ROC curves": ROOT / "outputs" / "figures" / "roc_curves.png",
        "Confusion matrix": ROOT / "outputs" / "figures" / "confusion_matrix.png",
    }
    available_evidence = {
        label: path for label, path in evidence_paths.items() if path.exists()
    }
    if available_evidence:
        with st.expander("Evaluation evidence", icon=":material/insights:"):
            st.caption("Reference figures from the held-out evaluation set.")
            evidence_columns = st.columns(len(available_evidence))
            for column, (label, path) in zip(evidence_columns, available_evidence.items()):
                with column:
                    st.image(
                        path,
                        caption=label,
                        width="stretch",
                        alt=label,
                    )


    # ========================================================
    # DISCLAIMER
    # ========================================================

    st.warning(
        "SkinSight is an educational and research prototype. Its predictions and Grad-CAM visualizations are not medical diagnoses and must not be used for clinical decision-making.",
        icon=":material/medical_information:",
    )


else:

    st.space("large")
    st.markdown(
        '<div class="empty-panel"><strong>Ready when you are.</strong><br>Upload a dermatoscopic image above to begin analysis.</div>',
        unsafe_allow_html=True,
    )

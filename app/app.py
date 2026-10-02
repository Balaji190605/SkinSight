import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
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


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="SkinSight",
    page_icon="S",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM STYLING
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 42px;
        font-weight: 700;
        margin-bottom: 0px;
    }

    .subtitle {
        font-size: 20px;
        color: #666666;
        margin-top: 0px;
    }

    .section-title {
        font-size: 25px;
        font-weight: 650;
        margin-top: 20px;
        margin-bottom: 10px;
    }

    .prediction-box {
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #dddddd;
        background-color: #f8f9fa;
        text-align: center;
    }

    .prediction-class {
        font-size: 30px;
        font-weight: 700;
    }

    .confidence {
        font-size: 22px;
        margin-top: 8px;
    }

    .small-note {
        color: #666666;
        font-size: 14px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title("SkinSight")

    st.markdown(
        """
        **Explainable Skin Lesion Classification**

        SkinSight is a research and educational prototype
        that uses transfer learning with EfficientNetB0 to
        classify dermatoscopic skin-lesion images.
        """
    )

    st.markdown("---")

    st.markdown("### Model")

    st.write("**Architecture:** EfficientNetB0")
    st.write("**Learning:** Transfer Learning + Fine-tuning")
    st.write("**Input:** 224 × 224 pixels")
    st.write("**Classes:** 7")

    st.markdown("---")

    st.markdown("### Dataset")

    st.write("**Dataset:** HAM10000")
    st.write("**Images:** 10,015")
    st.write("**Lesion-aware split:** Yes")

    st.markdown("---")

    st.markdown("### Evaluation")

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
    '<div class="main-title">SkinSight</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Explainable Skin Lesion Classification using Transfer Learning'
    '</div>',
    unsafe_allow_html=True
)

st.write("")


st.info(
    "Educational / Research Prototype — SkinSight is not a "
    "medical diagnostic system and must not be used for "
    "clinical decision-making."
)


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

st.markdown(
    '<div class="section-title">Upload Image</div>',
    unsafe_allow_html=True
)

uploaded = st.file_uploader(
    "Upload a dermatoscopic image",
    type=[
        "jpg",
        "jpeg",
        "png"
    ],
    help="Upload a skin-lesion image for model prediction."
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

    st.caption(
        f"File: {uploaded.name}"
    )


    model = load_model()

    tensor = preprocess(
        image
    )


    # ========================================================
    # PREDICTION
    # ========================================================

    probabilities = model.predict(
        tensor,
        verbose=0
    )[0]

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

    st.markdown(
        '<div class="section-title">Prediction</div>',
        unsafe_allow_html=True
    )

    image_col, prediction_col = st.columns(
        [1.1, 1]
    )


    with image_col:

        st.image(
            image,
            caption="Uploaded dermatoscopic image",
            use_container_width=True
        )


    with prediction_col:

        st.markdown(
            '<div class="prediction-box">',
            unsafe_allow_html=True
        )

        st.markdown(
            f'<div class="prediction-class">'
            f'{readable_prediction}'
            f'</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            f'<div class="confidence">'
            f'Confidence: {confidence * 100:.2f}%'
            f'</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            '</div>',
            unsafe_allow_html=True
        )

        st.write("")

        st.write(
            f"**Model class:** `{predicted_class}`"
        )


    # ========================================================
    # PROBABILITY TABLE
    # ========================================================

    st.markdown(
        '<div class="section-title">'
        'Class Probabilities'
        '</div>',
        unsafe_allow_html=True
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


    st.dataframe(
        probability_df,
        use_container_width=True,
        hide_index=True
    )


    # ========================================================
    # PROBABILITY CHART
    # ========================================================

    chart_df = probability_df[
        [
            "Description",
            "Probability (%)"
        ]
    ].set_index(
        "Description"
    )


    st.bar_chart(
        chart_df
    )


    # ========================================================
    # GRAD-CAM
    # ========================================================

    st.markdown("---")

    st.markdown(
        '<div class="section-title">'
        'Explainability — Grad-CAM'
        '</div>',
        unsafe_allow_html=True
    )


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

            st.markdown(
                "#### Original Image"
            )

            st.image(
                image,
                use_container_width=True
            )


        with grad_col2:

            st.markdown(
                "#### Grad-CAM Visualization"
            )

            st.image(
                overlay,
                use_container_width=True
            )


        st.caption(
            f"Feature layer used for Grad-CAM: "
            f"{layer_name}"
        )


        st.success(
            "Grad-CAM generated successfully."
        )


    except Exception as e:

        st.error(
            f"Grad-CAM generation failed: {e}"
        )


    # ========================================================
    # MODEL INFORMATION
    # ========================================================

    st.markdown("---")

    st.markdown(
        '<div class="section-title">'
        'About the Model'
        '</div>',
        unsafe_allow_html=True
    )


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


    st.write(
        """
        SkinSight uses EfficientNetB0 as a transfer-learning
        backbone. The ImageNet-pretrained network is adapted
        to classify seven categories from the HAM10000 dataset,
        followed by fine-tuning of the later convolutional layers.
        """
    )


    # ========================================================
    # DISCLAIMER
    # ========================================================

    st.markdown("---")

    st.warning(
        """
        **Important Disclaimer**

        SkinSight is an educational and research prototype.
        Its predictions and Grad-CAM visualizations are not
        medical diagnoses and must not be used for clinical
        decision-making.
        """
    )


else:

    st.markdown("")

    st.info(
        "Upload a dermatoscopic image above to begin analysis."
    )
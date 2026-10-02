# SkinSight

## Explainable Skin Lesion Classification using Transfer Learning

SkinSight is a research and educational prototype for multi-class classification of dermatoscopic skin-lesion images using EfficientNetB0 transfer learning, fine-tuning, and Grad-CAM-based visual explainability.

The system accepts a dermatoscopic image, predicts one of seven lesion categories, displays class probabilities, and generates a Grad-CAM visualization to provide an interpretable view of image regions that contributed to the model prediction.

> **Important:** SkinSight is not a medical diagnostic system. Its predictions and visual explanations are intended for research and educational purposes only and must not be used as a substitute for professional medical evaluation.

---

## Features

* Seven-class skin-lesion image classification
* EfficientNetB0 transfer learning
* Two-stage training with fine-tuning
* Class-weighted training for imbalanced data
* Lesion-aware train/validation/test splitting
* Held-out test-set evaluation
* Confusion matrix generation
* Per-class precision, recall, and F1-score
* Macro ROC-AUC evaluation
* Grad-CAM visual explainability
* Interactive Streamlit web application
* CLI-based image prediction

---

## Supported Classes

| Code    | Class                         |
| ------- | ----------------------------- |
| `akiec` | Actinic keratoses             |
| `bcc`   | Basal cell carcinoma          |
| `bkl`   | Benign keratosis-like lesions |
| `df`    | Dermatofibroma                |
| `mel`   | Melanoma                      |
| `nv`    | Melanocytic nevi              |
| `vasc`  | Vascular lesions              |

---

## Dataset

SkinSight uses the **HAM10000 (Human Against Machine with 10000 training images)** dataset.

The dataset contains:

* **10,015 dermatoscopic images**
* **7 lesion classes**
* **7,470 unique lesions**

Because multiple images can correspond to the same lesion, the project uses `lesion_id` during dataset splitting to reduce information leakage between training, validation, and test sets.

### Dataset Split

| Split      | Images |
| ---------- | -----: |
| Training   |  6,984 |
| Validation |  1,524 |
| Test       |  1,507 |
| Total      | 10,015 |

The dataset is not included in the repository because of its size and distribution considerations. Download it separately and place the required files under `data/raw/`.

Expected structure:

```text
data/
└── raw/
    ├── HAM10000_metadata.csv
    └── images/
        ├── ISIC_*.jpg
        └── ...
```

---

## Model

SkinSight uses **EfficientNetB0** with ImageNet-pretrained weights.

### Training Approach

#### Stage 1 - Transfer Learning

* EfficientNetB0 feature extractor frozen
* Classification head trained
* Learning rate: `1e-3`

#### Stage 2 - Fine-Tuning

* Last 30 EfficientNetB0 layers made trainable
* Batch Normalization layers kept frozen
* Learning rate: `1e-5`
* Class weights applied to address class imbalance

### Input

```text
224 x 224 RGB image
```

### Output

```text
7-class softmax probability distribution
```

---

## Evaluation Results

The final model was evaluated on the held-out test set containing **1,507 images**.

| Metric            |     Result |
| ----------------- | ---------: |
| Test Accuracy     | **67.55%** |
| Macro F1-score    | **50.92%** |
| Weighted F1-score | **70.29%** |
| Macro ROC-AUC     | **90.18%** |

Because HAM10000 is strongly class-imbalanced, accuracy alone does not fully describe model performance. Macro F1-score is also reported so that performance across minority classes is visible.

---

## Explainability

SkinSight uses **Grad-CAM (Gradient-weighted Class Activation Mapping)** to provide a visual explanation of the model prediction.

The Grad-CAM visualization highlights image regions that contribute more strongly to the selected prediction.

The visualization should be interpreted as a model-attribution tool rather than a medically validated indication of a lesion or disease region.

---

## Project Architecture

```text
                    HAM10000 Dataset
                           |
                           v
                 Data Preparation
                           |
                           v
              Lesion-Aware Data Split
                           |
             +-------------+-------------+
             |             |             |
             v             v             v
           Train          Val           Test
             |
             v
      Image Preprocessing
      + Data Augmentation
             |
             v
        EfficientNetB0
             |
```

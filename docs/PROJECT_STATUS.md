# SkinSight Project Status

## Project

**SkinSight: Explainable Skin Lesion Classification using Transfer Learning**

SkinSight is a research and educational prototype for classifying dermatoscopic skin-lesion images into seven categories using EfficientNetB0 transfer learning, fine-tuning, and Grad-CAM explainability.

---

## Implementation Status

### Dataset

* HAM10000 dataset integrated
* 10,015 images verified
* 10,015 metadata records verified
* 7 lesion classes verified
* 7,470 unique lesions identified
* Image integrity checked
* No missing images detected

### Data Splitting

A lesion-aware splitting strategy was implemented using `lesion_id` to reduce data leakage caused by multiple images belonging to the same lesion.

| Split      | Images | Percentage |
| ---------- | -----: | ---------: |
| Training   |  6,984 |     69.74% |
| Validation |  1,524 |     15.22% |
| Test       |  1,507 |     15.05% |
| Total      | 10,015 |       100% |

No lesion overlap exists between the training, validation, and test sets.

### Model

* Architecture: EfficientNetB0
* Pretrained weights: ImageNet
* Input size: 224 × 224
* Number of classes: 7
* Training approach: Transfer Learning + Fine-Tuning
* Class-weighted training: Implemented
* Model checkpointing: Implemented
* Early stopping: Implemented
* Learning-rate reduction: Implemented

Final trained model:

```text
models/skinsight_efficientnetb0.keras
```

### Training

Two-stage training was completed.

#### Stage 1 — Transfer Learning

* EfficientNetB0 feature extractor frozen
* Learning rate: `1e-3`
* Best validation accuracy: **65.68%**

#### Stage 2 — Fine-Tuning

* Last 30 EfficientNetB0 layers fine-tuned
* Batch Normalization layers kept frozen
* Learning rate: `1e-5`
* Best validation accuracy: **68.83%**
* Best validation loss: **0.8372**

The best Stage 2 checkpoint was restored after early stopping.

### Evaluation

Evaluation was performed on the held-out test set containing 1,507 images.

| Metric            |     Result |
| ----------------- | ---------: |
| Test Accuracy     | **67.55%** |
| Macro F1-score    | **50.92%** |
| Weighted F1-score | **70.29%** |
| Macro ROC-AUC     | **90.18%** |

Generated evaluation output:

```text
outputs/figures/confusion_matrix.png
```

### Prediction

The command-line prediction pipeline has been implemented and verified.

It:

* Loads the trained model
* Loads and preprocesses an input image
* Produces probabilities for all seven classes
* Displays the predicted class
* Displays prediction confidence

### Grad-CAM

Grad-CAM has been successfully implemented.

Final feature layer used:

```text
top_activation
```

Grad-CAM generation has been verified with HAM10000 images.

Example output:

```text
outputs/gradcam/ISIC_0027419_gradcam.jpg
```

### Streamlit Application

The interactive Streamlit application has been implemented and tested.

Implemented features:

* Image upload
* Image preprocessing
* Seven-class prediction
* Prediction confidence
* Probability distribution
* Human-readable class names
* Grad-CAM visualization
* Model information
* Dataset information
* Evaluation metrics
* Research and educational disclaimer

The Streamlit prediction pipeline was verified against the command-line prediction pipeline to ensure consistent preprocessing and class mapping.

### Qualitative Testing

Seven representative images, one from each class, were tested through the Streamlit interface.

Results:

* Correct predictions: 4
* Incorrect predictions: 3
* Demonstration result: **4/7 (57.14%)**

This qualitative demonstration is **not used as the model's formal accuracy**. Formal performance is based on the 1,507-image held-out test set.

---

## Known Limitations

* The HAM10000 dataset is strongly class-imbalanced.
* Minority classes have substantially fewer examples than melanocytic nevi.
* Performance varies between lesion classes.
* Some visually similar lesion categories are difficult to distinguish.
* Grad-CAM is an interpretability technique and has not been clinically validated.
* The system has not undergone clinical validation.
* The prototype must not be used for medical diagnosis or treatment decisions.

---

## Final Project Components

* [x] Project structure
* [x] Dataset preparation
* [x] Image verification
* [x] Lesion-aware data splitting
* [x] Class-weight calculation
* [x] TensorFlow data pipeline
* [x] Data augmentation
* [x] EfficientNetB0 model
* [x] Transfer learning
* [x] Fine-tuning
* [x] Model checkpointing
* [x] Early stopping
* [x] Learning-rate scheduling
* [x] Model evaluation
* [x] Confusion matrix
* [x] CLI prediction
* [x] Grad-CAM
* [x] Streamlit integration
* [x] Consistent preprocessing
* [x] Qualitative application testing
* [x] Project documentation

---

## Remaining Work

The core implementation is complete.

Remaining work is primarily final project presentation and documentation:

* [ ] Final training/evaluation visualizations
* [ ] Final report
* [ ] IEEE-style paper
* [ ] Project presentation
* [ ] Viva preparation
* [ ] Final repository/submission packaging

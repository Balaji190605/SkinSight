# SkinSight

## Explainable Skin Lesion Classification using Transfer Learning

SkinSight is a research and educational prototype for **multi-class classification of dermatoscopic skin-lesion images** using **EfficientNetB0 transfer learning** with **Grad-CAM explainability**.

The project is based on the **HAM10000 dataset** and classifies images into seven skin-lesion categories.

> **Important:** SkinSight is an educational/research prototype and is **not a medical diagnostic system**. Its predictions and visual explanations must not be used for clinical diagnosis or treatment decisions.

---

## Features

* Seven-class skin lesion classification
* EfficientNetB0 transfer-learning model
* Two-stage training with fine-tuning
* Class-weight handling for class imbalance
* Train/validation/test dataset splitting
* Confusion matrix evaluation
* ROC curve visualization
* Training accuracy and loss visualization
* Class distribution visualization
* Command-line prediction
* Grad-CAM explainability
* Streamlit web interface
* Modular Python project structure

---

## Supported Classes

| Code    | Class                                         |
| ------- | --------------------------------------------- |
| `akiec` | Actinic keratoses / intraepithelial carcinoma |
| `bcc`   | Basal cell carcinoma                          |
| `bkl`   | Benign keratosis-like lesions                 |
| `df`    | Dermatofibroma                                |
| `mel`   | Melanoma                                      |
| `nv`    | Melanocytic nevi                              |
| `vasc`  | Vascular lesions                              |

The class order used by the trained model is stored in:

```text
models/class_names.json
```

---

# Project Architecture

```text
SkinSight/
│
├── app/
│   └── app.py
│
├── data/
│   ├── raw/
│   │   ├── HAM10000_metadata.csv
│   │   └── images/
│   │       └── *.jpg
│   │
│   ├── processed/
│   └── splits/
│       ├── train.csv
│       ├── val.csv
│       └── test.csv
│
├── docs/
│   └── PROJECT_STATUS.md
│
├── models/
│   ├── class_names.json
│   ├── class_weights.json
│   ├── training_history.json
│   └── skinsight_efficientnetb0.keras
│
├── outputs/
│   ├── figures/
│   │   ├── class_distribution.png
│   │   ├── confusion_matrix.png
│   │   ├── roc_curves.png
│   │   ├── training_accuracy.png
│   │   └── training_loss.png
│   │
│   └── gradcam/
│       └── ISIC_0027419_gradcam.jpg
│
├── src/
│   ├── config.py
│   ├── data_preparation.py
│   ├── dataset.py
│   ├── evaluate.py
│   ├── gradcam.py
│   ├── model.py
│   ├── predict.py
│   ├── train.py
│   ├── visualize_dataset.py
│   ├── visualize_roc.py
│   └── visualize_training.py
│
├── .gitignore
├── README.md
└── requirements.txt
```

---

# 1. Prerequisites

Install the following before setting up the project:

* Git
* Python 3.10 or compatible Python version
* VS Code recommended
* Windows PowerShell / Terminal

Check the installations:

```powershell
python --version
git --version
```

The project has been developed and tested in a Python virtual environment using TensorFlow.

---

# 2. Clone the Repository

Open PowerShell and run:

```powershell
git clone https://github.com/Balaji190605/SkinSight.git
```

Move into the project:

```powershell
cd SkinSight
```

Verify:

```powershell
git status
```

You should see the `main` branch and a clean working tree.

---

# 3. Create a Virtual Environment

From inside the `SkinSight` directory:

```powershell
python -m venv .venv
```

Activate it:

```powershell
.\.venv\Scripts\Activate.ps1
```

After activation, the terminal should show:

```text
(.venv) PS A:\SkinSight\SkinSight>
```

If PowerShell blocks script execution, run:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

Then activate the environment again:

```powershell
.\.venv\Scripts\Activate.ps1
```

---

# 4. Install Dependencies

Install the required Python packages:

```powershell
pip install -r requirements.txt
```

Verify TensorFlow:

```powershell
python -c "import tensorflow as tf; print(tf.__version__)"
```

Also verify Streamlit:

```powershell
streamlit --version
```

---

# 5. Dataset Setup

SkinSight uses the **HAM10000 dataset**.

The dataset is intentionally **not included in this GitHub repository** because of its size and dataset distribution considerations.

Download the HAM10000 dataset separately.

The expected structure is:

```text
data/
└── raw/
    ├── HAM10000_metadata.csv
    └── images/
        ├── ISIC_0024306.jpg
        ├── ISIC_0024307.jpg
        ├── ...
        └── *.jpg
```

The metadata CSV must contain the required fields including:

```text
image_id
dx
```

where `dx` represents the lesion class.

---

# 6. Verify the Dataset

After placing the dataset in `data/raw/`, run:

```powershell
python src/data_preparation.py
```

This checks/prepares the dataset and generates the required train/validation/test split files.

The generated split files are:

```text
data/splits/train.csv
data/splits/val.csv
data/splits/test.csv
```

You can verify the splits with:

```powershell
Get-ChildItem .\data\splits
```

---

# 7. Train the Model

To train the EfficientNetB0 model:

```powershell
python src/train.py
```

The training pipeline includes:

1. Dataset loading
2. Image preprocessing
3. Data augmentation
4. Class weighting
5. EfficientNetB0 transfer learning
6. Initial training
7. Fine-tuning
8. Model checkpointing
9. Training-history generation

The trained model is saved locally as:

```text
models/skinsight_efficientnetb0.keras
```

### Important

The trained `.keras` model is excluded from Git using `.gitignore`.

Therefore, a colleague cloning the repository will need to either:

* train the model locally, or
* obtain the trained model separately.

---

# 8. Evaluate the Model

After training, run:

```powershell
python src/evaluate.py
```

Evaluation results are generated under:

```text
outputs/figures/
```

The project currently includes:

```text
confusion_matrix.png
roc_curves.png
```

---

# 9. Generate Training Visualizations

Run:

```powershell
python src/visualize_training.py
```

This generates:

```text
outputs/figures/training_accuracy.png
outputs/figures/training_loss.png
```

---

# 10. Generate Dataset Visualizations

To generate the class-distribution visualization:

```powershell
python src/visualize_dataset.py
```

The output is saved under:

```text
outputs/figures/class_distribution.png
```

---

# 11. Generate ROC Curves

Run:

```powershell
python src/visualize_roc.py
```

The generated ROC visualization is saved as:

```text
outputs/figures/roc_curves.png
```

---

# 12. Command-Line Prediction

Once the trained model exists, a single image can be classified using:

```powershell
python src/predict.py --image .\data\raw\images\ISIC_0027419.jpg
```

Example output:

```text
SkinSight prediction
--------------------
bkl   : 50.78%
mel   : 40.14%
nv    : 4.76%
akiec : 2.61%
bcc   : 1.27%
df    : 0.41%
vasc  : 0.04%

Predicted class: bkl
Confidence: 50.78%
```

The exact output will depend on the trained model and input image.

---

# 13. Grad-CAM Explainability

SkinSight includes Grad-CAM to visualize image regions contributing to the model prediction.

Run:

```powershell
python src/gradcam.py --image .\data\raw\images\ISIC_0027419.jpg
```

A Grad-CAM image is generated under:

```text
outputs/gradcam/
```

For example:

```text
outputs/gradcam/ISIC_0027419_gradcam.jpg
```

The Grad-CAM visualization should be interpreted as a model-explainability visualization rather than evidence of a medical diagnosis.

---

# 14. Run the Streamlit Application

After the trained model is available, start the application:

```powershell
streamlit run app/app.py
```

Streamlit will provide a local URL, normally similar to:

```text
http://localhost:8501
```

Open the URL in a browser.

The application allows the user to:

1. Upload a dermatoscopic image
2. View the uploaded image
3. Generate the model prediction
4. View the predicted class
5. View model confidence
6. View class probabilities
7. View explainability information

---

# 15. Current Model Output

The application supports the following output classes:

```text
akiec
bcc
bkl
df
mel
nv
vasc
```

For example:

```text
Predicted Class: bkl
Model Confidence: 50.78%
```

Confidence represents the model's predicted probability for the selected class. It should not be interpreted as clinical certainty.

---

# 16. Important Files

### Model

```text
models/skinsight_efficientnetb0.keras
```

Trained EfficientNetB0 model.

### Class mapping

```text
models/class_names.json
```

Contains the class ordering used by the model.

### Class weights

```text
models/class_weights.json
```

Stores class-weight information used during training.

### Training history

```text
models/training_history.json
```

Contains recorded training metrics.

### Prediction

```text
src/predict.py
```

Performs inference on a single image.

### Grad-CAM

```text
src/gradcam.py
```

Generates explainability visualizations.

### Streamlit application

```text
app/app.py
```

Runs the web interface.

---

# 17. Git and Files Not Included in the Repository

The following are intentionally excluded using `.gitignore`:

```text
data/raw/
data/processed/
data/splits/
models/*.keras
models/*.h5
.venv/
__pycache__/
```

This keeps large datasets, trained model binaries, and local environment files out of GitHub.

The repository contains placeholder `.gitkeep` files where necessary to preserve the folder structure.

---

# 18. Development Workflow

For further development, the recommended workflow is:

```text
Clone repository
       ↓
Create virtual environment
       ↓
Install requirements
       ↓
Add HAM10000 dataset
       ↓
Prepare dataset
       ↓
Train / obtain trained model
       ↓
Evaluate model
       ↓
Run predictions
       ↓
Generate Grad-CAM
       ↓
Run Streamlit application
       ↓
Refine UI / model / explainability
       ↓
Test changes
       ↓
Commit changes
       ↓
Push to GitHub
```

---

# 19. Making Changes

Before modifying the project, create a separate branch:

```powershell
git checkout -b refinement
```

After making changes:

```powershell
git status
```

Review the changed files:

```powershell
git diff
```

Add the changes:

```powershell
git add .
```

Commit:

```powershell
git commit -m "Refine SkinSight application"
```

Push the branch:

```powershell
git push -u origin refinement
```

For changes that are ready to be merged into `main`, create a Pull Request on GitHub.

---

# 20. Testing Checklist

Before considering a refinement complete, verify:

### Environment

```powershell
python --version
pip --version
```

### Python files

```powershell
python -m py_compile src\*.py
```

If wildcard compilation is not supported as expected in PowerShell, compile individual files or use:

```powershell
python -m compileall src
```

### Prediction

```powershell
python src\predict.py --image .\data\raw\images\ISIC_0027419.jpg
```

### Grad-CAM

```powershell
python src\gradcam.py --image .\data\raw\images\ISIC_0027419.jpg
```

### Streamlit

```powershell
streamlit run app\app.py
```

Confirm that:

* The application starts without errors.
* Image upload works.
* Prediction is displayed.
* Class probabilities are displayed.
* Grad-CAM functionality works.
* Existing model outputs are not accidentally broken.

---

# 21. Refinement Goals

The project is functional, but further refinement may include:

* Improve Streamlit UI/UX
* Integrate Grad-CAM directly into the application
* Improve prediction result presentation
* Add clearer class descriptions
* Add model performance metrics
* Display confusion matrix and ROC results in the application
* Improve error handling
* Add loading/progress indicators
* Improve responsive layout
* Add image validation
* Improve documentation
* Add automated testing
* Improve code organization
* Add deployment configuration if required
* Improve accessibility and user guidance

Any refinement should preserve the research/educational nature of the project and avoid presenting model predictions as medical diagnoses.

---

# 22. Disclaimer

SkinSight is an academic/research prototype developed for educational purposes.

The system classifies dermatoscopic images using a machine-learning model trained on the HAM10000 dataset. Model predictions may be incorrect, and Grad-CAM visualizations do not establish causality or clinical significance.

**SkinSight must not be used as a substitute for professional medical examination, diagnosis, or treatment.**

---

## Repository

GitHub:

https://github.com/Balaji190605/SkinSight

---

## Project Status

Current implementation includes:

* EfficientNetB0 transfer-learning model
* Seven-class classification
* Dataset preparation
* Train/validation/test splits
* Model training and fine-tuning
* Model evaluation
* Confusion matrix
* ROC curves
* Training curves
* Class-distribution visualization
* Command-line prediction
* Grad-CAM generation
* Streamlit application
* GitHub repository setup

Further work is focused on **application refinement, explainability integration, testing, documentation, and presentation quality**.

import json

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

from config import CLASS_NAMES, FIGURE_DIR, IMG_SIZE, MODEL_PATH, SPLIT_DIR
from dataset import load_dataframe, make_dataset


def main():
    test_df = load_dataframe(SPLIT_DIR / "test.csv")
    test_ds = make_dataset(test_df, training=False)

    model = tf.keras.models.load_model(MODEL_PATH)
    probabilities = model.predict(test_ds)
    predictions = np.argmax(probabilities, axis=1)
    y_true = test_df["label"].values

    print("\\nClassification report:")
    print(
        classification_report(
            y_true, predictions, target_names=CLASS_NAMES, digits=4, zero_division=0
        )
    )

    try:
        y_onehot = tf.keras.utils.to_categorical(y_true, len(CLASS_NAMES))
        auc = roc_auc_score(y_onehot, probabilities, multi_class="ovr", average="macro")
        print(f"Macro ROC-AUC (OvR): {auc:.4f}")
    except ValueError as e:
        print(f"ROC-AUC could not be calculated: {e}")

    cm = confusion_matrix(y_true, predictions)
    plt.figure(figsize=(9, 7))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("SkinSight Confusion Matrix")
    plt.tight_layout()
    path = FIGURE_DIR / "confusion_matrix.png"
    plt.savefig(path, dpi=200)
    plt.close()
    print(f"Saved: {path}")


if __name__ == "__main__":
    main()

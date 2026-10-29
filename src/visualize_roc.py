import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import auc, roc_curve
from sklearn.preprocessing import label_binarize

from config import (
    CLASS_NAMES,
    FIGURE_DIR,
    IMAGE_DIR,
    IMG_SIZE,
    MODEL_PATH,
    SPLIT_DIR,
)


def load_test_data():
    test_df = pd.read_csv(SPLIT_DIR / "test.csv")

    image_paths = [IMAGE_DIR / f"{image_id}.jpg" for image_id in test_df["image_id"]]

    labels = (
        test_df["dx"]
        .map({name: index for index, name in enumerate(CLASS_NAMES)})
        .values
    )

    return image_paths, labels


def load_images(image_paths):
    images = []

    for image_path in image_paths:
        image = tf.keras.utils.load_img(
            image_path,
            target_size=IMG_SIZE,
        )

        array = tf.keras.utils.img_to_array(image)
        images.append(array)

    return np.asarray(images, dtype=np.float32)


def main():
    print("Loading test dataset...")

    image_paths, labels = load_test_data()
    images = load_images(image_paths)

    print(f"Test images: {len(images)}")

    print("Loading trained model...")
    model = tf.keras.models.load_model(MODEL_PATH)

    print("Generating predictions...")
    probabilities = model.predict(
        images,
        batch_size=32,
        verbose=1,
    )

    y_true = label_binarize(
        labels,
        classes=np.arange(len(CLASS_NAMES)),
    )

    plt.figure(figsize=(10, 8))

    auc_values = []

    for index, class_name in enumerate(CLASS_NAMES):
        fpr, tpr, _ = roc_curve(
            y_true[:, index],
            probabilities[:, index],
        )

        class_auc = auc(fpr, tpr)
        auc_values.append(class_auc)

        plt.plot(
            fpr,
            tpr,
            label=f"{class_name.upper()} (AUC = {class_auc:.3f})",
        )

    macro_auc = np.mean(auc_values)

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Random Classifier",
    )

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(
        f"One-vs-Rest ROC Curves\nHAM10000 Test Set — Macro ROC-AUC = {macro_auc:.3f}"
    )

    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    output_path = FIGURE_DIR / "roc_curves.png"
    plt.savefig(output_path, dpi=300)
    plt.close()

    print()
    print("ROC visualization generated successfully.")
    print(f"Macro ROC-AUC: {macro_auc:.4f}")
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()

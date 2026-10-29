import argparse

import numpy as np
import tensorflow as tf

from config import CLASS_NAMES, IMG_SIZE, MODEL_PATH


def predict(image_path):
    model = tf.keras.models.load_model(MODEL_PATH)

    image = tf.keras.utils.load_img(image_path, target_size=IMG_SIZE)
    array = tf.keras.utils.img_to_array(image)
    tensor = tf.expand_dims(array, 0)

    probabilities = model.predict(tensor, verbose=0)[0]
    order = np.argsort(probabilities)[::-1]

    print("\\nSkinSight prediction")
    print("--------------------")
    for idx in order:
        print(f"{CLASS_NAMES[idx]:6s}: {probabilities[idx] * 100:.2f}%")

    print(f"\\nPredicted class: {CLASS_NAMES[order[0]]}")
    print(f"Confidence: {probabilities[order[0]] * 100:.2f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    args = parser.parse_args()
    predict(args.image)

import argparse
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf

from config import CLASS_NAMES, GRADCAM_DIR, IMG_SIZE, MODEL_PATH


def find_gradcam_layer(base_model):
    """
    Return the final 4D feature layer inside EfficientNetB0.
    """
    for layer in reversed(base_model.layers):
        try:
            if len(layer.output.shape) == 4:
                return layer
        except Exception:
            pass

    raise RuntimeError("Could not find a suitable Grad-CAM layer.")


def preprocess(path):
    image = tf.keras.utils.load_img(path, target_size=IMG_SIZE)

    array = tf.keras.utils.img_to_array(image)
    tensor = tf.expand_dims(array, 0)

    return image, tensor


def make_gradcam(model, base_model, image_tensor, target_layer, class_index):
    """
    Generate Grad-CAM using the final EfficientNetB0
    feature representation.
    """

    # Model that gives us the feature maps from the
    # selected internal EfficientNet layer.
    feature_model = tf.keras.Model(inputs=base_model.input, outputs=target_layer.output)

    with tf.GradientTape() as tape:
        # This tensor is now directly connected to the
        # prediction calculation below.
        conv_outputs = feature_model(image_tensor, training=False)

        # EfficientNetB0 ends at top_activation.
        # Pass those feature maps through SkinSight's
        # classification head.
        x = model.get_layer("global_average_pooling")(conv_outputs)

        x = model.get_layer("dropout")(x, training=False)

        predictions = model.get_layer("predictions")(x)

        class_score = predictions[:, class_index]

    grads = tape.gradient(class_score, conv_outputs)

    if grads is None:
        raise RuntimeError("Gradients could not be calculated for the target layer.")

    # Importance of each feature-map channel.
    weights = tf.reduce_mean(grads, axis=(1, 2))

    # Weighted sum of feature maps.
    cam = tf.reduce_sum(conv_outputs * weights[:, None, None, :], axis=-1)

    # Keep positive activations.
    cam = tf.maximum(cam, 0)

    # Normalize between 0 and 1.
    cam = cam / (tf.reduce_max(cam) + 1e-8)

    return cam[0].numpy()


def save_overlay(image_path):

    print("Loading model...")

    model = tf.keras.models.load_model(MODEL_PATH)

    # Get EfficientNetB0 nested inside SkinSight.
    base_model = model.get_layer("efficientnetb0")

    _, tensor = preprocess(image_path)

    print("Running prediction...")

    probabilities = model.predict(tensor, verbose=0)[0]

    class_index = int(np.argmax(probabilities))

    class_name = CLASS_NAMES[class_index]

    confidence = float(probabilities[class_index])

    print("Finding Grad-CAM layer...")

    target_layer = find_gradcam_layer(base_model)

    print(f"Grad-CAM layer: {target_layer.name}")

    print("Generating Grad-CAM...")

    heatmap = make_gradcam(model, base_model, tensor, target_layer, class_index)

    original = cv2.imread(str(image_path))

    if original is None:
        raise ValueError("Unable to read image.")

    heatmap = cv2.resize(heatmap, (original.shape[1], original.shape[0]))

    heatmap_uint8 = np.uint8(255 * heatmap)

    colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

    overlay = cv2.addWeighted(original, 0.55, colored, 0.45, 0)

    output = GRADCAM_DIR / f"{Path(image_path).stem}_gradcam.jpg"

    cv2.imwrite(str(output), overlay)

    print()
    print("SkinSight Grad-CAM")
    print("------------------")
    print(f"Prediction: {class_name}")
    print(f"Confidence: {confidence:.4f}")
    print(f"Grad-CAM layer: {target_layer.name}")
    print(f"Saved: {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--image", required=True)

    args = parser.parse_args()

    save_overlay(args.image)


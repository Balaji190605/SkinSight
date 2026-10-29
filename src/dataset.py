from pathlib import Path

import pandas as pd
import tensorflow as tf

from config import BATCH_SIZE, CLASS_NAMES, IMAGE_DIR, IMG_SIZE

AUTOTUNE = tf.data.AUTOTUNE
LABEL_MAP = {name: i for i, name in enumerate(CLASS_NAMES)}


def _resolve_image_path(image_id):
    candidates = [
        IMAGE_DIR / f"{image_id}.jpg",
        IMAGE_DIR / f"{image_id}.jpeg",
        IMAGE_DIR / f"{image_id}.png",
    ]
    for p in candidates:
        if p.exists():
            return str(p)
    return str(candidates[0])


def load_dataframe(csv_path):
    df = pd.read_csv(csv_path)
    df["path"] = df["image_id"].apply(_resolve_image_path)
    df["label"] = df["dx"].map(LABEL_MAP)
    return df


def decode_image(path, label, training=False):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, IMG_SIZE)
    image = tf.cast(image, tf.float32)

    if training:
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_flip_up_down(image)
        image = tf.image.random_brightness(image, max_delta=0.08)
        image = tf.image.random_contrast(image, lower=0.9, upper=1.1)

    return image, tf.one_hot(label, len(CLASS_NAMES))


def make_dataset(df, training=False, batch_size=BATCH_SIZE):
    ds = tf.data.Dataset.from_tensor_slices((df["path"].values, df["label"].values))
    if training:
        ds = ds.shuffle(len(df), seed=42, reshuffle_each_iteration=True)
    ds = ds.map(lambda p, y: decode_image(p, y, training), num_parallel_calls=AUTOTUNE)
    return ds.batch(batch_size).prefetch(AUTOTUNE)

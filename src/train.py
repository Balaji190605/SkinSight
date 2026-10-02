import json

import tensorflow as tf
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
)

from config import (
    SPLIT_DIR,
    MODEL_DIR,
    MODEL_PATH,
    IMG_SIZE,
    NUM_CLASSES,
)
from dataset import load_dataframe, make_dataset
from model import build_model


HISTORY_PATH = MODEL_DIR / "training_history.json"


def compile_model(model, lr):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=lr
        ),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )


def create_callbacks():
    return [
        EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.3,
            patience=2,
            min_lr=1e-7,
            verbose=1,
        ),
        ModelCheckpoint(
            MODEL_PATH,
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
    ]


def main():
    print("=" * 60)
    print("SkinSight - EfficientNetB0 Training")
    print("=" * 60)

    # ---------------------------------------------------------
    # LOAD DATA
    # ---------------------------------------------------------

    train_df = load_dataframe(SPLIT_DIR / "train.csv")
    val_df = load_dataframe(SPLIT_DIR / "val.csv")

    print(f"\nTraining images:   {len(train_df)}")
    print(f"Validation images: {len(val_df)}")

    train_ds = make_dataset(
        train_df,
        training=True,
    )

    val_ds = make_dataset(
        val_df,
        training=False,
    )

    # ---------------------------------------------------------
    # LOAD CLASS WEIGHTS
    # ---------------------------------------------------------

    with open(MODEL_DIR / "class_weights.json") as f:
        class_weights = {
            int(k): float(v)
            for k, v in json.load(f).items()
        }

    print("\nClass weights:")
    print(class_weights)

    # ---------------------------------------------------------
    # BUILD MODEL
    # ---------------------------------------------------------

    model, base = build_model(
        NUM_CLASSES,
        (*IMG_SIZE, 3),
    )

    print("\nModel:")
    model.summary()

    # ---------------------------------------------------------
    # STAGE 1
    # TRAIN CLASSIFICATION HEAD
    # ---------------------------------------------------------

    print("\n" + "=" * 60)
    print("STAGE 1: TRAINING CLASSIFICATION HEAD")
    print("=" * 60)

    compile_model(
        model,
        lr=1e-3,
    )

    print(
        f"Trainable parameters: "
        f"{sum(tf.size(v).numpy() for v in model.trainable_variables):,}"
    )

    history_stage1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=10,
        class_weight=class_weights,
        callbacks=create_callbacks(),
    )

    # ---------------------------------------------------------
    # LOAD BEST STAGE 1 MODEL
    # ---------------------------------------------------------

    if MODEL_PATH.exists():
        model = tf.keras.models.load_model(MODEL_PATH)
        base = model.get_layer("efficientnetb0")

    # ---------------------------------------------------------
    # STAGE 2
    # FINE-TUNE UPPER EFFICIENTNETB0 LAYERS
    # ---------------------------------------------------------

    print("\n" + "=" * 60)
    print("STAGE 2: FINE-TUNING EFFICIENTNETB0")
    print("=" * 60)

    base.trainable = True

    # Freeze all but the last 30 EfficientNetB0 layers.
    freeze_until = max(
        0,
        len(base.layers) - 30,
    )

    for layer in base.layers[:freeze_until]:
        layer.trainable = False

    # Keep BatchNormalization layers frozen during fine-tuning.
    for layer in base.layers:
        if isinstance(
            layer,
            tf.keras.layers.BatchNormalization,
        ):
            layer.trainable = False

    compile_model(
        model,
        lr=1e-5,
    )

    print(
        f"Trainable parameters: "
        f"{sum(tf.size(v).numpy() for v in model.trainable_variables):,}"
    )

    history_stage2 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=15,
        class_weight=class_weights,
        callbacks=create_callbacks(),
    )

    # ---------------------------------------------------------
    # LOAD BEST CHECKPOINT
    # ---------------------------------------------------------

    if MODEL_PATH.exists():
        model = tf.keras.models.load_model(MODEL_PATH)

    # ---------------------------------------------------------
    # SAVE TRAINING HISTORY
    # ---------------------------------------------------------

    history = {
        "stage1": {
            key: [float(x) for x in values]
            for key, values in history_stage1.history.items()
        },
        "stage2": {
            key: [float(x) for x in values]
            for key, values in history_stage2.history.items()
        },
    }

    with open(HISTORY_PATH, "w") as f:
        json.dump(history, f, indent=2)

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)

    print(f"\nBest model: {MODEL_PATH}")
    print(f"Training history: {HISTORY_PATH}")

    print("\nFinal model:")
    model.summary()


if __name__ == "__main__":
    main()

import json
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.utils.class_weight import compute_class_weight

from config import (
    METADATA_PATH,
    SPLIT_DIR,
    CLASS_NAMES,
    RANDOM_SEED,
    MODEL_DIR,
)


def main():
    if not METADATA_PATH.exists():
        raise FileNotFoundError(
            f"Metadata not found: {METADATA_PATH}"
        )

    df = pd.read_csv(METADATA_PATH)

    required = {"image_id", "dx", "lesion_id"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df[df["dx"].isin(CLASS_NAMES)].copy()
    df = df.reset_index(drop=True)

    print("\nDataset:")
    print(f"Images: {len(df)}")
    print(f"Unique lesions: {df['lesion_id'].nunique()}")

    print("\nClass distribution:")
    print(df["dx"].value_counts())

    # ---------------------------------------------------------
    # LESION-AWARE 70 / 15 / 15 SPLIT
    # ---------------------------------------------------------
    #
    # 20 stratified group folds:
    #
    # 14 folds -> TRAIN ~= 70%
    #  3 folds -> VAL   ~= 15%
    #  3 folds -> TEST  ~= 15%
    #
    # The grouping variable is lesion_id, so images belonging
    # to the same lesion cannot cross train/validation/test.
    # ---------------------------------------------------------

    sgkf = StratifiedGroupKFold(
        n_splits=20,
        shuffle=True,
        random_state=RANDOM_SEED,
    )

    fold_assignments = np.empty(len(df), dtype=int)

    for fold_number, (_, fold_indices) in enumerate(
        sgkf.split(
            df,
            y=df["dx"],
            groups=df["lesion_id"],
        )
    ):
        fold_assignments[fold_indices] = fold_number

    train_folds = set(range(0, 14))
    val_folds = set(range(14, 17))
    test_folds = set(range(17, 20))

    train_df = df[[
        fold in train_folds for fold in fold_assignments
    ]].copy()

    val_df = df[[
        fold in val_folds for fold in fold_assignments
    ]].copy()

    test_df = df[[
        fold in test_folds for fold in fold_assignments
    ]].copy()

    # ---------------------------------------------------------
    # VERIFY ALL DATA WAS ASSIGNED
    # ---------------------------------------------------------

    total = len(train_df) + len(val_df) + len(test_df)

    if total != len(df):
        raise RuntimeError(
            f"Split error: {total} images assigned out of {len(df)}."
        )

    # ---------------------------------------------------------
    # VERIFY NO LESION LEAKAGE
    # ---------------------------------------------------------

    train_lesions = set(train_df["lesion_id"])
    val_lesions = set(val_df["lesion_id"])
    test_lesions = set(test_df["lesion_id"])

    train_val_overlap = train_lesions & val_lesions
    train_test_overlap = train_lesions & test_lesions
    val_test_overlap = val_lesions & test_lesions

    if train_val_overlap or train_test_overlap or val_test_overlap:
        raise RuntimeError(
            "Lesion leakage detected between train/validation/test."
        )

    # ---------------------------------------------------------
    # SAVE SPLITS
    # ---------------------------------------------------------

    train_df.to_csv(SPLIT_DIR / "train.csv", index=False)
    val_df.to_csv(SPLIT_DIR / "val.csv", index=False)
    test_df.to_csv(SPLIT_DIR / "test.csv", index=False)

    # ---------------------------------------------------------
    # CLASS WEIGHTS
    # ---------------------------------------------------------

    weights = compute_class_weight(
        class_weight="balanced",
        classes=np.array(CLASS_NAMES),
        y=train_df["dx"].values,
    )

    class_weights = {
        i: float(weight)
        for i, weight in enumerate(weights)
    }

    with open(MODEL_DIR / "class_weights.json", "w") as f:
        json.dump(class_weights, f, indent=2)

    with open(MODEL_DIR / "class_names.json", "w") as f:
        json.dump(CLASS_NAMES, f, indent=2)

    # ---------------------------------------------------------
    # REPORT
    # ---------------------------------------------------------

    print("\n" + "=" * 60)
    print("LESION-AWARE DATA SPLIT")
    print("=" * 60)

    print(f"\nTrain:      {len(train_df)} images")
    print(f"Validation: {len(val_df)} images")
    print(f"Test:       {len(test_df)} images")
    print(f"Total:      {total} images")

    print("\nPercentages:")
    print(f"Train:      {len(train_df) / total * 100:.2f}%")
    print(f"Validation: {len(val_df) / total * 100:.2f}%")
    print(f"Test:       {len(test_df) / total * 100:.2f}%")

    print("\nUnique lesions:")
    print(f"Train:      {train_df['lesion_id'].nunique()}")
    print(f"Validation: {val_df['lesion_id'].nunique()}")
    print(f"Test:       {test_df['lesion_id'].nunique()}")

    print("\nTrain class distribution:")
    print(train_df["dx"].value_counts())

    print("\nValidation class distribution:")
    print(val_df["dx"].value_counts())

    print("\nTest class distribution:")
    print(test_df["dx"].value_counts())

    print("\nLesion overlap:")
    print(f"Train ∩ Validation: {len(train_val_overlap)}")
    print(f"Train ∩ Test:       {len(train_test_overlap)}")
    print(f"Validation ∩ Test:  {len(val_test_overlap)}")

    print("\nClass weights:")
    for class_index, weight in class_weights.items():
        print(f"{CLASS_NAMES[class_index]:6s}: {weight:.4f}")

    print("\nSaved:")
    print(SPLIT_DIR / "train.csv")
    print(SPLIT_DIR / "val.csv")
    print(SPLIT_DIR / "test.csv")
    print(MODEL_DIR / "class_weights.json")
    print(MODEL_DIR / "class_names.json")


if __name__ == "__main__":
    main()
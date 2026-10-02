
import pandas as pd
import matplotlib.pyplot as plt

from config import METADATA_PATH, FIGURE_DIR


CLASS_LABELS = {
    "akiec": "Actinic keratoses",
    "bcc": "Basal cell carcinoma",
    "bkl": "Benign keratosis-like lesions",
    "df": "Dermatofibroma",
    "mel": "Melanoma",
    "nv": "Melanocytic nevi",
    "vasc": "Vascular lesions",
}


def main():
    metadata = pd.read_csv(METADATA_PATH)

    class_counts = (
        metadata["dx"]
        .value_counts()
        .reindex(CLASS_LABELS.keys())
    )

    labels = [
        f"{code.upper()}\n{CLASS_LABELS[code]}"
        for code in class_counts.index
    ]

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(12, 7))

    bars = plt.bar(labels, class_counts.values)

    plt.xlabel("Lesion Class")
    plt.ylabel("Number of Images")
    plt.title("HAM10000 Class Distribution")

    plt.xticks(rotation=20, ha="right")

    for bar, count in zip(bars, class_counts.values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            str(count),
            ha="center",
            va="bottom",
        )

    plt.tight_layout()

    output_path = FIGURE_DIR / "class_distribution.png"
    plt.savefig(output_path, dpi=300)
    plt.close()

    print("Dataset visualization generated successfully.")
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()

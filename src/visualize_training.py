
import json
from pathlib import Path

import matplotlib.pyplot as plt

from config import MODEL_DIR, FIGURE_DIR


HISTORY_PATH = MODEL_DIR / "training_history.json"


def load_history():
    with open(HISTORY_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


def combine_history(history, metric):
    """Return a metric from flat history or concatenate both training stages."""
    if metric in history:
        return history[metric]

    values = []
    for stage in ("stage1", "stage2"):
        values.extend(history.get(stage, {}).get(metric, []))
    return values


def main():
    history = load_history()

    accuracy = combine_history(history, "accuracy")
    val_accuracy = combine_history(history, "val_accuracy")
    loss = combine_history(history, "loss")
    val_loss = combine_history(history, "val_loss")

    if not accuracy or not loss:
        raise ValueError(
            f"No training metrics found in {HISTORY_PATH}. "
            "Expected flat metrics or stage1/stage2 history."
        )

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    epochs = range(1, len(accuracy) + 1)

    # Accuracy
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, accuracy, label="Training Accuracy")
    plt.plot(epochs, val_accuracy, label="Validation Accuracy")

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("SkinSight Training and Validation Accuracy")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    accuracy_path = FIGURE_DIR / "training_accuracy.png"
    plt.savefig(accuracy_path, dpi=300)
    plt.close()

    # Loss
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, loss, label="Training Loss")
    plt.plot(epochs, val_loss, label="Validation Loss")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("SkinSight Training and Validation Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    loss_path = FIGURE_DIR / "training_loss.png"
    plt.savefig(loss_path, dpi=300)
    plt.close()

    print("Training visualizations generated successfully.")
    print(f"Saved: {accuracy_path}")
    print(f"Saved: {loss_path}")


if __name__ == "__main__":
    main()

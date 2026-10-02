from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
IMAGE_DIR = RAW_DIR / "images"
SPLIT_DIR = DATA_DIR / "splits"

MODEL_DIR = ROOT_DIR / "models"
OUTPUT_DIR = ROOT_DIR / "outputs"
FIGURE_DIR = OUTPUT_DIR / "figures"
GRADCAM_DIR = OUTPUT_DIR / "gradcam"

METADATA_PATH = RAW_DIR / "HAM10000_metadata.csv"
MODEL_PATH = MODEL_DIR / "skinsight_efficientnetb0.keras"
CLASS_NAMES_PATH = MODEL_DIR / "class_names.json"

IMG_SIZE = (224, 224)
NUM_CLASSES = 7
BATCH_SIZE = 32
RANDOM_SEED = 42

CLASS_NAMES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]

for directory in [SPLIT_DIR, MODEL_DIR, FIGURE_DIR, GRADCAM_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

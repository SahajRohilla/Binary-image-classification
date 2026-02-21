"""
data_preprocessing.py
---------------------
Downloads the Cats vs Dogs dataset (or uses a local copy),
pre-processes images to 224x224 RGB, and splits into
train / validation / test sets saving them to data/processed/.

Usage:
    python src/data_preprocessing.py
"""

import os
import sys
import shutil
import random
import yaml
import logging
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

# ── Logging ────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

# ── Load config ────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
with open(ROOT / "params.yaml") as f:
    cfg = yaml.safe_load(f)

DATA_CFG     = cfg["data"]
RAW_DIR      = ROOT / DATA_CFG["raw_dir"]
PROCESSED_DIR = ROOT / DATA_CFG["processed_dir"]
IMAGE_SIZE   = DATA_CFG["image_size"]
TRAIN_SPLIT  = DATA_CFG["train_split"]
VAL_SPLIT    = DATA_CFG["val_split"]
TEST_SPLIT   = DATA_CFG["test_split"]
SEED         = cfg["training"]["seed"]

CLASSES = ["cat", "dog"]
SPLITS  = ["train", "val", "test"]


def download_kaggle_dataset() -> None:
    """Download the Cats-vs-Dogs dataset from Kaggle if not already present."""
    raw_cats = RAW_DIR / "Cat"
    raw_dogs = RAW_DIR / "Dog"
    if raw_cats.exists() and raw_dogs.exists():
        log.info("Raw dataset already present – skipping download.")
        return

    log.info("Attempting Kaggle download (requires ~/.kaggle/kaggle.json) …")
    try:
        import kaggle  # noqa: F401
        RAW_DIR.mkdir(parents=True, exist_ok=True)
        os.system(
            f"kaggle datasets download -d shaunthesheep/microsoft-catsvsdogs-dataset "
            f"-p {RAW_DIR} --unzip"
        )
        log.info("Kaggle download complete.")
    except Exception as exc:
        log.warning(
            f"Kaggle download failed ({exc}). "
            "Please manually place 'Cat/' and 'Dog/' folders inside data/raw/ "
            "or unzip the dataset there, then re-run."
        )
        sys.exit(1)


def load_image_paths(class_name: str) -> list[Path]:
    """Return all valid image paths for a given class folder."""
    folder = RAW_DIR / class_name.capitalize()
    if not folder.exists():
        folder = RAW_DIR / class_name.lower()
    paths = sorted(
        p for p in folder.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    log.info(f"  {class_name}: {len(paths)} images found.")
    return paths


def preprocess_image(src: Path, dst: Path) -> bool:
    """Resize to IMAGE_SIZE×IMAGE_SIZE RGB and save as JPEG. Returns success flag."""
    try:
        with Image.open(src) as img:
            img = img.convert("RGB")
            img = img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.LANCZOS)
            dst.parent.mkdir(parents=True, exist_ok=True)
            img.save(dst, format="JPEG", quality=90)
        return True
    except Exception as exc:
        log.debug(f"Skipping {src.name}: {exc}")
        return False


def split_paths(paths: list[Path]) -> tuple[list, list, list]:
    """Shuffle and split paths into train / val / test lists."""
    random.seed(SEED)
    shuffled = paths.copy()
    random.shuffle(shuffled)
    n = len(shuffled)
    n_train = int(n * TRAIN_SPLIT)
    n_val   = int(n * VAL_SPLIT)
    return (
        shuffled[:n_train],
        shuffled[n_train : n_train + n_val],
        shuffled[n_train + n_val :],
    )


def process_class(class_name: str) -> dict:
    """Preprocess and split images for one class. Returns count dict."""
    log.info(f"Processing class: {class_name}")
    paths = load_image_paths(class_name)
    train_p, val_p, test_p = split_paths(paths)
    counts = {}
    for split_name, split_paths_list in zip(SPLITS, [train_p, val_p, test_p]):
        ok = 0
        for src in tqdm(split_paths_list, desc=f"  {split_name:5s}", leave=False):
            dst = PROCESSED_DIR / split_name / class_name.lower() / src.name
            if preprocess_image(src, dst):
                ok += 1
        counts[split_name] = ok
        log.info(f"    {split_name}: {ok} images saved.")
    return counts


def verify_dataset() -> None:
    """Print final counts per split and class."""
    log.info("\n── Dataset Summary ──────────────────────────────")
    for split in SPLITS:
        for cls in CLASSES:
            p = PROCESSED_DIR / split / cls
            n = len(list(p.glob("*.jpg"))) if p.exists() else 0
            log.info(f"  {split:5s}/{cls}: {n}")
    log.info("─────────────────────────────────────────────────\n")


def main() -> None:
    log.info("=== Data Preprocessing Pipeline ===")
    download_kaggle_dataset()

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    for cls in CLASSES:
        process_class(cls)

    verify_dataset()
    log.info("Pre-processing complete. Processed data saved to: %s", PROCESSED_DIR)


if __name__ == "__main__":
    main()

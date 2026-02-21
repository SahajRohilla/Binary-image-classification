"""
create_sample_data.py
---------------------
Creates a tiny SYNTHETIC dataset of 224×224 RGB images so the full
pipeline (train → evaluate → MLflow) can be tested immediately,
WITHOUT needing the real Kaggle download.

  Cat images  → solid orange-ish pixels  (easy to distinguish)
  Dog images  → solid blue-ish pixels

For the real assignment submission, replace data/raw/ with the actual
Kaggle Cats-vs-Dogs images and re-run data_preprocessing.py.

Usage:
    python src/create_sample_data.py
    python src/create_sample_data.py --n_train 200 --n_val 30 --n_test 30
"""

import argparse
import random
import sys
import yaml
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
with open(ROOT / "params.yaml") as f:
    cfg = yaml.safe_load(f)

IMAGE_SIZE    = cfg["data"]["image_size"]
PROCESSED_DIR = ROOT / cfg["data"]["processed_dir"]

# RGB base colours (+noise) for each class
CLASS_COLORS = {
    "cat": (210, 120,  50),   # warm orange
    "dog": ( 50, 100, 200),  # cool blue
}


def make_image(color_base: tuple, size: int = IMAGE_SIZE) -> Image.Image:
    """Create a solid-colour image with Gaussian noise for variety."""
    r, g, b = color_base
    noise = np.random.randint(-40, 40, (size, size, 3), dtype=np.int16)
    arr = np.clip(
        np.array([[[r, g, b]]], dtype=np.int16) + noise, 0, 255
    ).astype(np.uint8)
    return Image.fromarray(arr, mode="RGB")


def generate_split(
    class_name: str,
    split: str,
    n: int,
) -> None:
    dest = PROCESSED_DIR / split / class_name
    dest.mkdir(parents=True, exist_ok=True)
    color = CLASS_COLORS[class_name]
    for i in tqdm(range(n), desc=f"  {split}/{class_name}", leave=False):
        img = make_image(color)
        img.save(dest / f"{class_name}_{split}_{i:04d}.jpg", "JPEG")


def main(n_train: int, n_val: int, n_test: int) -> None:
    print("=" * 55)
    print(" Creating SYNTHETIC dataset for pipeline testing")
    print("=" * 55)
    print(f"  Image size : {IMAGE_SIZE}×{IMAGE_SIZE} RGB")
    print(f"  Train      : {n_train} per class  ({n_train*2} total)")
    print(f"  Val        : {n_val}  per class  ({n_val*2} total)")
    print(f"  Test       : {n_test}  per class  ({n_test*2} total)")
    print()

    for cls in CLASS_COLORS:
        generate_split(cls, "train", n_train)
        generate_split(cls, "val",   n_val)
        generate_split(cls, "test",  n_test)

    print()
    print("✅ Synthetic dataset created successfully!")
    print(f"   Location: {PROCESSED_DIR}")
    print()
    print("➡  Now run:  python src/train.py")
    print()
    print("NOTE: Replace with real Kaggle data before final submission.")
    print("      Real dataset: https://www.kaggle.com/datasets/shaunthesheep/microsoft-catsvsdogs-dataset")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create synthetic Cats vs Dogs dataset")
    parser.add_argument("--n_train", type=int, default=300, help="Images per class in train set")
    parser.add_argument("--n_val",   type=int, default=50,  help="Images per class in val set")
    parser.add_argument("--n_test",  type=int, default=50,  help="Images per class in test set")
    args = parser.parse_args()
    main(args.n_train, args.n_val, args.n_test)

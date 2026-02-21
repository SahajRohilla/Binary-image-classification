"""
train.py
--------
Train the CNN / Logistic model on the Cats vs Dogs dataset with:
  - Data augmentation (when enabled in params.yaml)
  - Early stopping & learning rate scheduling
  - Full MLflow experiment tracking:
      * Parameters, metrics (per epoch and final)
      * Artifacts: model weights (.h5), loss curve, accuracy curve, confusion matrix

Usage:
    python src/train.py
    python src/train.py --arch logistic --epochs 5
"""

import argparse
import logging
import os
import sys
import yaml
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")          # headless backend
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import mlflow.tensorflow
from sklearn.metrics import confusion_matrix, classification_report
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import (
    EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
)

# ── Project paths ───────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from model import build_model  # noqa: E402 (imported after sys.path update)

# ── Logging ─────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

# ── Config ──────────────────────────────────────────────────────────────────
with open(ROOT / "params.yaml") as f:
    cfg = yaml.safe_load(f)

DC  = cfg["data"]
MC  = cfg["model"]
TC  = cfg["training"]
MLC = cfg["mlflow"]

PROCESSED_DIR = ROOT / DC["processed_dir"]
MODELS_DIR    = ROOT / "models"
REPORTS_DIR   = ROOT / "reports"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_SIZE  = DC["image_size"]
BATCH_SIZE  = DC["batch_size"]
AUGMENT     = DC["augmentation"]
CLASSES     = ["cat", "dog"]               # alphabetical → Keras class_indices


# ── Data generators ─────────────────────────────────────────────────────────

def _check_data_exists() -> None:
    """Raise a clear error if processed data is missing or empty."""
    for split in ("train", "val", "test"):
        for cls in CLASSES:
            folder = PROCESSED_DIR / split / cls
            if not folder.exists() or not any(folder.iterdir()):
                raise RuntimeError(
                    f"\n\n{'='*60}\n"
                    f"  ❌ No images found in: {folder}\n\n"
                    f"  The processed dataset does not exist yet.\n"
                    f"  Run ONE of the following first:\n\n"
                    f"  Option A – Quick synthetic test dataset:\n"
                    f"    python src/create_sample_data.py\n\n"
                    f"  Option B – Real Kaggle dataset:\n"
                    f"    1. Place Cat/ and Dog/ folders in data/raw/\n"
                    f"    2. python src/data_preprocessing.py\n"
                    f"{'='*60}\n"
                )


def make_generators():
    """Build train / val / test Keras ImageDataGenerators."""
    _check_data_exists()
    if AUGMENT:
        train_datagen = ImageDataGenerator(
            rescale=1.0 / 255,
            rotation_range=20,
            width_shift_range=0.15,
            height_shift_range=0.15,
            horizontal_flip=True,
            zoom_range=0.15,
            shear_range=0.1,
            fill_mode="nearest",
        )
        log.info("Data augmentation ENABLED.")
    else:
        train_datagen = ImageDataGenerator(rescale=1.0 / 255)

    val_test_datagen = ImageDataGenerator(rescale=1.0 / 255)

    common_kwargs = dict(
        target_size=(IMAGE_SIZE, IMAGE_SIZE),
        batch_size=BATCH_SIZE,
        class_mode="binary",
        classes=CLASSES,
        shuffle=True,
        seed=TC["seed"],
    )

    train_gen = train_datagen.flow_from_directory(
        str(PROCESSED_DIR / "train"), **common_kwargs
    )
    val_gen = val_test_datagen.flow_from_directory(
        str(PROCESSED_DIR / "val"), **{**common_kwargs, "shuffle": False}
    )
    test_gen = val_test_datagen.flow_from_directory(
        str(PROCESSED_DIR / "test"), **{**common_kwargs, "shuffle": False}
    )

    log.info(
        "Generators ready — train: %d | val: %d | test: %d",
        train_gen.samples, val_gen.samples, test_gen.samples,
    )
    return train_gen, val_gen, test_gen


# ── Callbacks ───────────────────────────────────────────────────────────────

def make_callbacks(checkpoint_path: Path):
    return [
        EarlyStopping(
            monitor="val_loss",
            patience=TC["early_stopping_patience"],
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1,
        ),
        ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
    ]


# ── Plot helpers ────────────────────────────────────────────────────────────

def plot_training_curves(history, save_dir: Path):
    """Save loss and accuracy curves as PNG files."""
    epochs = range(1, len(history.history["loss"]) + 1)

    # Loss
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, history.history["loss"],     "b-o", label="Train Loss")
    ax.plot(epochs, history.history["val_loss"], "r-o", label="Val Loss")
    ax.set_title("Training vs Validation Loss", fontsize=14)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
    ax.legend(); ax.grid(True, alpha=0.3)
    loss_path = save_dir / "loss_curve.png"
    fig.savefig(loss_path, dpi=120, bbox_inches="tight")
    plt.close(fig)

    # Accuracy
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, history.history["accuracy"],     "b-o", label="Train Acc")
    ax.plot(epochs, history.history["val_accuracy"], "r-o", label="Val Acc")
    ax.set_title("Training vs Validation Accuracy", fontsize=14)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Accuracy")
    ax.legend(); ax.grid(True, alpha=0.3)
    acc_path = save_dir / "accuracy_curve.png"
    fig.savefig(acc_path, dpi=120, bbox_inches="tight")
    plt.close(fig)

    log.info("Saved training curves → %s", save_dir)
    return loss_path, acc_path


def plot_confusion_matrix(y_true, y_pred, save_dir: Path):
    """Save a labelled confusion matrix PNG."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=CLASSES, yticklabels=CLASSES, ax=ax,
    )
    ax.set_title("Confusion Matrix", fontsize=14)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    cm_path = save_dir / "confusion_matrix.png"
    fig.savefig(cm_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved confusion matrix → %s", cm_path)
    return cm_path


# ── Main training loop ──────────────────────────────────────────────────────

def train(arch: str, epochs: int):
    # ── MLflow tracking URI ──
    # Use .as_uri() to produce file:///E:/... format on Windows
    # (bare Windows paths like E:\... are misread as URI schemes by MLflow)
    mlflow.set_tracking_uri((ROOT / MLC["tracking_uri"]).as_uri())
    mlflow.set_experiment(MLC["experiment_name"])

    run_name = f"{MLC['run_name']}_{arch}"

    with mlflow.start_run(run_name=run_name) as run:
        log.info("MLflow run started: %s", run.info.run_id)

        # ── Log all params ──
        mlflow.log_params({
            "architecture":    arch,
            "image_size":      IMAGE_SIZE,
            "batch_size":      BATCH_SIZE,
            "epochs":          epochs,
            "learning_rate":   TC["learning_rate"],
            "optimizer":       TC["optimizer"],
            "dropout_rate":    MC["dropout_rate"],
            "dense_units":     MC["dense_units"],
            "augmentation":    AUGMENT,
            "early_stopping_patience": TC["early_stopping_patience"],
            "train_split":     DC["train_split"],
            "val_split":       DC["val_split"],
            "test_split":      DC["test_split"],
        })

        # ── Data ──
        train_gen, val_gen, test_gen = make_generators()

        # ── Model ──
        model = build_model(
            architecture=arch,
            learning_rate=TC["learning_rate"],
            optimizer=TC["optimizer"],
        )

        checkpoint_path = MODELS_DIR / f"best_{arch}.h5"
        callbacks       = make_callbacks(checkpoint_path)

        # ── Train ──
        log.info("Starting training for %d epochs …", epochs)
        history = model.fit(
            train_gen,
            epochs=epochs,
            validation_data=val_gen,
            callbacks=callbacks,
            verbose=1,
        )

        # ── Log per-epoch metrics ──
        for epoch_idx, (tr_loss, tr_acc, vl_loss, vl_acc) in enumerate(zip(
            history.history["loss"],
            history.history["accuracy"],
            history.history["val_loss"],
            history.history["val_accuracy"],
        ), start=1):
            mlflow.log_metrics({
                "train_loss":     tr_loss,
                "train_accuracy": tr_acc,
                "val_loss":       vl_loss,
                "val_accuracy":   vl_acc,
            }, step=epoch_idx)

        # ── Evaluate on test set ──
        log.info("Evaluating on test set …")
        test_gen.reset()
        results = model.evaluate(test_gen, verbose=0)
        metric_names = model.metrics_names
        test_metrics = dict(zip([f"test_{n}" for n in metric_names], results))
        mlflow.log_metrics(test_metrics)
        log.info("Test metrics: %s", test_metrics)

        # ── Confusion matrix ──
        test_gen.reset()
        y_pred_probs = model.predict(test_gen, verbose=0).ravel()
        y_pred       = (y_pred_probs >= 0.5).astype(int)
        y_true       = test_gen.classes

        report = classification_report(y_true, y_pred, target_names=CLASSES, output_dict=True)
        mlflow.log_dict(report, "classification_report.json")

        # ── Save artifacts (plots + model) ──
        loss_path, acc_path = plot_training_curves(history, REPORTS_DIR)
        cm_path             = plot_confusion_matrix(y_true, y_pred, REPORTS_DIR)

        mlflow.log_artifact(str(loss_path),   artifact_path="plots")
        mlflow.log_artifact(str(acc_path),    artifact_path="plots")
        mlflow.log_artifact(str(cm_path),     artifact_path="plots")

        # Save best model (already written by ModelCheckpoint)
        final_model_path = MODELS_DIR / f"final_{arch}_model.h5"
        model.save(str(final_model_path))
        mlflow.log_artifact(str(final_model_path), artifact_path="model")

        # Log model via MLflow tensorflow flavour for registry
        mlflow.tensorflow.log_model(model, artifact_path="mlflow_model")

        log.info("=" * 60)
        log.info("Run complete! Artifacts saved.")
        log.info("  Model     : %s", final_model_path)
        log.info("  MLflow UI : mlflow ui --backend-store-uri %s/mlruns", ROOT)
        log.info("=" * 60)

        return test_metrics


# ── CLI ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Cats vs Dogs classifier")
    parser.add_argument("--arch",   default=MC["architecture"], choices=["cnn", "logistic"])
    parser.add_argument("--epochs", default=TC["epochs"], type=int)
    args = parser.parse_args()

    train(arch=args.arch, epochs=args.epochs)

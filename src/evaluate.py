"""
evaluate.py
-----------
Load a saved model and produce a full evaluation report on the test set:
  - Accuracy, Precision, Recall, F1, AUC
  - Confusion matrix (saved + displayed)
  - Per-class classification report
  - Logs everything to the last MLflow run (or starts a new evaluation run)

Usage:
    python src/evaluate.py
    python src/evaluate.py --model models/final_cnn_model.h5
"""

import argparse
import logging
import sys
import yaml
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
# mlflow.keras removed (deprecated); using mlflow.set_tracking_uri directly
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve,
    ConfusionMatrixDisplay,
)
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

with open(ROOT / "params.yaml") as f:
    cfg = yaml.safe_load(f)

DC    = cfg["data"]
MLC   = cfg["mlflow"]
CLASSES = ["cat", "dog"]


def load_test_generator():
    datagen = ImageDataGenerator(rescale=1.0 / 255)
    gen = datagen.flow_from_directory(
        str(ROOT / DC["processed_dir"] / "test"),
        target_size=(DC["image_size"], DC["image_size"]),
        batch_size=DC["batch_size"],
        class_mode="binary",
        classes=CLASSES,
        shuffle=False,
    )
    log.info("Test generator: %d samples.", gen.samples)
    return gen


def evaluate(model_path: Path):
    log.info("Loading model from: %s", model_path)
    model = tf.keras.models.load_model(str(model_path))

    test_gen = load_test_generator()
    test_gen.reset()

    # ── Predictions ──
    y_probs = model.predict(test_gen, verbose=1).ravel()
    y_pred  = (y_probs >= 0.5).astype(int)
    y_true  = test_gen.classes

    # ── Metrics ──
    auc_score = roc_auc_score(y_true, y_probs)
    report    = classification_report(y_true, y_pred, target_names=CLASSES, output_dict=True)
    acc       = report["accuracy"]
    prec      = report["weighted avg"]["precision"]
    rec       = report["weighted avg"]["recall"]
    f1        = report["weighted avg"]["f1-score"]

    log.info("\n%s", classification_report(y_true, y_pred, target_names=CLASSES))
    log.info("ROC-AUC: %.4f", auc_score)

    # ── Confusion Matrix ──
    reports_dir = ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=CLASSES, yticklabels=CLASSES, ax=ax,
    )
    ax.set_title("Test Set Confusion Matrix", fontsize=14)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    cm_path = reports_dir / "eval_confusion_matrix.png"
    fig.savefig(cm_path, dpi=120, bbox_inches="tight")
    plt.close(fig)

    # ── ROC Curve ──
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(fpr, tpr, "b-", lw=2, label=f"ROC (AUC = {auc_score:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_title("ROC Curve – Cats vs Dogs", fontsize=14)
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.legend(); ax.grid(True, alpha=0.3)
    roc_path = reports_dir / "roc_curve.png"
    fig.savefig(roc_path, dpi=120, bbox_inches="tight")
    plt.close(fig)

    log.info("Evaluation plots saved to: %s", reports_dir)

    # ── Log to MLflow ──
    # Use .as_uri() to produce file:///E:/... format on Windows
    mlflow.set_tracking_uri((ROOT / MLC["tracking_uri"]).as_uri())
    mlflow.set_experiment(MLC["experiment_name"])

    with mlflow.start_run(run_name="evaluation"):
        mlflow.log_metrics({
            "test_accuracy":  acc,
            "test_precision": prec,
            "test_recall":    rec,
            "test_f1":        f1,
            "test_auc":       auc_score,
        })
        mlflow.log_dict(report, "classification_report.json")
        mlflow.log_artifact(str(cm_path),  artifact_path="eval_plots")
        mlflow.log_artifact(str(roc_path), artifact_path="eval_plots")

    log.info(
        "\nSummary ─────────────────────────────────────\n"
        "  Accuracy : %.4f\n  Precision: %.4f\n  Recall   : %.4f\n"
        "  F1-Score : %.4f\n  ROC-AUC  : %.4f\n"
        "─────────────────────────────────────────────",
        acc, prec, rec, f1, auc_score,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Cats vs Dogs model")
    parser.add_argument(
        "--model",
        default=str(ROOT / "models" / "final_cnn_model.keras"),
        help="Path to saved .keras model",
    )
    args = parser.parse_args()
    evaluate(Path(args.model))

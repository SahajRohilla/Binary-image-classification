"""
model.py
--------
Defines the CNN and Logistic Regression baseline models for
binary image classification (Cats vs Dogs).

Usage:
    from src.model import build_model
    model = build_model(architecture="cnn")
"""

import yaml
import logging
from pathlib import Path

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
with open(ROOT / "params.yaml") as f:
    cfg = yaml.safe_load(f)

MODEL_CFG    = cfg["model"]
IMAGE_SIZE   = cfg["data"]["image_size"]
NUM_CLASSES  = MODEL_CFG["num_classes"]
DROPOUT_RATE = MODEL_CFG["dropout_rate"]
DENSE_UNITS  = MODEL_CFG["dense_units"]


# ── CNN Architecture ────────────────────────────────────────────────────────

def build_cnn(input_shape: tuple = (224, 224, 3)) -> keras.Model:
    """
    Baseline CNN:
      3 × [Conv2D → BatchNorm → MaxPool]  feature extractor
      GlobalAveragePooling  →  Dense(128, relu)  →  Dropout  →  Dense(1, sigmoid)

    Returns a compiled Keras Model.
    """
    inputs = keras.Input(shape=input_shape, name="input_image")

    # ── Block 1 ──
    x = layers.Conv2D(32, (3, 3), padding="same", activation="relu",
                      kernel_regularizer=regularizers.l2(1e-4), name="conv1_1")(inputs)
    x = layers.BatchNormalization(name="bn1")(x)
    x = layers.MaxPooling2D((2, 2), name="pool1")(x)

    # ── Block 2 ──
    x = layers.Conv2D(64, (3, 3), padding="same", activation="relu",
                      kernel_regularizer=regularizers.l2(1e-4), name="conv2_1")(x)
    x = layers.BatchNormalization(name="bn2")(x)
    x = layers.MaxPooling2D((2, 2), name="pool2")(x)

    # ── Block 3 ──
    x = layers.Conv2D(128, (3, 3), padding="same", activation="relu",
                      kernel_regularizer=regularizers.l2(1e-4), name="conv3_1")(x)
    x = layers.BatchNormalization(name="bn3")(x)
    x = layers.MaxPooling2D((2, 2), name="pool3")(x)

    # ── Head ──
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dense(DENSE_UNITS, activation="relu",
                     kernel_regularizer=regularizers.l2(1e-4), name="dense1")(x)
    x = layers.Dropout(DROPOUT_RATE, name="dropout")(x)
    outputs = layers.Dense(1, activation="sigmoid", name="output")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="baseline_cnn")
    return model


# ── Logistic Regression (flattened pixels) ─────────────────────────────────

def build_logistic(input_shape: tuple = (224, 224, 3)) -> keras.Model:
    """
    Logistic Regression baseline:
      Flatten → Dense(1, sigmoid)

    Extremely fast to train; useful as a sanity-check baseline.
    """
    inputs  = keras.Input(shape=input_shape, name="input_image")
    x       = layers.Flatten(name="flatten")(inputs)
    outputs = layers.Dense(1, activation="sigmoid", name="output")(x)
    model   = keras.Model(inputs=inputs, outputs=outputs, name="logistic_regression")
    return model


# ── Factory ────────────────────────────────────────────────────────────────

def build_model(
    architecture: str = "cnn",
    learning_rate: float = 0.001,
    optimizer: str = "adam",
) -> keras.Model:
    """
    Build and compile a model.

    Args:
        architecture: "cnn" or "logistic"
        learning_rate: optimizer learning rate
        optimizer: "adam" or "sgd"

    Returns:
        Compiled Keras Model
    """
    input_shape = (IMAGE_SIZE, IMAGE_SIZE, 3)
    arch = architecture.lower()

    if arch == "cnn":
        model = build_cnn(input_shape)
    elif arch == "logistic":
        model = build_logistic(input_shape)
    else:
        raise ValueError(f"Unknown architecture: {architecture!r}. Choose 'cnn' or 'logistic'.")

    # ── Optimizer ──
    if optimizer.lower() == "adam":
        opt = keras.optimizers.Adam(learning_rate=learning_rate)
    elif optimizer.lower() == "sgd":
        opt = keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer!r}. Choose 'adam' or 'sgd'.")

    model.compile(
        optimizer=opt,
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            keras.metrics.AUC(name="auc"),
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
        ],
    )

    log.info("Model '%s' built with %d parameters.", model.name, model.count_params())
    model.summary(print_fn=log.info)
    return model


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    m = build_model(architecture="cnn")
    print(f"\nTotal parameters: {m.count_params():,}")

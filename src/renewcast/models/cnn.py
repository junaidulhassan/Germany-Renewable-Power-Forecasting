"""CNN regressors that map one weather frame to a national generation value."""

from __future__ import annotations

import keras
from keras import layers

from renewcast.config import ModelConfig, TrainingConfig


def build_cnn(input_shape: tuple[int, int, int], cfg: ModelConfig) -> keras.Model:
    """Build an uncompiled model. ``input_shape`` is (height, width, channels)."""
    if cfg.architecture == "baseline":
        return _baseline(input_shape, cfg)
    return _compact(input_shape, cfg)


def _baseline(input_shape: tuple[int, int, int], cfg: ModelConfig) -> keras.Model:
    """Notebook architecture: Conv/Pool blocks -> Flatten -> Dense -> Dropout -> Dense(1).

    With a 469x360 input and filters [32, 64] this has ~82.9M parameters, almost all in the
    first Dense layer, which is why the `compact` variant exists.
    """
    stack: list[keras.layers.Layer] = [layers.Input(shape=input_shape)]
    for filters in cfg.filters:
        stack += [layers.Conv2D(filters, (3, 3), activation="relu"), layers.MaxPooling2D((2, 2))]
    stack += [
        layers.Flatten(),
        layers.Dense(cfg.dense_units, activation="relu"),
        layers.Dropout(cfg.dropout),
        layers.Dense(1, activation="linear", dtype="float32"),
    ]
    return keras.Sequential(stack, name="cnn_baseline")


def _compact(input_shape: tuple[int, int, int], cfg: ModelConfig) -> keras.Model:
    """Conv -> BatchNorm -> ReLU -> Pool blocks, with an extra stride-2 block to shrink the map."""
    stack: list[keras.layers.Layer] = [layers.Input(shape=input_shape)]
    for filters in cfg.filters:
        stack += [
            layers.Conv2D(filters, (3, 3), padding="same", use_bias=False),
            layers.BatchNormalization(),
            layers.Activation("relu"),
            layers.MaxPooling2D((2, 2)),
        ]
    stack += [
        layers.Conv2D(cfg.filters[-1], (3, 3), strides=2, padding="same", activation="relu"),
        layers.Flatten(),
        layers.Dropout(cfg.dropout),
        layers.Dense(cfg.dense_units, activation="relu"),
        layers.Dense(1, activation="linear", dtype="float32"),
    ]
    return keras.Sequential(stack, name="cnn_compact")


def compile_model(model: keras.Model, cfg: TrainingConfig) -> keras.Model:
    losses = {
        "mse": "mean_squared_error",
        "mae": "mean_absolute_error",
        "huber": keras.losses.Huber(),
    }
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=cfg.learning_rate),
        loss=losses[cfg.loss],
        metrics=["mae"],
    )
    return model

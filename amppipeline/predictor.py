"""CNN-RNN predictor of MIC against *Escherichia coli*.

The architecture follows the published description (Text S1, section S1.2):
two one-dimensional convolutional layers with 64 filters each, followed by a
256-unit LSTM and a fully connected output layer returning the predicted MIC.
The model is trained on log10-transformed MIC values with mean squared error,
the Adam optimizer, a batch size of 64, early stopping, and a maximum of
200 to 500 epochs (section S1.3).

The original study's trained weights were not preserved. This module is a
re-implementation of the described architecture, not a recovered model.
"""

from __future__ import annotations

import json
import os
from typing import Dict, Sequence

from .constants import MAX_LEN

DEFAULT_CONFIG: Dict[str, object] = {
    "max_len": MAX_LEN,
    "conv_filters": 64,
    "conv_layers": 2,
    "lstm_units": 256,
    "learning_rate": 1e-3,
    "batch_size": 64,
}


def _require_tensorflow():
    try:
        import tensorflow as tf  # noqa: F401
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "TensorFlow is required for model training and prediction. "
            "Install it with 'pip install \"tensorflow>=2.11,<2.16\"'."
        ) from exc
    return tf


def _resolved_config(config: Dict[str, object] | None) -> Dict[str, object]:
    resolved = dict(DEFAULT_CONFIG)
    if config:
        unknown = set(config) - set(resolved)
        if unknown:
            raise ValueError(
                "unknown predictor config keys: %s" % ", ".join(sorted(unknown))
            )
        resolved.update(config)
    return resolved


def build_predictor(config: Dict[str, object] | None = None):
    """Build and compile the CNN-RNN MIC predictor.

    Returns ``(model, config)``. The model output is the predicted log10 MIC in
    micromolar units; convert with ``10 ** prediction``.
    """
    _require_tensorflow()
    from tensorflow import keras
    from tensorflow.keras import layers

    config = _resolved_config(config)
    max_len = int(config["max_len"])
    conv_filters = int(config["conv_filters"])
    conv_layers = int(config["conv_layers"])
    lstm_units = int(config["lstm_units"])

    from .constants import ALPHABET

    inputs = keras.Input(shape=(max_len, len(ALPHABET)), name="one_hot")
    hidden = inputs
    for index in range(conv_layers):
        hidden = layers.Conv1D(
            conv_filters,
            kernel_size=5,
            padding="same",
            activation="relu",
            name="conv_%d" % (index + 1),
        )(hidden)
    hidden = layers.LSTM(lstm_units, name="lstm")(hidden)
    outputs = layers.Dense(1, name="predicted_log10_mic")(hidden)

    model = keras.Model(inputs, outputs, name="cnn_rnn_mic_predictor")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=float(config["learning_rate"])),
        loss="mse",
        metrics=["mae"],
    )
    return model, config


def save_predictor(model, config: Dict[str, object], weights_path: str) -> str:
    """Save weights and the architecture config next to each other."""
    directory = os.path.dirname(weights_path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    model.save_weights(weights_path)
    config_path = os.path.splitext(weights_path)[0] + ".config.json"
    with open(config_path, "w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return config_path


def load_predictor(weights_path: str, config_path: str | None = None):
    """Rebuild the predictor from its config and load saved weights."""
    if config_path is None:
        config_path = os.path.splitext(weights_path)[0] + ".config.json"
    with open(config_path, "r", encoding="utf-8") as handle:
        config = json.load(handle)
    model, config = build_predictor(config)
    model.load_weights(weights_path)
    return model, config

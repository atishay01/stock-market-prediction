"""Pure-numpy forward pass for the trained stacked-LSTM forecaster.

Used in production (Render free tier) where full TensorFlow doesn't fit
and TFLite conversion of the Keras 3 LSTM couldn't be made builtin-only.
The weights come from ``src/export_lstm_weights.py``; this module knows
nothing about training and needs nothing beyond numpy.

Architecture hard-coded for our model:
    Input (B, T, F) -> LSTM(64, return_sequences=True)
                    -> LSTM(32)
                    -> Dense(16, relu) -> Dense(1)
Dropout is an identity at inference so we skip it entirely.

Keras's LSTM uses the standard formulation with tanh cell activation and
sigmoid gate activations, gates packed in [i, f, c, o] order along the
last axis of the kernel matrices — matches what Keras's
``layer.get_weights()`` returns, so no re-slicing needed here.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


def _sigmoid(x: np.ndarray) -> np.ndarray:
    # Numerically stable sigmoid — avoids overflow for large negative inputs.
    out = np.empty_like(x)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    exp_x = np.exp(x[~pos])
    out[~pos] = exp_x / (1.0 + exp_x)
    return out


def _lstm_forward(x: np.ndarray, W: np.ndarray, U: np.ndarray, b: np.ndarray,
                  units: int, return_sequences: bool) -> np.ndarray:
    """Roll the Keras LSTM over the time axis of ``x`` (shape (B, T, F))."""
    B, T, _ = x.shape
    h = np.zeros((B, units), dtype=np.float32)
    c = np.zeros((B, units), dtype=np.float32)

    if return_sequences:
        outputs = np.empty((B, T, units), dtype=np.float32)

    for t in range(T):
        # Precompute the whole gate vector in one matmul per step.
        z = x[:, t, :] @ W + h @ U + b  # (B, 4*units)
        i = _sigmoid(z[:, :units])
        f = _sigmoid(z[:, units:2 * units])
        c_tilde = np.tanh(z[:, 2 * units:3 * units])
        o = _sigmoid(z[:, 3 * units:])
        c = f * c + i * c_tilde
        h = o * np.tanh(c)
        if return_sequences:
            outputs[:, t, :] = h

    return outputs if return_sequences else h


class NumpyLSTM:
    """Stacked LSTM + dense head, numpy-only."""

    def __init__(self, weights: dict[str, np.ndarray]):
        self.w = weights
        # Pre-cast once so we can call predict() without per-call conversion.
        self.lstm_units = (
            int(weights["lstm0_units"]),
            int(weights["lstm1_units"]),
        )
        self.return_seq = (
            bool(int(weights["lstm0_return_sequences"])),
            bool(int(weights["lstm1_return_sequences"])),
        )
        self.dense_acts = (
            weights["dense0_act"].item().decode("ascii"),
            weights["dense1_act"].item().decode("ascii"),
        )

    @classmethod
    def load(cls, path: str | Path) -> "NumpyLSTM":
        raw = np.load(path)
        return cls({k: raw[k] for k in raw.files})

    @staticmethod
    def _apply_activation(x: np.ndarray, name: str) -> np.ndarray:
        if name == "relu":
            return np.maximum(x, 0)
        if name == "linear":
            return x
        if name == "tanh":
            return np.tanh(x)
        if name == "sigmoid":
            return _sigmoid(x)
        raise ValueError(f"unsupported dense activation: {name!r}")

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Forward pass. Input shape (B, T, F); output shape (B,)."""
        x = np.ascontiguousarray(x, dtype=np.float32)

        # LSTM stack.
        h = _lstm_forward(
            x,
            self.w["lstm0_W"], self.w["lstm0_U"], self.w["lstm0_b"],
            self.lstm_units[0], self.return_seq[0],
        )
        h = _lstm_forward(
            h,
            self.w["lstm1_W"], self.w["lstm1_U"], self.w["lstm1_b"],
            self.lstm_units[1], self.return_seq[1],
        )

        # Dense head.
        h = self._apply_activation(h @ self.w["dense0_W"] + self.w["dense0_b"],
                                    self.dense_acts[0])
        h = self._apply_activation(h @ self.w["dense1_W"] + self.w["dense1_b"],
                                    self.dense_acts[1])
        return h.flatten()

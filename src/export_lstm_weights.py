"""Extract the trained LSTM's weights into a plain numpy .npz.

Why not TFLite? Keras 3 + TF 2.16 can't convert our LSTM to builtin-only
TFLite (TensorList ops have dynamic element_shape), and the alternative
SELECT_TF_OPS path needs the Flex delegate, which isn't in PyPI's
tflite-runtime. So we go one level lower: dump the weights and do the
forward pass in numpy. Total runtime dep: numpy (already in prod). Total
disk footprint: ~400 KB. Render free tier eats this for breakfast.

Run once after ``src/train.py``:

    python src/export_lstm_weights.py

Produces ``models/lstm_weights.npz``. The Flask app (via ``src/predict.py``
and ``src/lstm_numpy.py``) loads that file and never needs TensorFlow in
production.
"""
from __future__ import annotations

import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import sys
from pathlib import Path

import numpy as np
import tensorflow as tf

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"
SRC_KERAS = MODEL_DIR / "lstm_model.keras"
DST_NPZ = MODEL_DIR / "lstm_weights.npz"


def _split_lstm_kernel(kernel: np.ndarray, units: int):
    """Keras packs LSTM kernels as [i, f, c, o] along the last axis."""
    return {
        "i": kernel[:, :units],
        "f": kernel[:, units:2 * units],
        "c": kernel[:, 2 * units:3 * units],
        "o": kernel[:, 3 * units:],
    }


def main() -> None:
    if not SRC_KERAS.exists():
        print(f"ERROR: {SRC_KERAS} not found — train the LSTM first via src/train.py")
        sys.exit(1)

    print(f"Loading Keras model from {SRC_KERAS} ...")
    model = tf.keras.models.load_model(SRC_KERAS)
    model.summary()

    # The trained architecture is:
    #   Input -> LSTM(64, return_sequences=True) -> Dropout
    #        -> LSTM(32)                         -> Dropout
    #        -> Dense(16, relu) -> Dense(1)
    # Dropout layers are no-ops at inference time — skip them.
    lstm_layers = [l for l in model.layers if l.__class__.__name__ == "LSTM"]
    dense_layers = [l for l in model.layers if l.__class__.__name__ == "Dense"]
    assert len(lstm_layers) == 2, f"expected 2 LSTM layers, got {len(lstm_layers)}"
    assert len(dense_layers) == 2, f"expected 2 Dense layers, got {len(dense_layers)}"

    payload: dict[str, np.ndarray] = {}
    for idx, layer in enumerate(lstm_layers):
        W, U, b = layer.get_weights()  # input kernel, recurrent kernel, bias
        units = layer.units
        payload[f"lstm{idx}_W"] = W.astype(np.float32)
        payload[f"lstm{idx}_U"] = U.astype(np.float32)
        payload[f"lstm{idx}_b"] = b.astype(np.float32)
        payload[f"lstm{idx}_units"] = np.array(units, dtype=np.int32)
        payload[f"lstm{idx}_return_sequences"] = np.array(
            1 if layer.return_sequences else 0, dtype=np.int32
        )
        print(f"  LSTM[{idx}]: units={units}, "
              f"return_seq={layer.return_sequences}, "
              f"W {W.shape}, U {U.shape}, b {b.shape}")

    for idx, layer in enumerate(dense_layers):
        W, b = layer.get_weights()
        act = getattr(layer.activation, "__name__", "linear")
        payload[f"dense{idx}_W"] = W.astype(np.float32)
        payload[f"dense{idx}_b"] = b.astype(np.float32)
        # Encode activation as a short ASCII string; numpy load returns object
        # arrays for strings so we stick with fixed-length S8.
        payload[f"dense{idx}_act"] = np.array(act, dtype="S8")
        print(f"  Dense[{idx}]: units={W.shape[1]}, act={act}, W {W.shape}, b {b.shape}")

    np.savez(DST_NPZ, **payload)
    size_kb = DST_NPZ.stat().st_size / 1024
    print(f"\nWrote {DST_NPZ}  ({size_kb:.1f} KB)")

    # Sanity check: numpy forward pass must match Keras to float32 precision.
    print("\nSanity check — numpy vs Keras forward pass ...")
    sys.path.insert(0, str(ROOT / "src"))
    from lstm_numpy import NumpyLSTM  # noqa: E402

    rng = np.random.default_rng(0)
    x_test = rng.standard_normal((1, 20, 18)).astype(np.float32)
    keras_out = float(model.predict(x_test, verbose=0).flatten()[0])

    np_model = NumpyLSTM.load(DST_NPZ)
    numpy_out = float(np_model.predict(x_test)[0])

    diff = abs(keras_out - numpy_out)
    print(f"  Keras: {keras_out:.6f}")
    print(f"  NumPy: {numpy_out:.6f}")
    print(f"  diff:  {diff:.2e}")
    if diff > 1e-4:
        print("  WARNING: numerical mismatch larger than expected.")
        sys.exit(2)
    print("  OK — numpy forward pass matches Keras to float32 precision.")


if __name__ == "__main__":
    main()

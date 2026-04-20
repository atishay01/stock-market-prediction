"""Inference utilities: live yfinance fetch + load saved RF/LSTM artifacts.

The Flask app loads `PredictionBundle` once at startup and reuses it across
requests. Per-ticker OHLCV fetches are cached in-process for 5 minutes to
keep the dashboard responsive.

LSTM backend selection (in order of preference):
  1. ``models/lstm_weights.npz`` + ``src/lstm_numpy.py`` — pure numpy
     forward pass, ~140 KB weights, zero binary deps beyond numpy. This
     is the Render-free-tier path (512 MB RAM, can't fit TensorFlow).
  2. ``models/lstm_model.keras`` via full Keras — dev-only fallback for
     machines without the exported weights file.

Set the env var ``LOAD_LSTM=false`` to force-disable LSTM (RF-only).
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

from functools import lru_cache

import joblib
import numpy as np
import pandas as pd
import yfinance as yf
from curl_cffi import requests as _cc_requests
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

_YF_SESSION = _cc_requests.Session(impersonate="chrome")

from features import FEATURE_COLUMNS, build_inference_features


# ---------------------------------------------------------------------------
# Sentiment scoring
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def _vader() -> SentimentIntensityAnalyzer:
    return SentimentIntensityAnalyzer()


def score_headline(text: str) -> float:
    """VADER compound score in [-1, 1]. Empty text returns 0.0."""
    if not text or not text.strip():
        return 0.0
    return float(_vader().polarity_scores(text)["compound"])

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"

_HISTORY_CACHE: dict[tuple[str, str], tuple[float, pd.DataFrame]] = {}
_CACHE_TTL = 300  # seconds


@dataclass
class PredictionBundle:
    rf_model: object
    lstm_predict: object | None  # callable: np.ndarray[1,L,F] -> float (scaled)
    lstm_scaler: object | None
    lstm_norm: dict
    feature_columns: list[str]
    metrics: dict
    lstm_enabled: bool
    lstm_backend: str  # "tflite_runtime", "tf_lite", "keras", or "disabled"


def _lstm_enabled() -> bool:
    return os.environ.get("LOAD_LSTM", "true").lower() not in ("false", "0", "no")


def _make_numpy_predict(npz_path: Path):
    from lstm_numpy import NumpyLSTM
    model = NumpyLSTM.load(npz_path)

    def predict(x: np.ndarray) -> float:
        return float(model.predict(x)[0])

    return predict, "numpy"


def _make_keras_predict(keras_path: Path):
    """Dev-only fallback: load full Keras model. Requires full TensorFlow in
    memory (~500 MB), so this path is only taken when lstm_weights.npz
    hasn't been generated yet on a local machine."""
    from tensorflow.keras.models import load_model
    model = load_model(keras_path)

    def predict(x: np.ndarray) -> float:
        return float(model.predict(x, verbose=0).flatten()[0])

    return predict, "keras"


def load_bundle() -> PredictionBundle:
    metrics = json.loads((MODEL_DIR / "metrics.json").read_text())
    feature_columns = json.loads((MODEL_DIR / "feature_columns.json").read_text())

    lstm_predict = None
    lstm_scaler = None
    backend = "disabled"
    enabled = _lstm_enabled()

    if enabled:
        npz_path = MODEL_DIR / "lstm_weights.npz"
        keras_path = MODEL_DIR / "lstm_model.keras"
        if npz_path.exists():
            lstm_predict, backend = _make_numpy_predict(npz_path)
        elif keras_path.exists():
            try:
                lstm_predict, backend = _make_keras_predict(keras_path)
            except ImportError:
                print("[predict] No lstm_weights.npz and no TensorFlow — "
                      "RF-only. Run src/export_lstm_weights.py to generate "
                      "the numpy weights.")
                enabled = False
        else:
            print("[predict] No LSTM artifact found — RF-only.")
            enabled = False

        if lstm_predict is not None:
            lstm_scaler = joblib.load(MODEL_DIR / "lstm_scaler.pkl")
            print(f"[predict] LSTM backend: {backend}")
    else:
        print("[predict] LOAD_LSTM=false — LSTM disabled, RF-only inference.")

    rf_model = joblib.load(MODEL_DIR / "rf_model.pkl")

    return PredictionBundle(
        rf_model=rf_model,
        lstm_predict=lstm_predict,
        lstm_scaler=lstm_scaler,
        lstm_norm=metrics["lstm"]["normalization"],
        feature_columns=feature_columns,
        metrics=metrics,
        lstm_enabled=enabled and lstm_predict is not None,
        lstm_backend=backend,
    )


_CACHE_DIR = ROOT / "data" / "cache"


def _safe_name(ticker: str) -> str:
    return ticker.replace("^", "_").upper()


def _load_snapshot(ticker: str) -> pd.DataFrame | None:
    """Return bundled CSV snapshot for ticker, or None if unavailable.
    Used as fallback when yfinance is blocked (Yahoo aggressively rate-limits
    data-center IPs like Render's servers)."""
    p = _CACHE_DIR / f"{_safe_name(ticker)}.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p, parse_dates=["Date"])
    return df


def fetch_ticker_history(ticker: str, period: str = "1y") -> pd.DataFrame:
    """Download recent OHLCV for a ticker. Tries yfinance first; falls back
    to bundled CSV snapshot if blocked. In-memory cached for 5 minutes."""
    key = (ticker.upper(), period)
    now = time.time()
    cached = _HISTORY_CACHE.get(key)
    if cached and now - cached[0] < _CACHE_TTL:
        return cached[1]

    df = None
    try:
        df = yf.download(
            ticker, period=period, progress=False, auto_adjust=False,
            threads=False, session=_YF_SESSION,
        )
    except Exception as exc:
        print(f"[predict] yfinance fetch failed for '{ticker}': {exc}")

    if df is None or (hasattr(df, "empty") and df.empty):
        snap = _load_snapshot(ticker)
        if snap is None or snap.empty:
            raise ValueError(
                f"No live data for '{ticker}' and no bundled snapshot. "
                f"Try a ticker in: {sorted(p.stem for p in _CACHE_DIR.glob('*.csv'))}"
            )
        print(f"[predict] using bundled snapshot for '{ticker}' "
              f"(yfinance blocked or empty)")
        df = snap
    else:
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.reset_index()
        date_col = "Date" if "Date" in df.columns else df.columns[0]
        df = df.rename(columns={date_col: "Date"})
        df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)
        if "Adj Close" not in df.columns:
            df["Adj Close"] = df["Close"]
        df = df[["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"]].copy()

    _HISTORY_CACHE[key] = (now, df)
    return df


def predict_next_close(bundle: PredictionBundle, ticker: str, sentiment_score: float) -> dict:
    history_raw = fetch_ticker_history(ticker, period="6mo")
    features = build_inference_features(history_raw, sentiment_score)
    if features.empty:
        raise ValueError(f"Not enough data to build features for '{ticker}'.")

    lookback = int(bundle.lstm_norm["lookback"])
    if len(features) < lookback:
        raise ValueError(
            f"Need at least {lookback} feature rows for LSTM; got {len(features)} for '{ticker}'."
        )

    last_close = float(history_raw["Close"].iloc[-1])

    row = features[FEATURE_COLUMNS].iloc[[-1]].values
    rf_return = float(bundle.rf_model.predict(row)[0])
    rf_pred = last_close * (1.0 + rf_return)

    if bundle.lstm_enabled and bundle.lstm_predict is not None:
        seq = features[FEATURE_COLUMNS].tail(lookback).values
        x = bundle.lstm_scaler.transform(seq)[None, :, :].astype(np.float32)
        scaled = bundle.lstm_predict(x)
        lstm_return = scaled * bundle.lstm_norm["y_std"] + bundle.lstm_norm["y_mean"]
        lstm_pred = last_close * (1.0 + lstm_return)
    else:
        # LSTM disabled (or no artifact available locally) — surface the saved
        # holdout metric so the dashboard can still show it.
        lstm_return = None
        lstm_pred = None

    last_date = pd.to_datetime(history_raw["Date"].iloc[-1])
    next_date = (last_date + pd.offsets.BDay(1)).strftime("%Y-%m-%d")

    per_ticker = bundle.metrics.get("per_ticker_rf_metrics", {})
    return {
        "ticker": ticker.upper(),
        "last_close": round(last_close, 2),
        "last_date": last_date.strftime("%Y-%m-%d"),
        "next_trading_date": next_date,
        "rf_prediction": round(rf_pred, 2),
        "lstm_prediction": round(lstm_pred, 2) if lstm_pred is not None else None,
        "rf_change_pct": round(rf_return * 100, 3),
        "lstm_change_pct": round(lstm_return * 100, 3) if lstm_return is not None else None,
        "sentiment_score": round(float(sentiment_score), 3),
        "rf_train_mape": per_ticker.get(ticker.upper(), {}).get("mape"),
        "lstm_enabled": bundle.lstm_enabled,
        "lstm_backend": bundle.lstm_backend,
    }


def recent_series(ticker: str, tail: int = 180) -> dict:
    history = fetch_ticker_history(ticker, period="1y")
    window = history.tail(tail)
    return {
        "ticker": ticker.upper(),
        "dates": window["Date"].dt.strftime("%Y-%m-%d").tolist(),
        "close": [round(float(v), 2) for v in window["Close"].values],
    }


def holdout_backtest(bundle: PredictionBundle, ticker: str, tail: int = 120) -> dict:
    history = fetch_ticker_history(ticker, period="2y")
    features = build_inference_features(history, sentiment_score=0.0)
    if len(features) < 30:
        raise ValueError(f"Too few rows ({len(features)}) to backtest '{ticker}'.")

    tail = min(tail, len(features) - 1)
    window = features.tail(tail + 1).copy().reset_index(drop=True)
    close_at_t = window["Close"].iloc[:-1].values
    actual_next = window["Close"].iloc[1:].values

    X = window[FEATURE_COLUMNS].iloc[:-1].values
    pred_returns = bundle.rf_model.predict(X)
    pred_close = close_at_t * (1.0 + pred_returns)
    # Persistence baseline: tomorrow = today. Shown on the chart so the viewer
    # can judge whether the model is doing anything beyond this trivial forecast.
    persistence = close_at_t

    # Per-ticker headline numbers — surfaced as a badge above the chart.
    rf_mape = float(np.mean(np.abs((actual_next - pred_close) / actual_next)) * 100)
    persist_mape = float(np.mean(np.abs((actual_next - persistence) / actual_next)) * 100)
    lift_pct = float((persist_mape - rf_mape) / persist_mape * 100) if persist_mape > 0 else 0.0
    actual_returns = (actual_next - close_at_t) / close_at_t
    pred_returns_arr = (pred_close - close_at_t) / close_at_t
    direction_acc = float((np.sign(actual_returns) == np.sign(pred_returns_arr)).mean() * 100)

    dates = window["Date"].iloc[:-1].dt.strftime("%Y-%m-%d").tolist()
    return {
        "ticker": ticker.upper(),
        "dates": dates,
        "actual": [round(float(v), 2) for v in actual_next],
        "predicted": [round(float(v), 2) for v in pred_close],
        "persistence": [round(float(v), 2) for v in persistence],
        "rf_mape": round(rf_mape, 3),
        "persistence_mape": round(persist_mape, 3),
        "lift_vs_persistence_pct": round(lift_pct, 2),
        "directional_accuracy": round(direction_acc, 2),
        "n_days": int(len(actual_next)),
    }

"""Inference utilities: live yfinance fetch + load saved RF/LSTM artifacts.

The Flask app loads `PredictionBundle` once at startup and reuses it across
requests. Per-ticker OHLCV fetches are cached in-process for 5 minutes to
keep the dashboard responsive.

Set the env var ``LOAD_LSTM=false`` to skip loading the Keras model. This
is used in the Render free-tier deploy, where 512MB RAM cannot fit
TensorFlow. The RF model handles all predictions in that mode.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yfinance as yf
from curl_cffi import requests as _cc_requests

_YF_SESSION = _cc_requests.Session(impersonate="chrome")

from features import FEATURE_COLUMNS, build_inference_features

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"

_HISTORY_CACHE: dict[tuple[str, str], tuple[float, pd.DataFrame]] = {}
_CACHE_TTL = 300  # seconds


@dataclass
class PredictionBundle:
    rf_model: object
    lstm_model: object | None
    lstm_scaler: object | None
    lstm_norm: dict
    feature_columns: list[str]
    metrics: dict
    lstm_enabled: bool


def _lstm_enabled() -> bool:
    return os.environ.get("LOAD_LSTM", "true").lower() not in ("false", "0", "no")


def load_bundle() -> PredictionBundle:
    metrics = json.loads((MODEL_DIR / "metrics.json").read_text())
    feature_columns = json.loads((MODEL_DIR / "feature_columns.json").read_text())

    lstm_model = None
    lstm_scaler = None
    enabled = _lstm_enabled()
    if enabled:
        # Keras first so TF thread pools come up before joblib/loky.
        from tensorflow.keras.models import load_model
        lstm_model = load_model(MODEL_DIR / "lstm_model.keras")
        lstm_scaler = joblib.load(MODEL_DIR / "lstm_scaler.pkl")
    else:
        print("[predict] LOAD_LSTM=false — LSTM disabled, RF-only inference.")

    rf_model = joblib.load(MODEL_DIR / "rf_model.pkl")

    return PredictionBundle(
        rf_model=rf_model,
        lstm_model=lstm_model,
        lstm_scaler=lstm_scaler,
        lstm_norm=metrics["lstm"]["normalization"],
        feature_columns=feature_columns,
        metrics=metrics,
        lstm_enabled=enabled,
    )


def fetch_ticker_history(ticker: str, period: str = "1y") -> pd.DataFrame:
    """Download recent OHLCV for a ticker. In-memory cached for 5 minutes."""
    key = (ticker.upper(), period)
    now = time.time()
    cached = _HISTORY_CACHE.get(key)
    if cached and now - cached[0] < _CACHE_TTL:
        return cached[1]

    # Use yf.download (not Ticker.history) — Ticker.history returns empty for
    # some symbols (e.g. MSFT) due to a yfinance quirk, while yf.download works.
    try:
        df = yf.download(
            ticker, period=period, progress=False, auto_adjust=False,
            threads=False, session=_YF_SESSION,
        )
    except Exception as exc:
        raise ValueError(f"yfinance fetch failed for '{ticker}': {exc}")
    if df is None or df.empty:
        raise ValueError(f"No data returned for ticker '{ticker}' — check the symbol.")
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

    if bundle.lstm_enabled and bundle.lstm_model is not None:
        seq = features[FEATURE_COLUMNS].tail(lookback).values
        x = bundle.lstm_scaler.transform(seq)[None, :, :]
        scaled = float(bundle.lstm_model.predict(x, verbose=0).flatten()[0])
        lstm_return = scaled * bundle.lstm_norm["y_std"] + bundle.lstm_norm["y_mean"]
        lstm_pred = last_close * (1.0 + lstm_return)
    else:
        # LSTM disabled in this deployment (e.g. Render free tier).
        # Surface the saved holdout metric so the dashboard can still show it.
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

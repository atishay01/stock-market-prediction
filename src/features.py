"""Scale-free feature engineering for cross-ticker next-day return forecasting.

Every feature is a ratio, return, or z-score so a single model generalizes
across tickers with very different price ranges (e.g. ^GSPC ~4000 and AAPL ~200).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

OHLCV_COLS = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]

FEATURE_COLUMNS = [
    "return_1d", "return_5d", "return_10d",
    "volume_change", "volume_ratio_20",
    "high_low_spread", "open_close_spread",
    "ma_5_ratio", "ma_10_ratio", "ma_20_ratio",
    "volatility_10", "momentum_10_pct",
    "close_lag_1_ret", "close_lag_2_ret", "close_lag_3_ret",
    "close_lag_4_ret", "close_lag_5_ret",
    "sentiment_score",
]


def _scale_free(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["return_1d"] = out["Close"].pct_change()
    out["return_5d"] = out["Close"].pct_change(5)
    out["return_10d"] = out["Close"].pct_change(10)
    out["volume_change"] = out["Volume"].pct_change().replace(
        [np.inf, -np.inf], np.nan).clip(-5, 5)

    rolling_vol = out["Volume"].rolling(20).mean()
    out["volume_ratio_20"] = (out["Volume"] / rolling_vol).replace(
        [np.inf, -np.inf], np.nan).clip(0, 10)

    out["high_low_spread"] = (out["High"] - out["Low"]) / out["Close"]
    out["open_close_spread"] = (out["Close"] - out["Open"]) / out["Open"]

    for w in (5, 10, 20):
        ma = out["Close"].rolling(w).mean()
        out[f"ma_{w}_ratio"] = (ma - out["Close"]) / out["Close"]

    out["volatility_10"] = out["return_1d"].rolling(10).std()
    shifted = out["Close"].shift(10)
    out["momentum_10_pct"] = (out["Close"] - shifted) / shifted

    for lag in range(1, 6):
        out[f"close_lag_{lag}_ret"] = out["Close"].shift(lag) / out["Close"] - 1
    return out


def build_features(df: pd.DataFrame, sentiment_col: pd.Series | None = None) -> pd.DataFrame:
    """Training-time feature build. Derives sentiment proxy from prior-day returns
    when no external sentiment series is supplied. Adds `target_return`."""
    out = _scale_free(df)
    if sentiment_col is None:
        proxy = out["Close"].pct_change().shift(1).clip(-0.05, 0.05) / 0.05
        out["sentiment_score"] = proxy.fillna(0.0)
    else:
        out["sentiment_score"] = sentiment_col.reindex(out.index).fillna(0.0).clip(-1, 1)
    out["target_return"] = out["Close"].pct_change().shift(-1)
    return out.dropna().reset_index(drop=True)


def build_inference_features(df: pd.DataFrame, sentiment_score: float = 0.0) -> pd.DataFrame:
    """Inference-time feature build. The last row's sentiment is set from the
    caller-provided VADER score; all prior rows get 0 (no look-ahead)."""
    out = _scale_free(df)
    out["sentiment_score"] = 0.0
    if len(out) > 0:
        out.iloc[-1, out.columns.get_loc("sentiment_score")] = float(
            np.clip(sentiment_score, -1, 1)
        )
    return out.dropna().reset_index(drop=True)

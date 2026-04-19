"""Flask dashboard: any-ticker next-day close forecasting.

Pulls OHLCV live via yfinance, engineers scale-free features, and runs the
trained cross-ticker RandomForest + LSTM to produce a next-day prediction.
"""
from __future__ import annotations

import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

# TF imported only when LSTM is enabled — Render free tier cannot fit it.
if os.environ.get("LOAD_LSTM", "true").lower() not in ("false", "0", "no"):
    import tensorflow as tf  # noqa: E402,F401

import sys
from pathlib import Path

from flask import Flask, jsonify, render_template, request

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from predict import (
    holdout_backtest, load_bundle, predict_next_close, recent_series, score_headline,
)

app = Flask(__name__, template_folder=str(ROOT / "templates"), static_folder=str(ROOT / "static"))

print("Loading model artifacts ...")
BUNDLE = load_bundle()
print("Ready.")


def _clean_ticker(raw: str | None) -> str:
    if not raw:
        return "AAPL"
    t = raw.strip().upper()
    return t or "AAPL"


@app.route("/")
def index():
    return render_template(
        "index.html",
        metrics=BUNDLE.metrics,
        feature_count=len(BUNDLE.feature_columns),
    )


@app.get("/api/history")
def api_history():
    ticker = _clean_ticker(request.args.get("ticker"))
    try:
        return jsonify(recent_series(ticker, tail=180))
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


@app.get("/api/backtest")
def api_backtest():
    ticker = _clean_ticker(request.args.get("ticker"))
    try:
        return jsonify(holdout_backtest(BUNDLE, ticker, tail=120))
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


@app.post("/api/predict")
def api_predict():
    payload = request.get_json(silent=True) or {}
    ticker = _clean_ticker(payload.get("ticker"))
    headline = (payload.get("headline") or "").strip()
    explicit = payload.get("sentiment_score")

    if explicit is not None:
        try:
            sentiment = float(explicit)
        except (TypeError, ValueError):
            return jsonify({"error": "sentiment_score must be a number"}), 400
    else:
        sentiment = score_headline(headline)

    try:
        result = predict_next_close(BUNDLE, ticker, sentiment)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400

    result["headline"] = headline
    return jsonify(result)


@app.get("/api/metrics")
def api_metrics():
    return jsonify(BUNDLE.metrics)


if __name__ == "__main__":
    # Local dev: bind to 127.0.0.1:5000.
    # Production (Render): gunicorn binds to $PORT via the Procfile, so this
    # branch isn't entered.
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)

# Any-Ticker Next-Day Close Forecast

Cross-ticker stock forecasting project: live OHLCV pulled from yfinance,
18 scale-free engineered features + VADER news-sentiment, a **Random Forest**
model, a **Keras LSTM** deep model, and a **Flask** dashboard that accepts
any ticker symbol and a free-text headline.

One model, 15 training tickers, generalises from ^GSPC (~7,000) to NVDA
(~200) because every input is a ratio or return — nothing is an absolute
price.

## Architecture

```
yfinance (curl_cffi Chrome impersonation)
        │  15 training tickers × 7 years
        ▼
src/features.py  ── 18 scale-free features
                    returns (1/5/10d), volume ratios, HL & OC spreads,
                    MA(5/10/20) ratios, volatility, momentum,
                    5 lagged close returns, VADER sentiment
        │
        ├── src/train.py   ── RandomForestRegressor (n_est=300, depth=10)
        │                      Keras LSTM (2-layer, dropout, MSE)
        │                      writes models/*.{pkl,keras,json}
        │
        └── src/predict.py ── live yfinance fetch (5-min cache),
                              feature build, next-day return prediction,
                              close reconstruction
                 │
                 ▼
         app.py  (Flask)  ── / (dashboard with ticker input),
                              /api/predict?ticker=…&headline=…,
                              /api/history?ticker=…,
                              /api/backtest?ticker=…,
                              /api/metrics
```

## Quick start

```bash
pip install -r requirements.txt
python src/train.py            # downloads 15 tickers × 7y, trains RF + LSTM
python app.py                  # serves http://127.0.0.1:5000
```

On Windows you can just double-click `run.bat`.

## Training tickers

`^GSPC, AAPL, MSFT, GOOGL, AMZN, META, NVDA, TSLA, JPM, V, JNJ, WMT, PG, XOM, KO`
— ~26,000 feature rows, 80/20 time-split *within each ticker*.

## Metrics from the last training run

Written to `models/metrics.json` and surfaced on the dashboard.

| Model          | MAPE   | Directional accuracy |
|----------------|--------|----------------------|
| Random Forest  | 1.29%  | 53.08%               |
| LSTM           | 1.30%  | 53.43%               |

(Cross-ticker MAPE is higher than a single-ticker S&P-only model because
the same weights must cover NVDA volatility and KO stability.)

## Flask API

| Method | Route                    | Purpose                                        |
|--------|--------------------------|------------------------------------------------|
| GET    | `/`                      | Dashboard (ticker input, chips, charts)        |
| GET    | `/api/history?ticker=X`  | Last ~180 closes for X                         |
| GET    | `/api/backtest?ticker=X` | Predicted vs actual next-day close, last ~120d |
| GET    | `/api/metrics`           | Training metrics JSON                          |
| POST   | `/api/predict`           | `{ticker, headline}` → next-day close          |

## File layout

```
recruiter_ready_stock_project/
├── app.py                     Flask entry — 5 routes, ~95 lines
├── run.bat                    One-click launcher (Windows)
├── Procfile                   Production start command (Render/Heroku)
├── .python-version            Python 3.11.9 pin
├── requirements.txt           Dev/training deps (incl. tensorflow)
├── requirements-prod.txt      Render free-tier deps (no tensorflow)
├── README.md
├── models/                    written by src/train.py
│   ├── rf_model.pkl
│   ├── lstm_model.keras
│   ├── lstm_scaler.pkl
│   ├── feature_columns.json
│   └── metrics.json
├── src/
│   ├── features.py            18 scale-free engineered features
│   ├── train.py               multi-ticker downloader + RF + LSTM trainer
│   ├── predict.py             inference + VADER sentiment + CSV fallback
│   └── snapshot_data.py       refresh data/cache/*.csv (run locally)
├── data/
│   ├── sp_dataset.csv         original S&P historical data
│   └── cache/                 2y OHLCV snapshots (production fallback)
├── templates/index.html       Dashboard
├── static/{style.css, dashboard.js}
└── notebooks/stock_analysis.ipynb
```

## Production deploy notes (Render)

Render's free tier (512 MB RAM) cannot fit TensorFlow (~500 MB). Instead of
disabling the LSTM in prod, the trained Keras model's weights are exported
to `models/lstm_weights.npz` (~140 KB) via `src/export_lstm_weights.py`,
and the Flask app runs a pure-numpy forward pass (`src/lstm_numpy.py`) at
inference time. Both RF and LSTM serve live predictions in production with
no TensorFlow dependency — the numpy path matches Keras output to float32
precision (diff 0.0 on sanity-check inputs).

Set `LOAD_LSTM=false` to force RF-only mode if needed.

Yahoo Finance blocks data-center IPs, so the deployed server falls back to
bundled CSV snapshots in `data/cache/` when live yfinance fails. Refresh the
snapshots locally with `python src/snapshot_data.py` and commit.

Locally, yfinance is tried first via a `curl_cffi` Chrome-impersonating
session (avoids `YFRateLimitError`); the CSV fallback only kicks in when
the live fetch returns empty.

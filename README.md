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
├── app.py                     Flask entry
├── run.bat                    One-click launcher (Windows)
├── requirements.txt
├── README.md
├── models/                    written by src/train.py
│   ├── rf_model.pkl
│   ├── lstm_model.keras
│   ├── lstm_scaler.pkl
│   ├── feature_columns.json
│   └── metrics.json
├── src/
│   ├── features.py            18 scale-free features
│   ├── sentiment.py           VADER headline scorer
│   ├── train.py               multi-ticker yfinance downloader + trainer
│   └── predict.py             live per-ticker inference
├── templates/index.html
├── static/{style.css, dashboard.js}
├── notebooks/stock_analysis.ipynb
└── data/sp_dataset.csv        historical S&P reference
```

## Notes on rate limiting

Yahoo Finance aggressively rate-limits plain `requests`. The project uses a
`curl_cffi` session with Chrome TLS fingerprint (`impersonate="chrome"`) to
avoid `YFRateLimitError` during bulk training downloads. The same session is
reused in `predict.py` for live dashboard fetches.

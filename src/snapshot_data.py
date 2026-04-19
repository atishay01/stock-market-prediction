"""One-shot script: pre-fetches 2y of OHLCV for every supported ticker and
saves to ../data/cache/{TICKER}.csv. Used as a fallback when yfinance is
blocked from the deployment server (Yahoo blocks data-center IPs).

Run this locally (where yfinance works), then commit data/cache/*.csv.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import yfinance as yf
from curl_cffi import requests as _cc_requests

ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = ROOT / "data" / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

TICKERS = [
    "^GSPC", "AAPL", "MSFT", "GOOGL", "AMZN", "META",
    "NVDA", "TSLA", "JPM", "V", "JNJ", "WMT", "PG", "XOM", "KO",
]

_SESSION = _cc_requests.Session(impersonate="chrome")


def _safe_name(ticker: str) -> str:
    return ticker.replace("^", "_").upper()


def main() -> None:
    for t in TICKERS:
        try:
            df = yf.download(t, period="2y", progress=False, auto_adjust=False,
                             threads=False, session=_SESSION)
        except Exception as exc:
            print(f"  [skip] {t}: {exc}")
            continue
        if df is None or df.empty:
            print(f"  [skip] {t}: empty")
            continue
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.reset_index()
        date_col = "Date" if "Date" in df.columns else df.columns[0]
        df = df.rename(columns={date_col: "Date"})
        df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)
        if "Adj Close" not in df.columns:
            df["Adj Close"] = df["Close"]
        df = df[["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"]]
        out = CACHE_DIR / f"{_safe_name(t)}.csv"
        df.to_csv(out, index=False)
        print(f"  {t:<8} {len(df):>5} rows -> {out.name}")


if __name__ == "__main__":
    main()

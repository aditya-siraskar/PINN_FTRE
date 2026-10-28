"""Real data for Stage 6: daily prices from Yahoo Finance and their log returns.

Prices are cached in data/raw/ (gitignored). The log returns used by the backtests are
written to data/processed/<market>_returns.csv, which is committed: Yahoo revises
adjusted prices, so the committed file is what makes the results reproducible.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW, PROCESSED = ROOT / "data" / "raw", ROOT / "data" / "processed"


def download_prices(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    import yfinance as yf
    px = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)["Close"]
    return px[tickers]


def load_returns(market: str, tickers: list[str], start: str, end: str, refresh: bool = False) -> pd.DataFrame:
    """Daily log returns (dates where every ticker trades), from the committed file if present."""
    path = PROCESSED / f"{market}_returns.csv"
    if path.exists() and not refresh:
        return pd.read_csv(path, index_col=0, parse_dates=True)
    raw = RAW / f"{market}_prices.csv"
    if raw.exists() and not refresh:
        px = pd.read_csv(raw, index_col=0, parse_dates=True)
    else:
        px = download_prices(tickers, start, end)
        RAW.mkdir(parents=True, exist_ok=True)
        px.to_csv(raw)
    px = px.loc[start:end]
    r = np.log(px).diff().dropna(how="any")
    r = r[(r != 0).any(axis=1)]                     # exchange holidays carried forward by Yahoo
    PROCESSED.mkdir(parents=True, exist_ok=True)
    r.to_csv(path, float_format="%.8f")
    return r

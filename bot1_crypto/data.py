"""ccxt data fetcher for crypto OHLCV.

Pulls bars from a public exchange (default: Binance) and saves to parquet.
"""
from __future__ import annotations

import time
from pathlib import Path

import ccxt
import pandas as pd


_OHLCV_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]


def fetch_history(
    symbol: str,
    timeframe: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    exchange_name: str = "binance",
) -> pd.DataFrame:
    """Fetch a full historical range, paginating through ccxt's per-call limit."""
    exchange = getattr(ccxt, exchange_name)({"enableRateLimit": True})
    chunk_ms = exchange.parse_timeframe(timeframe) * 1000

    if start.tzinfo is None:
        start = start.tz_localize("UTC")
    if end.tzinfo is None:
        end = end.tz_localize("UTC")

    since = int(start.timestamp() * 1000)
    end_ms = int(end.timestamp() * 1000)
    out: list[list] = []

    while since < end_ms:
        chunk = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=1000)
        if not chunk:
            break
        out.extend(chunk)
        next_since = chunk[-1][0] + chunk_ms
        if next_since <= since:
            break
        since = next_since
        time.sleep(exchange.rateLimit / 1000)

    df = pd.DataFrame(out, columns=_OHLCV_COLUMNS)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = (
        df.drop_duplicates(subset=["timestamp"])
        .sort_values("timestamp")
        .reset_index(drop=True)
    )
    return df[df["timestamp"] <= end].reset_index(drop=True)


def save_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def load_parquet(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)

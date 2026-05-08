"""Fetch historical OHLCV bars via ccxt and save to parquet."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from bot1_crypto.data import fetch_history, save_parquet


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--symbol", action="append", default=None,
                   help="Repeatable. Default: BTC/USDT and ETH/USDT")
    p.add_argument("--timeframe", default="1h")
    p.add_argument("--start", default="2023-01-01")
    p.add_argument("--end", default="2025-12-31")
    p.add_argument("--exchange", default="binance")
    p.add_argument("--out-dir", default="data/raw")
    args = p.parse_args()

    symbols = args.symbol or ["BTC/USDT", "ETH/USDT"]
    out_dir = Path(args.out_dir)
    start = pd.Timestamp(args.start, tz="UTC")
    end = pd.Timestamp(args.end, tz="UTC")

    for sym in symbols:
        print(f"Fetching {sym} {args.timeframe} from {start.date()} to {end.date()}...")
        df = fetch_history(sym, args.timeframe, start, end, exchange_name=args.exchange)
        sym_safe = sym.replace("/", "_")
        path = out_dir / f"{sym_safe}_{args.timeframe}_{start.date()}_{end.date()}.parquet"
        save_parquet(df, path)
        print(f"  -> {path} ({len(df)} bars)")


if __name__ == "__main__":
    main()

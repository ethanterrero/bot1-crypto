"""Zero-shot directional accuracy evaluator.

Walks through historical data, runs Kronos at each step, and scores whether
the predicted close direction matches realized.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import pandas as pd
from tqdm import tqdm

from .predictor import KronosPredictor


@dataclass
class WindowResult:
    timestamp: pd.Timestamp
    last_close: float
    pred_close: float
    realized_close: float
    pred_direction: int
    realized_direction: int
    hit: bool


def evaluate_directional(
    df: pd.DataFrame,
    predictor: KronosPredictor,
    lookback: int = 400,
    pred_len: int = 24,
    stride: int = 24,
    sample_count: int = 1,
) -> pd.DataFrame:
    """Walk-forward directional accuracy.

    For each window: take `lookback` bars, predict `pred_len` ahead, check
    whether predicted close at horizon is on the same side of the last actual
    close as the realized close at that horizon.
    """
    if len(df) < lookback + pred_len:
        raise ValueError(
            f"Not enough bars: need >={lookback + pred_len}, got {len(df)}."
        )

    rows: list[WindowResult] = []
    end = len(df) - pred_len
    starts = range(lookback, end, stride)

    for start in tqdm(starts, desc="windows"):
        hist = df.iloc[start - lookback : start].reset_index(drop=True)
        future = df.iloc[start : start + pred_len].reset_index(drop=True)
        last_close = float(hist["close"].iloc[-1])

        pred = predictor.predict(hist, pred_len=pred_len, sample_count=sample_count)
        pred_close = float(pred["close"].iloc[-1])
        realized_close = float(future["close"].iloc[-1])

        pred_dir = 1 if pred_close > last_close else -1
        real_dir = 1 if realized_close > last_close else -1

        rows.append(
            WindowResult(
                timestamp=hist["timestamp"].iloc[-1],
                last_close=last_close,
                pred_close=pred_close,
                realized_close=realized_close,
                pred_direction=pred_dir,
                realized_direction=real_dir,
                hit=(pred_dir == real_dir),
            )
        )

    return pd.DataFrame([asdict(r) for r in rows])


def summarize(results: pd.DataFrame) -> dict:
    """Aggregate metrics. `edge_vs_baseline` subtracts the majority-class hit rate."""
    n = len(results)
    if n == 0:
        return {"n": 0}

    hit_rate = float(results["hit"].mean())
    pred_long_share = float((results["pred_direction"] == 1).mean())
    real_long_share = float((results["realized_direction"] == 1).mean())
    majority_baseline = max(real_long_share, 1.0 - real_long_share)

    return {
        "n": n,
        "hit_rate": hit_rate,
        "pred_long_share": pred_long_share,
        "real_long_share": real_long_share,
        "majority_baseline": majority_baseline,
        "edge_vs_baseline": hit_rate - majority_baseline,
    }

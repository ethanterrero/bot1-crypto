"""Cheap directional baselines.

Each baseline takes the same lookback DataFrame the predictor sees and returns
+1 (long) or -1 (short). Used to gate Kronos: if it can't beat AR(1) on a hit
rate basis, the foundation model isn't earning its compute.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd


Baseline = Callable[[pd.DataFrame, int], int]


def buy_and_hold(df: pd.DataFrame, pred_len: int) -> int:
    return 1


def last_bar_momentum(df: pd.DataFrame, pred_len: int) -> int:
    closes = df["close"].to_numpy()
    return 1 if closes[-1] > closes[-2] else -1


def sma_crossover(df: pd.DataFrame, pred_len: int, fast: int = 24, slow: int = 72) -> int:
    closes = df["close"].to_numpy()
    if len(closes) < slow:
        return 1
    fast_ma = closes[-fast:].mean()
    slow_ma = closes[-slow:].mean()
    return 1 if fast_ma >= slow_ma else -1


def ar1_log_return(df: pd.DataFrame, pred_len: int) -> int:
    """Direction of cumulative pred_len-bar return under an AR(1) on log returns.

    Closed-form: with r_t = phi * r_{t-1} + eps and last observed return r,
    E[sum_{k=1..h} r_{t+k}] = r * phi * (1 - phi^h) / (1 - phi).
    """
    closes = df["close"].to_numpy()
    if len(closes) < 3:
        return 1
    log_ret = np.diff(np.log(closes))
    if len(log_ret) < 2:
        return 1
    x = log_ret[:-1]
    y = log_ret[1:]
    denom = float(np.dot(x, x))
    if denom == 0.0:
        return 1 if log_ret[-1] >= 0 else -1
    phi = float(np.dot(x, y) / denom)
    phi = max(min(phi, 0.999), -0.999)
    last = float(log_ret[-1])
    if abs(1.0 - phi) < 1e-9:
        expected = last * pred_len
    else:
        expected = last * phi * (1.0 - phi**pred_len) / (1.0 - phi)
    return 1 if expected >= 0 else -1


BASELINES: dict[str, Baseline] = {
    "buy_and_hold": buy_and_hold,
    "last_bar_momentum": last_bar_momentum,
    "sma_24_72": sma_crossover,
    "ar1_log_return": ar1_log_return,
}

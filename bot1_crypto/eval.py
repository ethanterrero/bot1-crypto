"""Walk-forward directional eval with conviction + baselines.

For each window we draw `n_samples` Kronos forecasts (independent calls so we
can measure dispersion), compute a conviction score, and score the same window
under a small bench of cheap baselines. The bucketed/baseline summary is the
point — the headline hit rate is uninformative on its own.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, field

import numpy as np
import pandas as pd
from tqdm import tqdm

from .baselines import BASELINES
from .predictor import KronosPredictor


@dataclass
class WindowResult:
    timestamp: pd.Timestamp
    last_close: float
    realized_close: float
    realized_log_return: float
    realized_direction: int
    pred_mean_close: float
    pred_mean_log_return: float
    pred_std_log_return: float
    conviction: float
    pred_direction: int
    hit: bool
    baseline_dirs: dict = field(default_factory=dict)
    baseline_hits: dict = field(default_factory=dict)


def _flatten_baselines(rows: list[WindowResult]) -> pd.DataFrame:
    base = pd.DataFrame([
        {k: v for k, v in asdict(r).items() if k not in ("baseline_dirs", "baseline_hits")}
        for r in rows
    ])
    for name in BASELINES:
        base[f"dir_{name}"] = [r.baseline_dirs.get(name) for r in rows]
        base[f"hit_{name}"] = [r.baseline_hits.get(name) for r in rows]
    return base


def evaluate_directional(
    df: pd.DataFrame,
    predictor: KronosPredictor,
    lookback: int = 400,
    pred_len: int = 24,
    stride: int = 24,
    n_samples: int = 20,
    T: float = 1.0,
    top_p: float = 0.9,
) -> pd.DataFrame:
    """Walk forward, predict, score Kronos + baselines per window."""
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
        realized_close = float(future["close"].iloc[-1])
        realized_log_return = float(np.log(realized_close / last_close))
        real_dir = 1 if realized_close > last_close else -1

        sample_paths = predictor.predict_samples(
            hist, pred_len=pred_len, n_samples=n_samples, T=T, top_p=top_p,
        )
        sample_horizon_closes = sample_paths[:, -1]
        sample_log_returns = np.log(sample_horizon_closes / last_close)
        mean_lr = float(sample_log_returns.mean())
        std_lr = float(sample_log_returns.std(ddof=1)) if n_samples > 1 else 0.0
        conviction = abs(mean_lr) / std_lr if std_lr > 0 else float("inf")
        pred_mean_close = float(np.exp(mean_lr) * last_close)
        pred_dir = 1 if mean_lr >= 0 else -1
        hit = pred_dir == real_dir

        baseline_dirs = {name: int(fn(hist, pred_len)) for name, fn in BASELINES.items()}
        baseline_hits = {name: bool(d == real_dir) for name, d in baseline_dirs.items()}

        rows.append(
            WindowResult(
                timestamp=hist["timestamp"].iloc[-1],
                last_close=last_close,
                realized_close=realized_close,
                realized_log_return=realized_log_return,
                realized_direction=real_dir,
                pred_mean_close=pred_mean_close,
                pred_mean_log_return=mean_lr,
                pred_std_log_return=std_lr,
                conviction=conviction,
                pred_direction=pred_dir,
                hit=hit,
                baseline_dirs=baseline_dirs,
                baseline_hits=baseline_hits,
            )
        )

    return _flatten_baselines(rows)


def _bucketed_hit_rate(results: pd.DataFrame, n_buckets: int) -> pd.DataFrame:
    """Hit rate by conviction bucket (lowest = bucket 0). Drops inf convictions
    (zero dispersion) into the top bucket; those are rare and indicate degenerate
    prediction agreement."""
    finite = np.isfinite(results["conviction"])
    df = results.loc[finite].copy()
    if len(df) < n_buckets:
        return pd.DataFrame()
    df["bucket"] = pd.qcut(df["conviction"], n_buckets, labels=False, duplicates="drop")
    grouped = df.groupby("bucket").agg(
        n=("hit", "size"),
        hit_rate=("hit", "mean"),
        mean_conviction=("conviction", "mean"),
        long_share=("pred_direction", lambda s: (s == 1).mean()),
    ).reset_index()
    return grouped


def summarize(results: pd.DataFrame, conviction_buckets: int = 10) -> dict:
    """Aggregate Kronos + baseline metrics + conviction bucketing."""
    n = len(results)
    if n == 0:
        return {"n": 0}

    real_long_share = float((results["realized_direction"] == 1).mean())
    majority_baseline = max(real_long_share, 1.0 - real_long_share)
    kronos_hit = float(results["hit"].mean())

    out: dict = {
        "n": n,
        "real_long_share": real_long_share,
        "majority_baseline": majority_baseline,
        "kronos_hit_rate": kronos_hit,
        "kronos_pred_long_share": float((results["pred_direction"] == 1).mean()),
        "kronos_edge_vs_majority": kronos_hit - majority_baseline,
    }

    for name in BASELINES:
        col = f"hit_{name}"
        if col in results.columns:
            hit = float(results[col].mean())
            out[f"baseline_{name}_hit_rate"] = hit
            out[f"kronos_edge_vs_{name}"] = kronos_hit - hit

    out["mean_pred_std_log_return"] = float(results["pred_std_log_return"].mean())
    out["median_conviction"] = float(
        results.loc[np.isfinite(results["conviction"]), "conviction"].median()
    )
    out["conviction_buckets"] = _bucketed_hit_rate(results, conviction_buckets).to_dict("records")
    return out

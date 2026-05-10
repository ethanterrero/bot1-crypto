"""Smoke tests.

Predictor tests skip if Kronos repo isn't accessible at KRONOS_PATH (or default
~/Desktop/Kronos). Baselines test runs unconditionally — pure numpy.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

KRONOS_PATH = Path(os.environ.get("KRONOS_PATH", os.path.expanduser("~/Desktop/Kronos")))
KRONOS_AVAILABLE = (KRONOS_PATH / "model").is_dir()


def _toy_df(n: int = 100) -> pd.DataFrame:
    ts = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    return pd.DataFrame({
        "timestamp": ts,
        "open":  [40000.0 + i for i in range(n)],
        "high":  [40100.0 + i for i in range(n)],
        "low":   [39900.0 + i for i in range(n)],
        "close": [40050.0 + i for i in range(n)],
        "volume": [10.0] * n,
    })


@pytest.mark.skipif(not KRONOS_AVAILABLE, reason="Kronos repo not at KRONOS_PATH")
def test_predict_returns_correct_shape():
    from bot1_crypto.predictor import KronosPredictor

    predictor = KronosPredictor(device="cpu")
    out = predictor.predict(_toy_df(), pred_len=10, sample_count=1)
    assert len(out) == 10
    assert {"open", "high", "low", "close"} <= set(out.columns)


@pytest.mark.skipif(not KRONOS_AVAILABLE, reason="Kronos repo not at KRONOS_PATH")
def test_predict_samples_returns_expected_shape():
    from bot1_crypto.predictor import KronosPredictor

    predictor = KronosPredictor(device="cpu")
    paths = predictor.predict_samples(_toy_df(), pred_len=10, n_samples=3)
    assert paths.shape == (3, 10)
    assert np.all(paths > 0)


def test_data_module_imports():
    from bot1_crypto import baselines, data, eval as eval_mod  # noqa: F401
    from bot1_crypto.predictor import KronosPredictor  # noqa: F401


def test_baselines_return_signed_ints():
    from bot1_crypto.baselines import BASELINES

    df = _toy_df(200)
    for name, fn in BASELINES.items():
        d = fn(df, pred_len=24)
        assert d in (-1, 1), f"{name} returned {d}"


def test_ar1_baseline_uses_recent_return_sign_when_phi_positive():
    """If phi > 0 and last return is positive, AR(1) forecast is positive."""
    from bot1_crypto.baselines import ar1_log_return

    n = 200
    rng = np.random.default_rng(0)
    rets = np.zeros(n)
    rets[0] = 0.001
    phi = 0.6
    for i in range(1, n):
        rets[i] = phi * rets[i - 1] + rng.normal(0, 1e-4)
    closes = np.exp(np.cumsum(rets)) * 40000.0
    closes[-1] = closes[-2] * 1.01

    df = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC"),
        "open": closes, "high": closes, "low": closes, "close": closes,
        "volume": [1.0] * n,
    })
    assert ar1_log_return(df, pred_len=24) == 1

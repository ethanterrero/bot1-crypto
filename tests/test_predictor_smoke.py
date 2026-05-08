"""Smoke test: predictor instantiates and predict() returns the right shape.

Skipped if Kronos repo isn't accessible at KRONOS_PATH (or default ~/Desktop/Kronos).
"""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import pytest

KRONOS_PATH = Path(os.environ.get("KRONOS_PATH", os.path.expanduser("~/Desktop/Kronos")))
KRONOS_AVAILABLE = (KRONOS_PATH / "model").is_dir()


@pytest.mark.skipif(not KRONOS_AVAILABLE, reason="Kronos repo not at KRONOS_PATH")
def test_predict_returns_correct_shape():
    from bot1_crypto.predictor import KronosPredictor

    predictor = KronosPredictor(device="cpu")

    n = 100
    ts = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    df = pd.DataFrame({
        "timestamp": ts,
        "open":  [40000.0 + i for i in range(n)],
        "high":  [40100.0 + i for i in range(n)],
        "low":   [39900.0 + i for i in range(n)],
        "close": [40050.0 + i for i in range(n)],
        "volume": [10.0] * n,
    })

    out = predictor.predict(df, pred_len=10, sample_count=1)
    assert len(out) == 10
    assert {"open", "high", "low", "close"} <= set(out.columns)


def test_data_module_imports():
    """Cheap import-time check that doesn't need Kronos."""
    from bot1_crypto import data, eval as eval_mod  # noqa: F401
    from bot1_crypto.predictor import KronosPredictor  # noqa: F401

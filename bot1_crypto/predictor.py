"""Kronos predictor wrapper.

Loads Kronos from HuggingFace and exposes a simple predict() that takes a
ccxt-formatted DataFrame and returns forecasted OHLCV at the same cadence.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def _ensure_kronos_on_path() -> None:
    """Kronos isn't on PyPI; the user clones the repo and we import its `model` package."""
    kronos_path = Path(os.environ.get("KRONOS_PATH", str(Path.home() / "Desktop" / "Kronos")))
    if not (kronos_path / "model").is_dir():
        raise RuntimeError(
            f"Kronos not found at {kronos_path}. "
            "Clone https://github.com/shiyu-coder/Kronos to ~/Desktop/Kronos or set KRONOS_PATH."
        )
    if str(kronos_path) not in sys.path:
        sys.path.insert(0, str(kronos_path))


class KronosPredictor:
    """Thin wrapper around Kronos for crypto OHLCV forecasting."""

    REQUIRED_COLUMNS = {"timestamp", "open", "high", "low", "close", "volume"}

    def __init__(
        self,
        model_name: str = "NeoQuasar/Kronos-small",
        tokenizer_name: str = "NeoQuasar/Kronos-Tokenizer-base",
        max_context: int = 512,
        device: str = "cpu",
    ):
        _ensure_kronos_on_path()
        from model import Kronos, KronosTokenizer  # noqa: E402
        from model import KronosPredictor as _RawPredictor  # noqa: E402

        tokenizer = KronosTokenizer.from_pretrained(tokenizer_name)
        model = Kronos.from_pretrained(model_name)
        self._predictor = _RawPredictor(model, tokenizer, max_context=max_context, device=device)
        self.max_context = max_context

    def predict(
        self,
        df: pd.DataFrame,
        pred_len: int,
        T: float = 1.0,
        top_p: float = 0.9,
        sample_count: int = 1,
    ) -> pd.DataFrame:
        """Predict the next pred_len bars given a ccxt-formatted lookback DataFrame.

        df columns: timestamp, open, high, low, close, volume. `amount` (notional)
        is optional; if absent it's filled with close * volume.
        Returns a DataFrame of forecasted OHLCV indexed by future timestamps inferred
        from the lookback's cadence.
        """
        missing = self.REQUIRED_COLUMNS - set(df.columns)
        if missing:
            raise ValueError(f"df missing columns: {missing}")
        if len(df) < 2:
            raise ValueError("Need at least 2 rows of history to infer cadence.")

        df = df.copy()
        if "amount" not in df.columns:
            df["amount"] = df["close"] * df["volume"]

        x_df = df[["open", "high", "low", "close", "volume", "amount"]].reset_index(drop=True)
        x_timestamp = pd.to_datetime(df["timestamp"]).reset_index(drop=True)

        cadence = x_timestamp.iloc[-1] - x_timestamp.iloc[-2]
        last = x_timestamp.iloc[-1]
        y_timestamp = pd.Series([last + cadence * (i + 1) for i in range(pred_len)])

        return self._predictor.predict(
            df=x_df,
            x_timestamp=x_timestamp,
            y_timestamp=y_timestamp,
            pred_len=pred_len,
            T=T,
            top_p=top_p,
            sample_count=sample_count,
        )

    def predict_samples(
        self,
        df: pd.DataFrame,
        pred_len: int,
        n_samples: int = 20,
        T: float = 1.0,
        top_p: float = 0.9,
    ) -> np.ndarray:
        """Draw n_samples independent forecasts and return predicted close paths.

        Kronos averages samples internally when sample_count>1 (see kronos.py:467),
        so to recover dispersion we call predict() n_samples times with sample_count=1
        and stack the results. Returns shape (n_samples, pred_len).
        """
        missing = self.REQUIRED_COLUMNS - set(df.columns)
        if missing:
            raise ValueError(f"df missing columns: {missing}")
        if len(df) < 2:
            raise ValueError("Need at least 2 rows of history to infer cadence.")

        df = df.copy()
        if "amount" not in df.columns:
            df["amount"] = df["close"] * df["volume"]

        x_df = df[["open", "high", "low", "close", "volume", "amount"]].reset_index(drop=True)
        x_timestamp = pd.to_datetime(df["timestamp"]).reset_index(drop=True)
        cadence = x_timestamp.iloc[-1] - x_timestamp.iloc[-2]
        last = x_timestamp.iloc[-1]
        y_timestamp = pd.Series([last + cadence * (i + 1) for i in range(pred_len)])

        out = np.empty((n_samples, pred_len), dtype=np.float64)
        for i in range(n_samples):
            pred = self._predictor.predict(
                df=x_df,
                x_timestamp=x_timestamp,
                y_timestamp=y_timestamp,
                pred_len=pred_len,
                T=T,
                top_p=top_p,
                sample_count=1,
                verbose=False,
            )
            out[i] = pred["close"].to_numpy()
        return out

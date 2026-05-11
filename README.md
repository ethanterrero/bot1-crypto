# bot1-crypto

Research side project of [bot1](https://github.com/ethanterrero/bot1).
Tests whether [Kronos](https://github.com/shiyu-coder/Kronos) — an open-source
foundation model for OHLCV candlesticks — has zero-shot directional edge on
hourly crypto bars.

> **Status:** experimental. No live trading. Zero-shot evaluation only at this stage.

---

## Why this exists

Kronos is a decoder-only transformer pre-trained on K-line data from 45 global
exchanges (paper: [arXiv 2508.02739](https://arxiv.org/abs/2508.02739),
AAAI 2026). Its training distribution covers crypto and global equities heavily
— BTC/USDT is the model's [live demo](https://shiyu-coder.github.io/Kronos-demo/).

bot1 trades NQ/ES futures intraday, which is almost certainly out of Kronos's
training distribution. Before integrating Kronos into bot1, we want to test
it on a market it actually saw during pre-training.

The headline question: **does Kronos zero-shot beat 50% directional accuracy
on hourly BTC and ETH?** If yes, it's worth wiring into bot1 (or building a
crypto strategy around). If no, no harm done — we move on.

---

## What's here

```
bot1-crypto/
├── bot1_crypto/
│   ├── predictor.py    # KronosPredictor wrapper (ccxt DataFrame → forecast)
│   ├── data.py         # ccxt Binance fetcher + parquet I/O
│   └── eval.py         # walk-forward directional accuracy evaluator
├── scripts/
│   ├── fetch_data.py     # pull historical hourly bars
│   └── eval_zero_shot.py # run the directional eval
└── tests/
    └── test_predictor_smoke.py
```

---

## Setup

### 1. Clone Kronos (sibling directory)

Kronos isn't on PyPI — its `model/` package is imported directly. The wrapper
expects Kronos at `~/Desktop/Kronos` by default; override via `KRONOS_PATH`.

```bash
cd ~/Desktop
git clone https://github.com/shiyu-coder/Kronos
```

### 2. Install dependencies

```bash
cd ~/Desktop/bot1-crypto
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Includes `ccxt`, `pandas`, `torch`, `huggingface_hub`. First Kronos load
downloads ~100 MB of model weights from HuggingFace.

---

## Usage

### Pull historical data

```bash
python scripts/fetch_data.py \
  --symbol BTC/USDT --symbol ETH/USDT \
  --timeframe 1h \
  --start 2023-01-01 --end 2025-12-31
```

Saves to `data/raw/BTC_USDT_1h_*.parquet`. Free, no API key required.

### Run zero-shot directional eval

```bash
python scripts/eval_zero_shot.py \
  --data data/raw/BTC_USDT_1h_2023-01-01_2025-12-31.parquet \
  --lookback 400 --pred-len 24 --stride 24 \
  --n-samples 20 \
  --out results/btc_zero_shot.csv
```

Per window: draws `n-samples` independent Kronos forecasts, takes the mean
predicted close at horizon to call direction, and the cross-sample std of
predicted log returns as a dispersion measure. Conviction is
`|mean log return| / std log return`. The same window is also scored under
buy-and-hold, last-bar momentum, AR(1) on log returns, and 24/72 SMA crossover.

Output (abridged):
```
windows: 1247
realized long share:    0.5102
majority baseline:      0.5102

Kronos hit rate:        0.5234
  edge vs majority:     +0.0132

baseline                 hit_rate    kronos_edge
buy_and_hold              0.5102        +0.0132
last_bar_momentum         0.5180        +0.0054
sma_24_72                 0.5215        +0.0019
ar1_log_return            0.5161        +0.0073

conv_decile     n   hit_rate   mean_conv   long_share
0             125     0.4960      0.0541      0.5840
...
9             124     0.5806      0.7412      0.5403
```

### What to look at

- **kronos_edge_vs_ar1_log_return** is the load-bearing number, not
  `edge_vs_majority`. AR(1) on log returns is the cheapest non-trivial
  predictor. If Kronos doesn't beat it, the foundation model isn't earning
  its compute on this task.
- **Top conviction decile**. The unconditional hit rate is mush. The
  question is whether the top-decile-conviction calls are materially better
  than the average call — that's where a tradeable signal would live, if
  one exists.
- **Caveat — in-sample contamination.** Kronos was pre-trained on K-lines
  from 45 exchanges and BTC/USDT is featured in their public demo, so any
  evaluation window before Kronos's training cutoff is partially measuring
  memorization, not generalization. Restrict to windows after the cutoff
  for a clean read.

---

## What's NOT in scope (yet)

- Trade execution / paper trading — eval only at v1
- Fee-aware PnL backtest (sized positions, costs, Sharpe/DD)
- Out-of-time slice gated to Kronos's training cutoff
- Funding-rate / basis overlay
- Finetuning Kronos on crypto data
- Cross-pair / multi-timeframe blending

If the zero-shot eval shows real signal, those become the v2 backlog.

---

## Decision gate

- `edge_vs_baseline >= 0.05` (5pp over majority baseline) on out-of-sample data
  with 500+ windows → worth building a strategy on top
- `0 < edge_vs_baseline < 0.05` → test conviction bucketing; signal may exist
  in the high-conviction tail
- `edge_vs_baseline <= 0` → Kronos doesn't have zero-shot edge here. Try
  finetuning, or table it

---

## License

No license granted yet. Not ready to share.

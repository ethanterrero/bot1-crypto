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
  --out results/btc_zero_shot.csv
```

Walks forward through history, runs Kronos at each step, scores whether the
predicted next-24h close direction matches realized.

Output:
```
metric                              value
hit_rate                           0.5234
pred_long_share                    0.5871
real_long_share                    0.5102
edge_vs_baseline                   0.0132
n                                   1247
```

### Interpreting `edge_vs_baseline`

`hit_rate - max(real_long_share, 1 - real_long_share)`. Subtracts the trivial
"always predict the majority direction" baseline. Positive = real signal,
zero = model isn't beating a one-line heuristic.

---

## What's NOT in scope (yet)

- Trade execution / paper trading — eval only at v1
- Conviction-weighted sizing (sample spread → confidence)
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

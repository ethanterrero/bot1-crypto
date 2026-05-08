"""Run zero-shot directional eval of Kronos on a parquet file."""
from __future__ import annotations

import argparse
from pathlib import Path

from bot1_crypto.data import load_parquet
from bot1_crypto.eval import evaluate_directional, summarize
from bot1_crypto.predictor import KronosPredictor


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, help="Path to parquet file")
    p.add_argument("--lookback", type=int, default=400)
    p.add_argument("--pred-len", type=int, default=24)
    p.add_argument("--stride", type=int, default=24)
    p.add_argument("--model", default="NeoQuasar/Kronos-small")
    p.add_argument("--tokenizer", default="NeoQuasar/Kronos-Tokenizer-base")
    p.add_argument("--device", default="cpu")
    p.add_argument("--sample-count", type=int, default=1)
    p.add_argument("--out", default=None, help="CSV path for per-window results")
    args = p.parse_args()

    df = load_parquet(Path(args.data))
    print(f"Loaded {len(df)} bars from {args.data}")

    predictor = KronosPredictor(
        model_name=args.model,
        tokenizer_name=args.tokenizer,
        max_context=max(512, args.lookback),
        device=args.device,
    )
    results = evaluate_directional(
        df, predictor,
        lookback=args.lookback,
        pred_len=args.pred_len,
        stride=args.stride,
        sample_count=args.sample_count,
    )
    summary = summarize(results)

    print()
    print("=" * 50)
    print(f"{'metric':<25} {'value':>20}")
    print("-" * 50)
    for k, v in summary.items():
        if isinstance(v, float):
            print(f"{k:<25} {v:>20.4f}")
        else:
            print(f"{k:<25} {v:>20}")
    print()

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        results.to_csv(out_path, index=False)
        print(f"Wrote {len(results)} per-window rows to {out_path}")


if __name__ == "__main__":
    main()

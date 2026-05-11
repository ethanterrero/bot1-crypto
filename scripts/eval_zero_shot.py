"""Run zero-shot directional eval of Kronos on a parquet file.

Reports Kronos hit rate vs. majority baseline AND vs. cheap baselines
(buy-and-hold, last-bar momentum, AR(1), SMA crossover), plus hit rate
bucketed by Kronos's conviction (|mean pred return| / std across samples).

The conviction bucketing is the point: the unconditional hit rate is mush.
The interesting question is whether the top-decile-conviction calls are
materially better than the average call.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from bot1_crypto.data import load_parquet
from bot1_crypto.eval import evaluate_directional, summarize
from bot1_crypto.predictor import KronosPredictor


def _print_summary(summary: dict) -> None:
    n = summary.get("n", 0)
    print()
    print("=" * 60)
    print(f"windows: {n}")
    print(f"realized long share:    {summary['real_long_share']:.4f}")
    print(f"majority baseline:      {summary['majority_baseline']:.4f}")
    print()
    print(f"Kronos hit rate:        {summary['kronos_hit_rate']:.4f}")
    print(f"Kronos pred long share: {summary['kronos_pred_long_share']:.4f}")
    print(f"  edge vs majority:     {summary['kronos_edge_vs_majority']:+.4f}")
    print()
    print(f"{'baseline':<22} {'hit_rate':>10} {'kronos_edge':>14}")
    print("-" * 50)
    for k, v in summary.items():
        if k.startswith("baseline_") and k.endswith("_hit_rate"):
            name = k.removeprefix("baseline_").removesuffix("_hit_rate")
            edge = summary.get(f"kronos_edge_vs_{name}", float("nan"))
            print(f"{name:<22} {v:>10.4f} {edge:>+14.4f}")
    print()
    print(f"mean pred-std (log ret): {summary['mean_pred_std_log_return']:.5f}")
    print(f"median conviction:       {summary['median_conviction']:.4f}")
    print()
    buckets = summary.get("conviction_buckets") or []
    if buckets:
        print(f"{'conv_decile':<12} {'n':>5} {'hit_rate':>10} {'mean_conv':>10} {'long_share':>11}")
        print("-" * 52)
        for b in buckets:
            print(
                f"{int(b['bucket']):<12} {int(b['n']):>5} "
                f"{b['hit_rate']:>10.4f} {b['mean_conviction']:>10.4f} "
                f"{b['long_share']:>11.4f}"
            )
    print()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, help="Path to parquet file")
    p.add_argument("--lookback", type=int, default=400)
    p.add_argument("--pred-len", type=int, default=24)
    p.add_argument("--stride", type=int, default=24)
    p.add_argument("--model", default="NeoQuasar/Kronos-small")
    p.add_argument("--tokenizer", default="NeoQuasar/Kronos-Tokenizer-base")
    p.add_argument("--device", default="cpu")
    p.add_argument("--n-samples", type=int, default=20,
                   help="Independent Kronos samples per window for dispersion.")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--conviction-buckets", type=int, default=10)
    p.add_argument("--out", default=None, help="CSV path for per-window results")
    p.add_argument("--summary-out", default=None, help="JSON path for summary")
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
        n_samples=args.n_samples,
        T=args.temperature,
        top_p=args.top_p,
    )
    summary = summarize(results, conviction_buckets=args.conviction_buckets)

    _print_summary(summary)

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        results.to_csv(out_path, index=False)
        print(f"Wrote {len(results)} per-window rows to {out_path}")

    if args.summary_out:
        sp = Path(args.summary_out)
        sp.parent.mkdir(parents=True, exist_ok=True)
        sp.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
        print(f"Wrote summary to {sp}")


if __name__ == "__main__":
    main()

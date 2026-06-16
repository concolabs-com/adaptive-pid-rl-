#!/usr/bin/env python3
"""Aggregate per-seed eval_seed_summary.csv files across all seeds of a run.

Needed because stage2_meta_rl_reproduction writes its top-level aggregate only
over the seeds of a single invocation; Stage 6a was run as a split (seeds
7/21/42 then 84/123 after the sleep stall), so its built-in aggregate is
incomplete. This recomputes mean / std / 95% bootstrap CI over the seed-level
means for every scenario and metric.

Usage:
  python utils/aggregate_seeds.py benchmark_results/stage6a_context_hipmdp [--out ...]
"""

import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd

METRICS = ["settling_time_mean", "overshoot_mean", "iae_mean", "final_abs_error_mean", "success_rate"]


def bootstrap_ci(values: np.ndarray, n: int = 10_000, alpha: float = 0.05):
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    if len(values) < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(0)
    means = rng.choice(values, size=(n, len(values)), replace=True).mean(axis=1)
    return (float(np.percentile(means, 100 * alpha / 2)), float(np.percentile(means, 100 * (1 - alpha / 2))))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    files = sorted(glob.glob(str(run_dir / "seed_*" / "eval_seed_summary.csv")))
    if not files:
        print(f"No per-seed files under {run_dir}")
        return 1

    frames = []
    for f in files:
        seed = Path(f).parent.name.replace("seed_", "")
        frames.append(pd.read_csv(f).assign(_seed=seed))
    df = pd.concat(frames, ignore_index=True)

    rows = []
    for scenario, grp in df.groupby("scenario"):
        # Each seed already contributes one row per scenario (its own episode
        # mean); aggregate those seed-level values.
        row = {"scenario": scenario, "n_seeds": grp["_seed"].nunique()}
        for m in METRICS:
            if m not in grp.columns:
                continue
            vals = grp[m].to_numpy(dtype=float)
            lo, hi = bootstrap_ci(vals)
            row[f"{m}"] = float(np.nanmean(vals))
            row[f"{m}_sd"] = float(np.nanstd(vals, ddof=1)) if len(vals) > 1 else 0.0
            row[f"{m}_ci_lo"] = lo
            row[f"{m}_ci_hi"] = hi
        rows.append(row)

    out = Path(args.out) if args.out else run_dir / "aggregate_all_seeds.csv"
    summary = pd.DataFrame(rows)
    summary.to_csv(out, index=False)
    print(f"Wrote {out}  ({df['_seed'].nunique()} seeds, {len(files)} files)")
    cols = [
        "scenario",
        "n_seeds",
        "settling_time_mean",
        "settling_time_mean_sd",
        "settling_time_mean_ci_lo",
        "settling_time_mean_ci_hi",
        "success_rate",
    ]
    print(summary[[c for c in cols if c in summary.columns]].round(2).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

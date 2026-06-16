#!/usr/bin/env python3
"""
Baseline: Fixed PID with standard industrial anti-windup — the fair classical
comparator the thesis was missing.

The original comparison set was:
  - Fixed PID + brake_integral_reset (env aid zeroing the integrator in the
    braking zone) — an unrealistically generous baseline, and
  - Fixed PID with NO anti-windup at all in the no-reset env — an
    unrealistically naive baseline (real PID implementations ship anti-windup).

This script adds the middle ground: plain fixed-gain PID equipped with
textbook anti-windup (Astrom & Hagglund, "PID Controllers", 1995):
  - back-calculation (tracking time constant Tt), and
  - conditional integration (clamping),
evaluated both in the reset-on (V4, target 5 m) and no-reset (V5, target 8 m)
environments. If anti-windup PID solves the no-reset task, the RL-vs-classical
comparison must be made against THIS baseline, not the naive one.

Usage:
  python stage_baseline_antiwindup_pid.py                 # all modes, both envs
  python stage_baseline_antiwindup_pid.py --modes backcalc --envs noreset
"""

import argparse
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")

from stage_baseline_fixed_pid import (
    ENV_KWARGS_V4,
    ENV_KWARGS_V5,
    EVAL_SEEDS,
    SCENARIOS_V5,
    make_env,
    plot_trajectories,
    run_episode,
)
from utils.pid import AntiWindupPIDController

OUTPUT_ROOT = Path("benchmark_results/baseline_antiwindup")

# In the no-reset env the integral reset never fires, so the only windup
# protection is the controller's own anti-windup. In the reset-on env the
# env aid additionally zeroes the integrator in the braking zone (|e| < 2 m).
ENV_VARIANTS = {
    "reset": dict(no_reset=False, kwargs=ENV_KWARGS_V4),
    "noreset": dict(no_reset=True, kwargs=ENV_KWARGS_V5),
}

MODES = ["backcalc", "clamp", "none"]


def install_antiwindup_pid(env, mode: str, tt: float) -> None:
    """Replace the env's internal PIDController with the anti-windup variant.

    AntiWindupPIDController exposes the same attribute surface (kp/ki/kd,
    setpoint, output_limits, _integral, reset(), update()) so the env's
    do_simulation loop and brake_integral_reset hook work unchanged.
    """
    base = env.unwrapped
    base.pid = AntiWindupPIDController(
        kp=base.base_gains["kp"],
        ki=base.base_gains["ki"],
        kd=base.base_gains["kd"],
        setpoint=base.target_pos,
        output_limits=(-1.0, 1.0),
        mode=mode,
        tt=tt,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--modes", type=str, default=",".join(MODES))
    parser.add_argument("--envs", type=str, default="reset,noreset")
    parser.add_argument("--tt", type=float, default=1.0, help="Back-calculation tracking time constant (s)")
    args = parser.parse_args()

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    env_names = [e.strip() for e in args.envs.split(",") if e.strip()]

    summary_rows = []
    for env_name in env_names:
        variant = ENV_VARIANTS[env_name]
        for mode in modes:
            out_dir = OUTPUT_ROOT / f"{mode}_{env_name}"
            out_dir.mkdir(parents=True, exist_ok=True)

            env = make_env(dynamic=False, no_reset=variant["no_reset"])
            target = float(env.unwrapped.target_pos)
            print(f"\n=== anti-windup mode={mode}  env={env_name}  target={target} m  tt={args.tt} ===")

            rows = []
            scenario_results = {}
            for scenario_name, mass, friction in SCENARIOS_V5:
                for seed in EVAL_SEEDS:
                    install_antiwindup_pid(env, mode=mode, tt=args.tt)
                    r = run_episode(env, scenario_name, mass, friction, seed)
                    r["aw_mode"] = mode
                    r["env_variant"] = env_name
                    rows.append({k: v for k, v in r.items() if not k.startswith("_")})
                    scenario_results[scenario_name] = r
                last = rows[-1]
                print(
                    f"  {scenario_name:20s} settling={last['settling_time_mean']:7.2f}s "
                    f"overshoot={last['overshoot_mean']:6.3f}m success={last['success_rate']:.0f}%"
                )
            env.close()

            df = pd.DataFrame(rows)
            df.to_csv(out_dir / "eval_seed_summary.csv", index=False)
            plot_trajectories(
                scenario_results,
                out_dir / "trajectories.png",
                f"Fixed PID + anti-windup ({mode}) — {env_name} env (target {target} m)",
                SCENARIOS_V5,
            )

            agg = (
                df.groupby("scenario")
                .agg(
                    settling=("settling_time_mean", "mean"),
                    overshoot=("overshoot_mean", "mean"),
                    iae=("iae_mean", "mean"),
                    success=("success_rate", "mean"),
                )
                .reset_index()
            )
            agg["aw_mode"] = mode
            agg["env_variant"] = env_name
            summary_rows.append(agg)

    summary = pd.concat(summary_rows, ignore_index=True)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUTPUT_ROOT / "antiwindup_comparison.csv", index=False)
    print("\n=== ANTI-WINDUP COMPARISON ===")
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

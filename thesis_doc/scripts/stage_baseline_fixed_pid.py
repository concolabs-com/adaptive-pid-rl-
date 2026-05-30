#!/usr/bin/env python3
"""
Baseline: Fixed PID — no RL, no adaptation.

Runs thesis_v4_cliff environment with constant action=[0,0,0] which maps to
base gains Kp=1.8, Ki=0.7, Kd=0.5. No training. Pure rollout across 3
scenarios to establish classical baseline for thesis comparison.

Usage:
  python stage_baseline_fixed_pid.py             # static eval (no disturbances)
  python stage_baseline_fixed_pid.py --dynamic   # dynamic eval (mid-episode disturbances)

Fixed gains:   Kp=1.8, Ki=0.7, Kd=0.5 (thesis_v4_cliff base, action=[0,0,0])
Scenarios:     Standard, Heavy and Slippery, Light and Grippy
Eval seeds:    70000-70009 (10 episodes per scenario)
Output:        benchmark_results/baseline_fixed_pid/          (static)
               benchmark_results/baseline_fixed_pid_dynamic/  (dynamic)
"""

import argparse
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import gymnasium as gym  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from agents.domain_randomization import DomainRandomizationWrapper  # noqa: E402
from envs.adaptive_suspension import AdaptiveSuspensionEnv  # noqa: E402

# thesis_v4_cliff env params (integral reset ON — for comparison with Stage 5a/5b)
ENV_KWARGS_V4 = dict(
    target_pos=5.0,
    hold_steps=25,
    max_episode_steps=5000,
    stop_tolerance=0.05,
    gain_base_kp=1.8,
    gain_base_ki=0.7,
    gain_base_kd=0.5,
    gain_delta_kp=1.0,
    gain_delta_ki=0.6,
    gain_delta_kd=2.0,
    gain_range_kp=(0.0, 6.0),
    gain_range_ki=(0.0, 3.0),
    gain_range_kd=(0.0, 5.0),
    safety_speed_governor_enabled=False,
    safety_hard_overshoot_m=-1.0,
    safety_overshoot_penalty=0.0,
    terminal_hold_bonus=50.0,
    terminal_hold_velocity_threshold=0.08,
    approach_progress_cutoff_m=2.0,
    brake_zone_vel_sq_coef=10.0,
    near_approach_zone_m=2.0,
    near_approach_coef=0.0,
    near_target_zone_m=0.8,
    near_target_coef=0.0,
    near_target_excess_thresh=0.10,
    near_target_excess_coef=0.0,
    decel_bonus_coef=10.0,
    near_target_init_prob=0.0,
    near_target_init_range_m=0.04,
    brake_integral_reset_enabled=True,
)

# thesis_v5_no_reset env params (integral reset OFF — true RL vs fixed PID test)
ENV_KWARGS_V5 = dict(
    target_pos=8.0,
    hold_steps=25,
    max_episode_steps=5000,
    stop_tolerance=0.05,
    gain_base_kp=1.8,
    gain_base_ki=0.7,
    gain_base_kd=0.5,
    gain_delta_kp=1.0,
    gain_delta_ki=0.6,
    gain_delta_kd=2.0,
    gain_range_kp=(0.0, 8.0),
    gain_range_ki=(0.0, 3.0),
    gain_range_kd=(0.0, 10.0),
    safety_speed_governor_enabled=False,
    safety_hard_overshoot_m=-1.0,
    safety_overshoot_penalty=0.0,
    terminal_hold_bonus=50.0,
    terminal_hold_velocity_threshold=0.08,
    approach_progress_cutoff_m=2.0,
    brake_zone_vel_sq_coef=10.0,
    near_approach_zone_m=2.0,
    near_approach_coef=0.0,
    near_target_zone_m=0.8,
    near_target_coef=0.0,
    near_target_excess_thresh=0.10,
    near_target_excess_coef=0.0,
    decel_bonus_coef=10.0,
    near_target_init_prob=0.0,
    near_target_init_range_m=0.04,
    brake_integral_reset_enabled=False,
)

# Matches thesis_v4_cliff disturbance settings used in Stage 5a/5b training
DYNAMIC_RANDOMIZATION_CONFIG = {
    "mass_range": (5.0, 20.0),
    "friction_range": (0.1, 2.0),
    "initial_randomization_enabled": False,  # scenarios set mass/friction directly
    "mid_episode_disturbance_enabled": True,
    "disturbance_mode": "step",
    "disturbance_step_range": (120, 220),
    "disturbance_mass_scale_range": (0.9, 1.3),
    "disturbance_friction_scale_range": (0.5, 1.4),
    "position_patch_enabled": True,
    "patch_x_range": (1.5, 2.4),
    "patch_friction_scale": 0.35,
}

SCENARIOS_V4 = [
    ("Standard", 10.0, 1.0),
    ("Heavy and Slippery", 20.0, 0.2),
    ("Light and Grippy", 5.0, 2.0),
    ("OOD Ultra Heavy", 35.0, 1.0),
    ("OOD Ultra Slippery", 20.0, 0.05),
]

SCENARIOS_V5 = [
    ("Standard", 10.0, 1.0),
    ("Heavy and Slippery", 20.0, 0.2),
    ("Light and Grippy", 5.0, 2.0),
]

TOLERANCE = 0.05
HOLD_STEPS = 25
MAX_STEPS = 5000
TARGET_POS = 5.0
EVAL_SEEDS = list(range(70000, 70010))

FIXED_ACTION = np.zeros(3, dtype=np.float32)  # Kp=1.8, Ki=0.7, Kd=0.5


def make_env(dynamic: bool, no_reset: bool):
    kwargs = ENV_KWARGS_V5 if no_reset else ENV_KWARGS_V4
    env = AdaptiveSuspensionEnv(**kwargs)
    if dynamic:
        env = DomainRandomizationWrapper(env, randomization_config=DYNAMIC_RANDOMIZATION_CONFIG)
    env = gym.wrappers.TimeLimit(env, max_episode_steps=MAX_STEPS)
    return env


def compute_settling_time(errors: np.ndarray, dt: float) -> tuple[float, int]:
    within = np.abs(errors) <= TOLERANCE
    n = len(within)
    for i in range(n):
        if within[i]:
            end = min(i + HOLD_STEPS, n)
            if np.all(within[i:end]):
                return float(i * dt), 1
    return float(MAX_STEPS * dt), 0


def run_episode(env, scenario_name: str, mass: float, friction: float, seed: int) -> dict:
    env.unwrapped.model.body_mass[1] = mass
    env.unwrapped.model.geom_friction[0, 0] = friction
    obs, _ = env.reset(seed=seed)

    dt = float(env.unwrapped.model.opt.timestep)
    positions, velocities, kp_hist, ki_hist, kd_hist, rewards = [], [], [], [], [], []

    for _ in range(MAX_STEPS):
        obs, reward, terminated, truncated, info = env.step(FIXED_ACTION)
        positions.append(float(info["state"]["pos"]))
        velocities.append(float(info["state"]["vel"]))
        kp_hist.append(float(info["gains"]["kp"]))
        ki_hist.append(float(info["gains"]["ki"]))
        kd_hist.append(float(info["gains"]["kd"]))
        rewards.append(float(reward))
        if terminated or truncated:
            break

    positions_arr = np.asarray(positions)
    errors_arr = TARGET_POS - positions_arr
    settling_s, settled = compute_settling_time(errors_arr, dt)
    overshoot = float(max(np.max(positions_arr) - TARGET_POS, 0.0))
    iae = float(np.sum(np.abs(errors_arr)) * dt)
    final_abs_error = float(abs(errors_arr[-1]))
    success = 1 if (settled and final_abs_error <= TOLERANCE) else 0

    return {
        "seed": seed,
        "scenario": scenario_name,
        "episodes": 1,
        "settling_time_mean": settling_s,
        "overshoot_mean": overshoot,
        "iae_mean": iae,
        "final_abs_error_mean": final_abs_error,
        "success_rate": 100.0 if success else 0.0,
        "mean_reward": float(np.mean(rewards)),
        "_positions": positions,
        "_velocities": velocities,
        "_kp": kp_hist,
        "_ki": ki_hist,
        "_kd": kd_hist,
        "_dt": dt,
    }


def plot_trajectories(scenario_results: dict, out_path: Path, title: str, scenarios: list) -> None:
    COLORS = {
        "Standard": "#2196F3",
        "Heavy and Slippery": "#F44336",
        "Light and Grippy": "#4CAF50",
    }
    fig, axes = plt.subplots(len(scenarios), 3, figsize=(15, 4 * len(scenarios)))
    fig.suptitle(title, fontsize=12)
    col_labels = ["Displacement (m)", "Velocity (m/s)", "PID Gains"]
    for j, lbl in enumerate(col_labels):
        axes[0, j].set_title(lbl, fontsize=10, fontweight="bold")

    for i, (scenario_name, mass, friction) in enumerate(scenarios):
        r = scenario_results[scenario_name]
        dt = r["_dt"]
        t = np.arange(len(r["_positions"])) * dt
        color = COLORS.get(scenario_name, "steelblue")

        ax0 = axes[i, 0]
        ax0.plot(t, r["_positions"], color=color)
        ax0.axhline(TARGET_POS, color="k", linestyle="--", linewidth=0.8, label="target")
        ax0.axhline(TARGET_POS + TOLERANCE, color="gray", linestyle=":", linewidth=0.6)
        ax0.axhline(TARGET_POS - TOLERANCE, color="gray", linestyle=":", linewidth=0.6)
        ax0.set_ylabel(scenario_name, fontsize=9)
        ax0.set_xlabel("Time (s)")
        ax0.legend(fontsize=7)

        ax1 = axes[i, 1]
        ax1.plot(t, r["_velocities"], color=color)
        ax1.axhline(0, color="k", linestyle="--", linewidth=0.6)
        ax1.set_xlabel("Time (s)")

        ax2 = axes[i, 2]
        ax2.plot(t, r["_kp"], label="Kp", color="tab:blue")
        ax2.plot(t, r["_ki"], label="Ki", color="tab:orange")
        ax2.plot(t, r["_kd"], label="Kd", color="tab:green")
        ax2.set_xlabel("Time (s)")
        ax2.legend(fontsize=7)

    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"  Trajectory plot saved: {out_path}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dynamic", action="store_true", default=False, help="Enable mid-episode disturbances + position patch"
    )
    parser.add_argument(
        "--no-reset",
        action="store_true",
        default=False,
        help="Disable brake_integral_reset — compare against Stage 5a/5b-NR",
    )
    args = parser.parse_args()

    if args.no_reset:
        output_dir = Path("benchmark_results/baseline_fixed_pid_no_reset")
        mode_label = "NO INTEGRAL RESET (target=8m, Kd fixed at 0.5)"
        SCENARIOS = SCENARIOS_V5
    elif args.dynamic:
        output_dir = Path("benchmark_results/baseline_fixed_pid_dynamic")
        mode_label = "DYNAMIC (mid-episode disturbances)"
        SCENARIOS = SCENARIOS_V4
    else:
        output_dir = Path("benchmark_results/baseline_fixed_pid")
        mode_label = "STATIC (fixed physics)"
        SCENARIOS = SCENARIOS_V4

    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("BASELINE: Fixed PID (no RL, no adaptation)")
    print(f"  mode    : {mode_label}")
    print("  gains   : Kp=1.8, Ki=0.7, Kd=0.5 (action=[0,0,0])")
    print("  env     : thesis_v4_cliff settings (no governor)")
    print("  seeds   : 70000-70009 (10 per scenario)")
    print(f"  output  : {output_dir}")
    print("=" * 65)
    print()

    env = make_env(dynamic=args.dynamic, no_reset=args.no_reset)

    all_records = []
    scenario_results_seed0 = {}

    for scenario_name, mass, friction in SCENARIOS:
        print(f"  Scenario: {scenario_name}  (mass={mass}kg, friction={friction})")
        for seed in EVAL_SEEDS:
            r = run_episode(env, scenario_name, mass, friction, seed)
            record = {k: v for k, v in r.items() if not k.startswith("_")}
            all_records.append(record)
            if seed == EVAL_SEEDS[0]:
                scenario_results_seed0[scenario_name] = r

        last = all_records[-1]
        print(
            f"    settling={last['settling_time_mean']:.3f}s  "
            f"overshoot={last['overshoot_mean']:.4f}m  "
            f"success={last['success_rate']:.0f}%  "
            f"final_err={last['final_abs_error_mean']:.4f}m"
        )

    env.close()

    df = pd.DataFrame(all_records)
    csv_path = output_dir / "eval_seed_summary.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n  Summary CSV saved: {csv_path}")

    title = f"Fixed PID Baseline ({mode_label}) — seed 70000"
    plot_trajectories(scenario_results_seed0, output_dir / "trajectories.png", title, SCENARIOS)

    print()
    print("=" * 65)
    print("SUMMARY (mean across seeds)")
    print("-" * 65)
    print(f"{'Scenario':<22} {'Settling(s)':>11} {'Overshoot(m)':>13} {'Success%':>9} {'FinalErr(m)':>12}")
    print("-" * 65)
    for scenario_name, _, _ in SCENARIOS:
        sub = df[df["scenario"] == scenario_name]
        if sub.empty:
            continue
        print(
            f"{scenario_name:<22} "
            f"{sub['settling_time_mean'].mean():>11.3f} "
            f"{sub['overshoot_mean'].mean():>13.4f} "
            f"{sub['success_rate'].mean():>9.1f} "
            f"{sub['final_abs_error_mean'].mean():>12.4f}"
        )
    print("=" * 65)

    return 0


if __name__ == "__main__":
    sys.exit(main())

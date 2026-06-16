#!/usr/bin/env python3
"""
Stage 4b: MRAC with a FEASIBLE reference model — fair classical adaptive baseline.

Fixes two problems that made the original stage4_mrac_baseline.py a strawman:

1. dt error: CONTROL_DT was hardcoded to 0.1 s but the env control period is
   env.dt = model.opt.timestep * frame_skip = 0.002 * 10 = 0.02 s. The reference
   model therefore evolved 5x faster than designed (effective tau_m = 0.6 s
   instead of 3 s), guaranteeing a persistently huge model-following error and
   divergent adaptation. All reported times were also 5x too large.

2. Reference-model feasibility: the vehicle's terminal speed is
   v_max = tau_max / (b * r) ≈ 0.5 / (0.03 / 0.03) = 0.5 m/s (actuator torque
   limit 0.5 N·m, joint damping 0.03, wheel radius 0.03 m). A first-order
   reference y_m(t) starting at 0 toward target d has initial speed d / tau_m.
   Feasibility requires d / tau_m <= v_max, i.e. tau_m >= d / v_max = 10 s for
   the 5 m task. The original tau_m = 3 s (effectively 0.6 s after the dt bug)
   demanded 1.67 m/s (8.3 m/s effective) — physically impossible, so e_mrac
   could never converge regardless of adaptation law.

This script evaluates MIT-rule MRAC with:
  - control_dt taken from env.dt (no hardcoding),
  - a (tau_m, sigma) config grid including feasible tau_m >= 10 s,
  - optional normalized MIT rule (Astrom & Wittenmark 1995, Sec. 5.3):
        dK/dt = -gamma * e_mrac * phi / (beta + phi^2)
    which bounds the effective adaptation rate when the regressor is large,
  - velocity regressor for Kp (correct sensitivity sign in all phases),
  - gain carry-over across episodes within a scenario (adaptive "learning").

Outputs one results directory per config under
benchmark_results/stage4b_mrac_feasible/<config>/ with the same CSV schema as
the original MRAC run so tables are directly comparable.
"""

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import mujoco
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from envs.adaptive_suspension import AdaptiveSuspensionEnv

OUTPUT_ROOT = Path("benchmark_results/stage4b_mrac_feasible")

SCENARIOS = [
    {"name": "Standard", "mass": 10.0, "friction": 1.0},
    {"name": "Heavy and Slippery", "mass": 20.0, "friction": 0.2},
    {"name": "Light and Grippy", "mass": 5.0, "friction": 2.0},
]

# Gain mapping must match AdaptiveSuspensionEnv constructor defaults.
KP_BASE, KP_DELTA, KP_RANGE = 1.8, 1.0, (0.0, 6.0)
KI_BASE, KI_DELTA, KI_RANGE = 0.7, 0.6, (0.0, 3.0)
KD_BASE, KD_DELTA, KD_RANGE = 0.0, 2.0, (0.0, 5.0)

N_EPISODES = 30
TARGET_POS = 5.0
MAX_STEPS = 5000
TOLERANCE = 0.05
HOLD_STEPS = 25
SEEDS = list(range(N_EPISODES))


@dataclass(frozen=True)
class MRACConfig:
    name: str
    tau_m: float
    sigma: float
    gamma_p: float = 0.40
    gamma_i: float = 0.06
    gamma_d: float = 0.02
    normalized: bool = True
    beta: float = 1.0  # normalization constant in gamma*e*phi/(beta+phi^2)


# Config grid: feasible tau_m (>=10 s for the 5 m task at v_max=0.5 m/s),
# one borderline-infeasible tau_m=5 for contrast, sigma on/off.
CONFIG_GRID = [
    MRACConfig(name="tau10_sigma005", tau_m=10.0, sigma=0.05),
    MRACConfig(name="tau10_sigma000", tau_m=10.0, sigma=0.0),
    MRACConfig(name="tau15_sigma005", tau_m=15.0, sigma=0.05),
    MRACConfig(name="tau5_sigma005", tau_m=5.0, sigma=0.05),
]


def gains_to_action(kp: float, ki: float, kd: float) -> np.ndarray:
    a0 = float(np.clip((kp - KP_BASE) / KP_DELTA, -1.0, 1.0))
    a1 = float(np.clip((ki - KI_BASE) / KI_DELTA, -1.0, 1.0))
    a2 = float(np.clip((kd - KD_BASE) / KD_DELTA, -1.0, 1.0))
    return np.array([a0, a1, a2], dtype=np.float32)


class FeasibleMRAC:
    """MIT-rule MRAC with feasible first-order reference model.

    Adaptation laws (normalized MIT rule when cfg.normalized):
        e_mrac = pos - y_m
        dKp/dt = -gamma_p * e_mrac * vel        / (beta + vel^2)
        dKi/dt = -gamma_i * e_mrac * int_ep     / (beta + int_ep^2)
        dKd/dt = -gamma_d * e_mrac * dep_dt     / (beta + dep_dt^2)
    each with sigma-leakage  - sigma * (K - K_base)  added when sigma > 0.
    """

    def __init__(self, target_pos: float, cfg: MRACConfig) -> None:
        self.target_pos = float(target_pos)
        self.cfg = cfg
        self.kp = float(KP_BASE)
        self.ki = float(KI_BASE)
        self.kd = float(KD_BASE)
        self._y_m = 0.0
        self._integral_ep = 0.0
        self._prev_ep = 0.0

    def reset_episode(self, initial_pos: float = 0.0) -> None:
        self._y_m = float(initial_pos)
        self._integral_ep = 0.0
        self._prev_ep = self.target_pos - initial_pos

    def step(self, pos: float, vel: float, dt: float) -> np.ndarray:
        cfg = self.cfg
        ep = self.target_pos - pos
        dep_dt = (ep - self._prev_ep) / dt if dt > 0.0 else 0.0
        self._integral_ep += ep * dt
        self._prev_ep = ep

        # Reference model (feasible: initial slope = target/tau_m <= v_max).
        self._y_m += dt * (-(self._y_m - self.target_pos) / cfg.tau_m)
        e_mrac = pos - self._y_m

        def update(gain, gamma, phi, base):
            if cfg.normalized:
                grad = gamma * e_mrac * phi / (cfg.beta + phi * phi)
            else:
                grad = gamma * e_mrac * phi
            leak = cfg.sigma * (gain - base)
            return gain - (grad + leak) * dt

        self.kp = float(np.clip(update(self.kp, cfg.gamma_p, vel, KP_BASE), *KP_RANGE))
        self.ki = float(np.clip(update(self.ki, cfg.gamma_i, self._integral_ep, KI_BASE), *KI_RANGE))
        self.kd = float(np.clip(update(self.kd, cfg.gamma_d, dep_dt, KD_BASE), *KD_RANGE))

        return gains_to_action(self.kp, self.ki, self.kd)


def make_env() -> AdaptiveSuspensionEnv:
    return AdaptiveSuspensionEnv(
        target_pos=TARGET_POS,
        hold_steps=HOLD_STEPS,
        max_episode_steps=MAX_STEPS,
        stop_tolerance=TOLERANCE,
        gain_base_kp=KP_BASE,
        gain_base_ki=KI_BASE,
        gain_base_kd=KD_BASE,
        gain_delta_kp=KP_DELTA,
        gain_delta_ki=KI_DELTA,
        gain_delta_kd=KD_DELTA,
        gain_range_kp=KP_RANGE,
        gain_range_ki=KI_RANGE,
        gain_range_kd=KD_RANGE,
    )


def apply_scenario_physics(env: AdaptiveSuspensionEnv, mass: float, friction: float) -> None:
    env.model.body_mass[1] = float(mass)
    env.model.geom_friction[0, 0] = float(friction)
    mujoco.mj_forward(env.model, env.data)


def run_episode(env, controller, control_dt: float, seed: int) -> dict:
    obs, _ = env.reset(seed=seed)
    controller.reset_episode(initial_pos=float(obs[0]))

    positions, errors = [], []
    kps, kis, kds = [], [], []
    steps_in_tol = 0
    success = False
    iae = 0.0

    for _ in range(MAX_STEPS):
        pos = float(obs[0])
        vel = float(obs[1])
        error = TARGET_POS - pos
        positions.append(pos)
        errors.append(error)
        kps.append(controller.kp)
        kis.append(controller.ki)
        kds.append(controller.kd)
        iae += abs(error) * control_dt

        action = controller.step(pos, vel, control_dt)
        obs, _, terminated, truncated, _ = env.step(action)

        steps_in_tol = (steps_in_tol + 1) if abs(error) < TOLERANCE else 0
        if steps_in_tol >= HOLD_STEPS:
            success = True
            break
        if terminated or truncated:
            break

    errors_arr = np.asarray(errors, dtype=np.float64)
    positions_arr = np.asarray(positions, dtype=np.float64)

    within = np.abs(errors_arr) <= TOLERANCE
    settling_step = MAX_STEPS
    if success and np.any(within):
        suffix = np.flip(np.cumprod(np.flip(within).astype(int))).astype(bool)
        idx = np.flatnonzero(suffix)
        if len(idx):
            settling_step = int(idx[0])
    settling_time_s = settling_step * control_dt if success else float(MAX_STEPS * control_dt)

    return {
        "success": int(success),
        "settling_time_s": settling_time_s,
        "overshoot_m": float(max(0.0, np.max(positions_arr) - TARGET_POS)),
        "iae": iae,
        "final_abs_error_m": float(abs(errors_arr[-1])) if len(errors_arr) else float("nan"),
        "n_steps": len(errors),
        "final_kp": controller.kp,
        "final_ki": controller.ki,
        "final_kd": controller.kd,
        "kp_trace": kps,
        "ki_trace": kis,
        "kd_trace": kds,
        "pos_trace": positions,
    }


def evaluate_config(cfg: MRACConfig) -> pd.DataFrame:
    out_dir = OUTPUT_ROOT / cfg.name
    out_dir.mkdir(parents=True, exist_ok=True)

    all_rows = []
    trace_fig, trace_axes = plt.subplots(1, len(SCENARIOS), figsize=(5 * len(SCENARIOS), 4))

    for ax, scenario in zip(np.atleast_1d(trace_axes), SCENARIOS):
        env = make_env()
        control_dt = float(env.dt)  # 0.02 s — from the model, never hardcoded
        controller = FeasibleMRAC(target_pos=TARGET_POS, cfg=cfg)

        print(f"\n[{cfg.name}] {scenario['name']}  mass={scenario['mass']} fr={scenario['friction']}  dt={control_dt}")
        for ep, seed in enumerate(SEEDS):
            apply_scenario_physics(env, scenario["mass"], scenario["friction"])
            r = run_episode(env, controller, control_dt, seed)
            row = {k: v for k, v in r.items() if not k.endswith("_trace")}
            row.update(
                episode=ep,
                scenario=scenario["name"],
                mass=scenario["mass"],
                friction=scenario["friction"],
                seed=seed,
                config=cfg.name,
                tau_m=cfg.tau_m,
                sigma=cfg.sigma,
            )
            all_rows.append(row)
            if ep in (0, N_EPISODES - 1):
                t = np.arange(len(r["pos_trace"])) * control_dt
                ax.plot(t, r["pos_trace"], label=f"ep{ep}", alpha=0.8)
            if ep % 10 == 0 or ep == N_EPISODES - 1:
                print(
                    f"  ep={ep:2d} ok={r['success']} t={r['settling_time_s']:6.1f}s "
                    f"ov={r['overshoot_m']:.2f} kp={r['final_kp']:.2f} ki={r['final_ki']:.2f} kd={r['final_kd']:.2f}"
                )
        env.close()
        ax.axhline(TARGET_POS, color="k", ls="--", lw=0.8)
        ax.set_title(f"{scenario['name']}")
        ax.set_xlabel("time (s)")
        ax.set_ylabel("position (m)")
        ax.legend()

    trace_fig.suptitle(f"MRAC feasible — {cfg.name} (tau_m={cfg.tau_m}s, sigma={cfg.sigma})")
    trace_fig.tight_layout()
    trace_fig.savefig(out_dir / "mrac_trajectories.png", dpi=150)
    plt.close(trace_fig)

    df = pd.DataFrame(all_rows)
    df.to_csv(out_dir / "mrac_raw.csv", index=False)

    metrics = ["settling_time_s", "overshoot_m", "iae", "final_abs_error_m", "success"]
    rows = []
    for scenario, grp in df.groupby("scenario"):
        row = {"scenario": scenario, "n_episodes": len(grp), "config": cfg.name}
        for m in metrics:
            row[f"{m}_mean"] = float(grp[m].mean())
            row[f"{m}_std"] = float(grp[m].std())
        rows.append(row)
    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir / "mrac_summary.csv", index=False)
    print(f"\n[{cfg.name}] summary:\n{summary.to_string(index=False)}")
    return df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", type=str, default="", help="Comma-separated config names to run (default: all)")
    args = parser.parse_args()

    selected = [c.strip() for c in args.configs.split(",") if c.strip()]
    grid = [c for c in CONFIG_GRID if not selected or c.name in selected]

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    frames = [evaluate_config(cfg) for cfg in grid]

    combined = pd.concat(frames, ignore_index=True)
    combined.to_csv(OUTPUT_ROOT / "mrac_all_configs_raw.csv", index=False)

    pivot = (
        combined.groupby(["config", "scenario"])
        .agg(
            success_rate=("success", "mean"),
            settling_mean=("settling_time_s", "mean"),
            overshoot_mean=("overshoot_m", "mean"),
        )
        .reset_index()
    )
    pivot.to_csv(OUTPUT_ROOT / "mrac_config_comparison.csv", index=False)
    print("\n=== CONFIG COMPARISON ===")
    print(pivot.to_string(index=False))


if __name__ == "__main__":
    main()

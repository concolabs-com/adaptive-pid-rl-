#!/usr/bin/env python3
"""
Mass-sweep discriminativeness evaluation.

Sweeps vehicle mass over a continuum (default 5 -> 50 kg, training range was
5-20 kg) at nominal friction/actuator and measures settling time, overshoot,
and success for each controller:

  - fixed       : Fixed PID, action [0,0,0] (Kp=1.8, Ki=0.7, Kd=0.5)
  - aw_backcalc : Fixed PID + back-calculation anti-windup (fair classical)
  - rl_blind    : Stage 5b/6b blind agent checkpoint (6-dim obs x 10 frames)
  - rl_context  : Stage 5a/6a context agent checkpoint; evaluated BOTH with
                  the corrected context push (true mass/friction scales) and
                  with the legacy broken context (1,1) to quantify the impact
                  of the v1 static-eval context bug.

Produces the per-mass curves figure for the discriminativeness analysis.

Usage:
  python eval_mass_sweep.py                       # old stage5 checkpoints
  python eval_mass_sweep.py --context-model ... --blind-model ... --out-dir ...
"""

import argparse
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from agents.model import Agent
from stage2_meta_rl_reproduction import (
    apply_eval_physics,
    compute_hold_success,
    compute_settling_time,
    make_eval_env,
    protocol_preset_defaults,
)
from utils.pid import AntiWindupPIDController

TOLERANCE = 0.05
HOLD_STEPS = 25
MAX_STEPS = 5000
TARGET = 5.0
STACK_SIZE = 10


def build_env(obs_keep_dims: int):
    # thesis_v4_cliff env parameters (reward terms irrelevant for eval metrics,
    # but brake_integral_reset and gain mapping must match training).
    p = protocol_preset_defaults("thesis_v4_cliff")
    return make_eval_env(
        stack_size=STACK_SIZE,
        target_pos=TARGET,
        max_eval_steps=MAX_STEPS,
        hold_steps=HOLD_STEPS,
        obs_keep_dims=obs_keep_dims,
        stop_tolerance=TOLERANCE,
        gain_base_kp=1.8,
        gain_base_ki=0.7,
        gain_base_kd=float(p["gain_base_kd"]),
        gain_delta_kp=1.0,
        gain_delta_ki=0.6,
        gain_delta_kd=float(p["gain_delta_kd"]),
        gain_range_kp=(0.0, 6.0),
        gain_range_ki=(0.0, 3.0),
        gain_range_kd=(0.0, 5.0),
        terminal_hold_bonus=50.0,
        terminal_hold_velocity_threshold=0.08,
        approach_progress_cutoff_m=float(p["approach_progress_cutoff_m"]),
        brake_zone_vel_sq_coef=float(p["brake_zone_vel_sq_coef"]),
        near_approach_zone_m=float(p["near_approach_zone_m"]),
        near_approach_coef=0.0,
        near_target_zone_m=0.8,
        near_target_coef=0.0,
        near_target_excess_thresh=0.10,
        near_target_excess_coef=0.0,
        decel_bonus_coef=float(p["decel_bonus_coef"]),
        brake_integral_reset_enabled=True,
    )


def rollout(
    env, action_fn, mass: float, friction: float, actuator: float, push_context: bool, seed: int = 70000
) -> dict:
    # The PLANT physics (mass, friction, actuator strength) are always applied
    # via apply_eval_physics so every controller experiences the same dynamics.
    # push_context only governs whether the context OBSERVATION carries the true
    # scales: blind/fixed controllers drop or ignore those dims, but for the
    # context agent push_context=False emulates the broken (1,1,1) context.
    apply_eval_physics(env, mass, friction, actuator)
    obs, _ = env.reset(seed=seed)
    apply_eval_physics(env, mass, friction, actuator)
    if not push_context:
        # Nominal context for subsequent steps. Blind agent drops these obs
        # dims and Fixed PID ignores the observation, so this only affects the
        # context agent's broken-context comparison. obs shape is left intact.
        env.unwrapped.set_disturbance_context(mass_scale=1.0, friction_scale=1.0, actuator_scale=1.0)

    positions = []
    dt = float(env.unwrapped.dt)
    for _ in range(MAX_STEPS):
        action = action_fn(obs)
        obs, _, terminated, truncated, info = env.step(action)
        positions.append(float(info["state"]["pos"]))
        if terminated or truncated:
            break

    pos = np.asarray(positions)
    errors = TARGET - pos
    settling, settled = compute_settling_time(errors, dt, TOLERANCE, MAX_STEPS * dt)
    return {
        "settling_time_s": settling,
        "settled": settled,
        "overshoot": float(max(0.0, np.max(pos) - TARGET)) if len(pos) else float("nan"),
        "success": compute_hold_success(errors, TOLERANCE, HOLD_STEPS) if len(pos) else 0,
        "steps": len(pos),
    }


def agent_action_fn(agent):
    def fn(obs):
        with torch.no_grad():
            t = torch.as_tensor(np.asarray(obs), dtype=torch.float32).unsqueeze(0)
            a = agent.actor_mean(t.reshape(1, -1))
            return torch.clamp(a, -1.0, 1.0).squeeze(0).numpy()

    return fn


def fixed_action_fn(_obs):
    return np.zeros(3, dtype=np.float32)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--context-model", default="benchmark_results/stage5a_context_cliff/seed_7/models/meta_rl_agent.pth"
    )
    ap.add_argument("--blind-model", default="benchmark_results/stage5b_blind_cliff/seed_7/models/meta_rl_agent.pth")
    ap.add_argument(
        "--context-dims", type=int, default=8, help="obs dims of the context checkpoint (8 = old, 9 = stage6)"
    )
    ap.add_argument(
        "--axis",
        choices=["mass", "actuator"],
        default="mass",
        help="Which hidden parameter to sweep (other axes held nominal)",
    )
    ap.add_argument("--masses", default="5,7.5,10,12.5,15,17.5,20,25,30,35,40,45,50")
    ap.add_argument("--actuators", default="0.4,0.5,0.6,0.8,1.0,1.2,1.4,1.6,1.8,2.0")
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    if args.axis == "mass":
        sweep_vals = [float(m) for m in args.masses.split(",")]
        train_lo, train_hi = 5.0, 20.0
    else:
        sweep_vals = [float(a) for a in args.actuators.split(",")]
        train_lo, train_hi = 0.6, 1.4
    out_dir = Path(args.out_dir) if args.out_dir else Path(f"benchmark_results/{args.axis}_sweep")
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- controllers ---
    env_full = build_env(obs_keep_dims=args.context_dims)
    env_blind = build_env(obs_keep_dims=6)

    context_agent = Agent(env_full)
    context_agent.load_state_dict(torch.load(args.context_model, map_location="cpu", weights_only=True))
    context_agent.eval()
    blind_agent = Agent(env_blind)
    blind_agent.load_state_dict(torch.load(args.blind_model, map_location="cpu", weights_only=True))
    blind_agent.eval()

    def install_aw(env):
        base = env.unwrapped
        base.pid = AntiWindupPIDController(
            kp=base.base_gains["kp"],
            ki=base.base_gains["ki"],
            kd=base.base_gains["kd"],
            setpoint=base.target_pos,
            output_limits=(-1.0, 1.0),
            mode="backcalc",
            tt=1.0,
        )

    def phys(v):
        # Map a sweep value to (mass, friction, actuator); non-swept axes nominal.
        return (v, 1.0, 1.0) if args.axis == "mass" else (10.0, 1.0, v)

    from utils.pid import PIDController

    rows = []
    for v in sweep_vals:
        m, fr, act = phys(v)
        r = rollout(env_blind, fixed_action_fn, m, fr, act, push_context=False)
        rows.append({"controller": "Fixed PID", args.axis: v, **r})

        install_aw(env_blind)
        r = rollout(env_blind, fixed_action_fn, m, fr, act, push_context=False)
        rows.append({"controller": "PID + anti-windup", args.axis: v, **r})
        env_blind.unwrapped.pid = PIDController(setpoint=env_blind.unwrapped.target_pos, output_limits=(-1.0, 1.0))

        r = rollout(env_blind, agent_action_fn(blind_agent), m, fr, act, push_context=False)
        rows.append({"controller": "RL blind", args.axis: v, **r})

        r = rollout(env_full, agent_action_fn(context_agent), m, fr, act, push_context=True)
        rows.append({"controller": "RL context (true ctx)", args.axis: v, **r})

        done = [row for row in rows if row[args.axis] == v]
        print(
            f"{args.axis}={v:6.2f}  "
            + "  ".join(
                f"{d['controller']}: {d['settling_time_s']:6.2f}s/{'OK' if d['success'] else 'FAIL'}" for d in done
            )
        )

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / f"{args.axis}_sweep_raw.csv", index=False)

    xlabel = "Mass (kg)" if args.axis == "mass" else "Actuator strength scale"
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for name, grp in df.groupby("controller"):
        grp = grp.sort_values(args.axis)
        axes[0].plot(grp[args.axis], grp["settling_time_s"], marker="o", ms=3, label=name)
        axes[1].plot(grp[args.axis], grp["overshoot"], marker="o", ms=3, label=name)
        axes[2].plot(grp[args.axis], grp["success"] * 100.0, marker="o", ms=3, label=name)
    for ax, ylabel in zip(axes, ["Settling time (s)", "Overshoot (m)", "Success (%)"]):
        ax.axvspan(train_lo, train_hi, alpha=0.08, color="green")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.suptitle(f"{args.axis.capitalize()} sweep (green band = training range), target=5 m")
    fig.tight_layout()
    fig.savefig(out_dir / f"{args.axis}_sweep.png", dpi=150)
    print(f"\nSaved {out_dir}/{args.axis}_sweep_raw.csv and {args.axis}_sweep.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""
RQ2 mechanism analysis — gain trajectories and adaptation lag.

Rolls out the Stage 6 context and blind agents at a grid of fixed operating
points (mass x actuator) and records the scheduled gains Kp/Ki/Kd over the
episode. Produces:

1. Steady-state gain vs hidden parameter — does the context agent learn a
   monotone parameter -> gain mapping (true gain scheduling)?
2. Adaptation lag — at episode start, how long until the blind agent's gain
   vector converges to the context agent's steady-state schedule for the same
   physics:  lag = first t with ||K_blind(t) - K_context_ss|| < eps.
   This quantifies the cost of inferring dynamics from trajectory vs reading
   them from the observation.

Output: benchmark_results/gain_analysis/
"""

import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from agents.model import Agent
from stage2_meta_rl_reproduction import apply_eval_physics, make_eval_env, protocol_preset_defaults

OUTPUT_DIR = Path("benchmark_results/gain_analysis")
MODEL_CTX = "benchmark_results/stage6a_context_hipmdp/seed_7/models/meta_rl_agent.pth"
MODEL_BLIND = "benchmark_results/stage6b_blind_hipmdp/seed_7/models/meta_rl_agent.pth"
STACK = 10
TARGET = 5.0
MAX_STEPS = 1500


def build_env(keep):
    p = protocol_preset_defaults("thesis_v6_hipmdp")
    return make_eval_env(
        stack_size=STACK,
        target_pos=TARGET,
        max_eval_steps=MAX_STEPS,
        hold_steps=25,
        obs_keep_dims=keep,
        stop_tolerance=0.05,
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
        approach_progress_cutoff_m=2.0,
        brake_zone_vel_sq_coef=10.0,
        near_approach_zone_m=2.0,
        near_approach_coef=0.0,
        near_target_zone_m=0.8,
        near_target_coef=0.0,
        near_target_excess_thresh=0.10,
        near_target_excess_coef=0.0,
        decel_bonus_coef=10.0,
        brake_integral_reset_enabled=True,
    )


def rollout_gains(env, agent, mass, friction, actuator, push_context):
    apply_eval_physics(env, mass, friction, actuator)
    obs, _ = env.reset(seed=70000)
    apply_eval_physics(env, mass, friction, actuator)
    if not push_context:
        env.unwrapped.set_disturbance_context(1.0, 1.0, 1.0)
    gains = []
    for _ in range(MAX_STEPS):
        with torch.no_grad():
            t = torch.as_tensor(np.asarray(obs), dtype=torch.float32).unsqueeze(0)
            a = torch.clamp(agent.actor_mean(t.reshape(1, -1)), -1, 1).squeeze(0).numpy()
        obs, _, term, trunc, info = env.step(a)
        g = info["gains"]
        gains.append([g["kp"], g["ki"], g["kd"]])
        if term or trunc:
            break
    return np.asarray(gains)


def steady_state(gains, window=25):
    # Mean gains over the last `window` pre-termination steps (the hold phase).
    return gains[-window:].mean(axis=0) if len(gains) >= window else gains.mean(axis=0)


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    env_ctx = build_env(0)  # 9-dim context
    env_blind = build_env(6)  # 6-dim blind
    ctx = Agent(env_ctx)
    ctx.load_state_dict(torch.load(MODEL_CTX, map_location="cpu", weights_only=True))
    ctx.eval()
    blind = Agent(env_blind)
    blind.load_state_dict(torch.load(MODEL_BLIND, map_location="cpu", weights_only=True))
    blind.eval()

    # --- 1. Steady-state gain vs parameter (sweep actuator at nominal mass) ---
    ss_rows = []
    for act in [0.6, 0.8, 1.0, 1.2, 1.4]:
        for mass in [5.0, 10.0, 20.0]:
            g_ctx = steady_state(rollout_gains(env_ctx, ctx, mass, 1.0, act, True))
            ss_rows.append(dict(agent="context", mass=mass, actuator=act, kp=g_ctx[0], ki=g_ctx[1], kd=g_ctx[2]))
            g_bl = steady_state(rollout_gains(env_blind, blind, mass, 1.0, act, False))
            ss_rows.append(dict(agent="blind", mass=mass, actuator=act, kp=g_bl[0], ki=g_bl[1], kd=g_bl[2]))
    ss = pd.DataFrame(ss_rows)
    ss.to_csv(OUTPUT_DIR / "steady_state_gains.csv", index=False)

    # --- 2. Adaptation lag: blind gains converging to context steady state ---
    lag_rows = []
    EPS = 0.25  # gain-vector L2 distance threshold
    for mass, act in [(5.0, 1.0), (10.0, 1.0), (20.0, 1.0), (10.0, 0.6), (10.0, 1.4)]:
        g_ctx = rollout_gains(env_ctx, ctx, mass, 1.0, act, True)
        g_bl = rollout_gains(env_blind, blind, mass, 1.0, act, False)
        ctx_ss = steady_state(g_ctx)
        dt = float(env_blind.unwrapped.dt)
        dist = np.linalg.norm(g_bl - ctx_ss[None, :], axis=1)
        below = np.where(dist < EPS)[0]
        lag = float(below[0] * dt) if len(below) else float("nan")
        # Also: distance between the two agents' OWN trajectories early on.
        n = min(len(g_ctx), len(g_bl))
        early_gap = float(np.linalg.norm(g_ctx[:n] - g_bl[:n], axis=1)[:50].mean())
        lag_rows.append(
            dict(
                mass=mass,
                actuator=act,
                adaptation_lag_s=lag,
                ctx_ss_kp=ctx_ss[0],
                ctx_ss_ki=ctx_ss[1],
                ctx_ss_kd=ctx_ss[2],
                early_gain_gap=early_gap,
            )
        )
        print(
            f"mass={mass:4.1f} act={act:.1f}  blind->ctx_ss lag={lag if lag==lag else float('nan'):.2f}s  "
            f"early_gap={early_gap:.3f}  ctx_ss Kd={ctx_ss[2]:.2f}"
        )
    lag = pd.DataFrame(lag_rows)
    lag.to_csv(OUTPUT_DIR / "adaptation_lag.csv", index=False)

    # --- Plot: steady-state Kd vs actuator (mass=10), both agents ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    sub = ss[ss.mass == 10.0].sort_values("actuator")
    for gi, gname in enumerate(["kp", "ki", "kd"]):
        for ag, color in [("context", "#1E88E5"), ("blind", "#43A047")]:
            d = sub[sub.agent == ag]
            axes[gi].plot(d["actuator"], d[gname], marker="o", color=color, label=ag)
        axes[gi].set_xlabel("Actuator scale")
        axes[gi].set_ylabel(f"steady-state {gname.upper()}")
        axes[gi].grid(alpha=0.3)
        axes[gi].legend(fontsize=8)
    fig.suptitle("Learned gain schedule vs actuator strength (mass=10 kg)")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "gain_vs_actuator.png", dpi=140)
    plt.close(fig)

    print(f"\nSaved steady_state_gains.csv, adaptation_lag.csv, gain_vs_actuator.png to {OUTPUT_DIR}")
    print("\n=== steady-state gains (mass=10) ===")
    print(sub[["agent", "actuator", "kp", "ki", "kd"]].round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

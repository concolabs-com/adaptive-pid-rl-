#!/usr/bin/env python3
"""
Stage 6 shock probe — mid-approach step change in a hidden parameter.

Fires a large parameter shock during the approach phase (before the braking
zone) and measures how each controller recovers:

  - mass shock:      mass x2.5 at step 30-60
  - actuator shock:  actuator x0.5 at step 30-60 (motor suddenly weakened)

Controllers:
  Fixed PID            action=[0,0,0] — cannot react
  Stage 6a (context)   sees the new mass/actuator scale in obs immediately
  Stage 6b (blind)     must re-infer the change from trajectory

Recovery metrics (post-shock):
  post_shock_max_err   largest |error| reached after the shock fires
  recovery_time_s      time from shock to re-entry into the +-0.05 m band
  success / overshoot / settling as usual

Output: benchmark_results/stage6_shock_probe/
"""

import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import gymnasium as gym
import matplotlib.pyplot as plt
import torch

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from agents.domain_randomization import DomainRandomizationWrapper
from agents.model import Agent
from envs.adaptive_suspension import AdaptiveSuspensionEnv
from stage2_meta_rl_reproduction import ObservationFeatureSelectWrapper, protocol_preset_defaults

OUTPUT_DIR = Path("benchmark_results/stage6_shock_probe")
MODEL_CTX = Path("benchmark_results/stage6a_context_hipmdp/seed_7/models/meta_rl_agent.pth")
MODEL_BLIND = Path("benchmark_results/stage6b_blind_hipmdp/seed_7/models/meta_rl_agent.pth")

STACK_SIZE = 10
TOLERANCE = 0.05
HOLD_STEPS = 25
MAX_STEPS = 5000
TARGET_POS = 5.0
EVAL_SEEDS = list(range(70000, 70010))
FIXED_ACTION = np.zeros(3, dtype=np.float32)

_P = protocol_preset_defaults("thesis_v6_hipmdp")
ENV_KWARGS = dict(
    target_pos=5.0,
    hold_steps=25,
    max_episode_steps=MAX_STEPS,
    stop_tolerance=0.05,
    gain_base_kp=1.8,
    gain_base_ki=0.7,
    gain_base_kd=float(_P["gain_base_kd"]),
    gain_delta_kp=1.0,
    gain_delta_ki=0.6,
    gain_delta_kd=float(_P["gain_delta_kd"]),
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
)

SHOCKS = {
    "mass_x2.5": dict(disturbance_mass_scale_range=(2.5, 2.5), disturbance_actuator_scale_range=(1.0, 1.0)),
    "actuator_x0.5": dict(disturbance_mass_scale_range=(1.0, 1.0), disturbance_actuator_scale_range=(0.5, 0.5)),
}


def shock_config(shock: dict) -> dict:
    cfg = {
        "mass_range": (5.0, 20.0),
        "friction_range": (0.1, 2.0),
        "actuator_strength_range": (1.0, 1.0),
        "initial_randomization_enabled": False,
        "mid_episode_disturbance_enabled": True,
        "disturbance_mode": "step",
        "disturbance_step_range": (30, 60),
        "disturbance_friction_scale_range": (1.0, 1.0),
        "position_patch_enabled": False,
    }
    cfg.update(shock)
    return cfg


def make_eval_env(mass: float, cfg: dict, obs_keep_dims: int):
    base = AdaptiveSuspensionEnv(**ENV_KWARGS)
    base.model.body_mass[1] = mass
    env = DomainRandomizationWrapper(base, randomization_config=cfg)
    if obs_keep_dims > 0:
        env = ObservationFeatureSelectWrapper(env, keep_dims=obs_keep_dims)
    env = gym.wrappers.TimeLimit(env, max_episode_steps=MAX_STEPS)
    env = gym.wrappers.FrameStackObservation(env, stack_size=STACK_SIZE)
    return env


def settling_time(errors: np.ndarray, dt: float):
    within = np.abs(errors) <= TOLERANCE
    n = len(within)
    for i in range(n):
        if within[i] and np.all(within[i : min(i + HOLD_STEPS, n)]):
            return float(i * dt), 1
    return float(MAX_STEPS * dt), 0


def run_episode(env, agent, seed, mass, cfg):
    env.unwrapped.model.body_mass[1] = mass
    obs, _ = env.reset(seed=seed)
    dt = float(env.unwrapped.dt)
    positions, kd_hist = [], []
    shock_step = None
    for step_i in range(MAX_STEPS):
        if agent is None:
            action = FIXED_ACTION
        else:
            with torch.no_grad():
                obs_t = torch.as_tensor(np.asarray(obs), dtype=torch.float32).unsqueeze(0)
                action = torch.clamp(agent.actor_mean(obs_t.reshape(1, -1)), -1, 1).squeeze(0).numpy()
        obs, _, term, trunc, info = env.step(action)
        positions.append(float(info["state"]["pos"]))
        kd_hist.append(float(info["gains"]["kd"]))
        if info.get("external_factors", {}).get("disturbance_fired") and shock_step is None:
            shock_step = step_i
        if term or trunc:
            break

    pos = np.asarray(positions)
    err = TARGET_POS - pos
    settle, settled = settling_time(err, dt)
    overshoot = float(max(np.max(pos) - TARGET_POS, 0.0))
    final_abs = float(abs(err[-1]))
    success = 1 if (settled and final_abs <= TOLERANCE) else 0

    # Post-shock recovery.
    post_max_err, recovery_s = float("nan"), float("nan")
    if shock_step is not None and shock_step < len(err):
        post = np.abs(err[shock_step:])
        post_max_err = float(np.max(post))
        for j in range(shock_step, len(err)):
            if np.all(np.abs(err[j : min(j + HOLD_STEPS, len(err))]) <= TOLERANCE):
                recovery_s = float((j - shock_step) * dt)
                break
    return dict(
        seed=seed,
        settling_time_s=settle,
        overshoot=overshoot,
        final_abs_error=final_abs,
        success_rate=100.0 * success,
        post_shock_max_err=post_max_err,
        recovery_time_s=recovery_s,
        shock_step=shock_step if shock_step is not None else -1,
        _pos=positions,
        _kd=kd_hist,
        _dt=dt,
    )


def main() -> int:
    for p in (MODEL_CTX, MODEL_BLIND):
        if not p.exists():
            print(f"ERROR missing model {p}", file=sys.stderr)
            return 1
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    agents_cfg = [("Fixed PID", None, 0), ("Stage 6a (Context)", MODEL_CTX, 0), ("Stage 6b (Blind)", MODEL_BLIND, 6)]
    rows = []
    for shock_name, shock in SHOCKS.items():
        cfg = shock_config(shock)
        print(f"\n=== shock: {shock_name}  (base mass 10 kg, step 30-60) ===")
        seed0 = {}
        for label, model_path, keep in agents_cfg:
            env = make_eval_env(10.0, cfg, keep)
            agent = None
            if model_path is not None:
                agent = Agent(env)
                agent.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
                agent.eval()
            recs = [run_episode(env, agent, s, 10.0, cfg) for s in EVAL_SEEDS]
            env.close()
            seed0[label] = recs[0]
            for r in recs:
                rows.append(
                    {k: v for k, v in r.items() if not k.startswith("_")} | {"agent": label, "shock": shock_name}
                )
            import numpy as _np

            print(
                f"  {label:<20} success={_np.mean([r['success_rate'] for r in recs]):5.0f}%  "
                f"recovery={_np.nanmean([r['recovery_time_s'] for r in recs]):6.2f}s  "
                f"post_max_err={_np.nanmean([r['post_shock_max_err'] for r in recs]):.2f}m  "
                f"overshoot={_np.mean([r['overshoot'] for r in recs]):.3f}m"
            )

        # Trajectory plot (seed 70000).
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        colors = {"Fixed PID": "#E53935", "Stage 6a (Context)": "#1E88E5", "Stage 6b (Blind)": "#43A047"}
        for label, r in seed0.items():
            t = np.arange(len(r["_pos"])) * r["_dt"]
            axes[0].plot(t, r["_pos"], color=colors[label], label=label)
            axes[1].plot(t, r["_kd"], color=colors[label], label=label)
            if r["shock_step"] > 0:
                for ax in axes:
                    ax.axvline(r["shock_step"] * r["_dt"], color=colors[label], ls=":", lw=0.8, alpha=0.6)
        axes[0].axhline(TARGET_POS, color="k", ls="--", lw=0.8)
        axes[0].set_title(f"Position — {shock_name}")
        axes[0].set_xlabel("Time (s)")
        axes[0].legend(fontsize=7)
        axes[1].set_title("Kd gain")
        axes[1].set_xlabel("Time (s)")
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / f"shock_{shock_name}.png", dpi=130)
        plt.close(fig)

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_DIR / "shock_summary.csv", index=False)
    summary = (
        df.groupby(["shock", "agent"])
        .agg(
            success=("success_rate", "mean"),
            recovery_s=("recovery_time_s", "mean"),
            post_max_err=("post_shock_max_err", "mean"),
            overshoot=("overshoot", "mean"),
        )
        .round(3)
    )
    summary.to_csv(OUTPUT_DIR / "shock_aggregate.csv")
    print("\n=== AGGREGATE ===\n", summary.to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())

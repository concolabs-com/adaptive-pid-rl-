#!/usr/bin/env python3
"""
RQ4 — Probing experiment: does the blind agent implicitly encode the hidden
physics parameters?

Protocol:
  1. Roll out the blind agent (deterministic) for N episodes with hidden
     parameters sampled per episode (mass, friction, and — for stage6
     checkpoints — actuator strength).
  2. At every control step record:
       - the flattened frame-stacked observation (the policy INPUT window),
       - the policy's penultimate-layer activations (what the policy ENCODES),
       - the true hidden parameters.
  3. Fit ridge-regression probes feature -> parameter with GroupKFold
     cross-validation grouped by episode (no within-episode leakage).
  4. Report R² overall and per episode-time bucket ("decodability over time"),
     for both feature sets.

Expected pattern: mass/actuator decodability rises within the first ~10-20
steps (one stack window) and saturates; friction R² ≈ 0 (rolling-contact
irrelevance, finding F8). If R²(representation) ≈ R²(input window), the policy
preserves the identifiable information; if it is much lower, the policy
discards it (pure robust control rather than implicit identification).

Usage:
  python probe_hidden_params.py                          # old stage5b (6-dim, mass+friction)
  python probe_hidden_params.py --model <stage6b.pth> --actuator-axis --out-dir ...
"""

import argparse
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from agents.model import Agent
from stage2_meta_rl_reproduction import make_eval_env, protocol_preset_defaults

TOLERANCE = 0.05
HOLD_STEPS = 25
MAX_STEPS = 1200
TARGET = 5.0
STACK_SIZE = 10

TIME_BUCKETS = [(0, 5), (5, 10), (10, 20), (20, 40), (40, 80), (80, 160), (160, 320), (320, 10_000)]


def build_env():
    p = protocol_preset_defaults("thesis_v4_cliff")
    return make_eval_env(
        stack_size=STACK_SIZE,
        target_pos=TARGET,
        max_eval_steps=MAX_STEPS,
        hold_steps=HOLD_STEPS,
        obs_keep_dims=6,
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


def set_physics(env, mass, friction, actuator, actuator_axis):
    base = env.unwrapped
    base.model.body_mass[1] = float(mass)
    base.model.geom_friction[0, 0] = float(friction)
    if actuator_axis:
        if not hasattr(base, "_probe_nominals"):
            base._probe_nominals = {
                "gainprm": base.model.actuator_gainprm.copy(),
                "frcrange": base.model.jnt_actfrcrange.copy(),
            }
        base.model.actuator_gainprm[:, 0] = base._probe_nominals["gainprm"][:, 0] * float(actuator)
        base.model.jnt_actfrcrange[:] = base._probe_nominals["frcrange"] * float(actuator)


def collect(model_path: str, n_episodes: int, seed0: int, actuator_axis: bool):
    env = build_env()
    agent = Agent(env)
    agent.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
    agent.eval()

    # Penultimate representation = output of the second Tanh in actor_mean
    # (Sequential: [Linear, Tanh, Linear, Tanh, Linear]).
    activations = {}
    agent.actor_mean[3].register_forward_hook(lambda mod, inp, out: activations.__setitem__("h", out.detach()))

    rng = np.random.default_rng(seed0)
    X_obs, X_act, y_mass, y_fric, y_actu, t_idx, ep_idx = [], [], [], [], [], [], []

    for ep in range(n_episodes):
        mass = float(rng.uniform(5.0, 20.0))
        friction = float(rng.uniform(0.1, 2.0))
        actuator = float(rng.uniform(0.6, 1.4)) if actuator_axis else 1.0

        set_physics(env, mass, friction, actuator, actuator_axis)
        obs, _ = env.reset(seed=int(seed0 * 1000 + ep))
        set_physics(env, mass, friction, actuator, actuator_axis)

        for step in range(MAX_STEPS):
            flat = np.asarray(obs, dtype=np.float32).reshape(-1)
            with torch.no_grad():
                a = agent.actor_mean(torch.as_tensor(flat).unsqueeze(0))
                a = torch.clamp(a, -1.0, 1.0)
            X_obs.append(flat)
            X_act.append(activations["h"].squeeze(0).numpy())
            y_mass.append(mass)
            y_fric.append(friction)
            y_actu.append(actuator)
            t_idx.append(step)
            ep_idx.append(ep)

            obs, _, terminated, truncated, _ = env.step(a.squeeze(0).numpy())
            if terminated or truncated:
                break
        if (ep + 1) % 20 == 0:
            print(f"  collected episode {ep + 1}/{n_episodes}")

    env.close()
    return (
        np.asarray(X_obs),
        np.asarray(X_act),
        np.asarray(y_mass),
        np.asarray(y_fric),
        np.asarray(y_actu),
        np.asarray(t_idx),
        np.asarray(ep_idx),
    )


def probe_r2(X, y, groups, n_splits=5):
    """GroupKFold ridge probe; returns out-of-fold predictions and R²."""
    if np.std(y) < 1e-12:
        return np.full_like(y, np.mean(y)), float("nan")
    preds = np.zeros_like(y, dtype=np.float64)
    gkf = GroupKFold(n_splits=n_splits)
    for train, test in gkf.split(X, y, groups):
        scaler = StandardScaler().fit(X[train])
        model = RidgeCV(alphas=np.logspace(-3, 3, 13))
        model.fit(scaler.transform(X[train]), y[train])
        preds[test] = model.predict(scaler.transform(X[test]))
    return preds, float(r2_score(y, preds))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="benchmark_results/stage5b_blind_cliff/seed_7/models/meta_rl_agent.pth")
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument(
        "--actuator-axis",
        action="store_true",
        default=False,
        help="Also randomize/probe actuator strength (stage6 checkpoints)",
    )
    ap.add_argument("--out-dir", default="benchmark_results/probe_stage5b")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Collecting {args.episodes} episodes from {args.model} ...")
    X_obs, X_act, y_mass, y_fric, y_actu, t_idx, ep_idx = collect(
        args.model, args.episodes, args.seed, args.actuator_axis
    )
    print(f"Dataset: {X_obs.shape[0]} steps, obs dim {X_obs.shape[1]}, act dim {X_act.shape[1]}")

    targets = {"mass": y_mass, "friction": y_fric}
    if args.actuator_axis:
        targets["actuator"] = y_actu

    rows = []
    pred_cache = {}
    for feat_name, X in [("input_window", X_obs), ("representation", X_act)]:
        for target_name, y in targets.items():
            preds, r2_all = probe_r2(X, y, ep_idx)
            pred_cache[(feat_name, target_name)] = preds
            rows.append(
                {
                    "features": feat_name,
                    "target": target_name,
                    "bucket": "all",
                    "t_lo": 0,
                    "t_hi": 10_000,
                    "n": len(y),
                    "r2": r2_all,
                }
            )
            print(f"  R²[{feat_name} -> {target_name}] (all steps) = {r2_all:.3f}")
            # Per-time-bucket R² computed on the SAME out-of-fold predictions
            # (probe trained on all phases, evaluated per phase).
            for lo, hi in TIME_BUCKETS:
                m = (t_idx >= lo) & (t_idx < hi)
                if m.sum() < 50 or np.std(y[m]) < 1e-9:
                    continue
                rows.append(
                    {
                        "features": feat_name,
                        "target": target_name,
                        "bucket": f"[{lo},{hi})",
                        "t_lo": lo,
                        "t_hi": hi,
                        "n": int(m.sum()),
                        "r2": float(r2_score(y[m], preds[m])),
                    }
                )

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "probe_r2.csv", index=False)

    # Plot: R² vs time bucket per (features, target)
    fig, ax = plt.subplots(figsize=(9, 5))
    for (feat, target), grp in df[df.bucket != "all"].groupby(["features", "target"]):
        grp = grp.sort_values("t_lo")
        centers = (grp["t_lo"] + np.minimum(grp["t_hi"], 600)) / 2 * 0.02
        ls = "-" if feat == "input_window" else "--"
        ax.plot(centers, grp["r2"], marker="o", ls=ls, label=f"{target} ({feat})")
    ax.set_xlabel("Episode time (s)")
    ax.set_ylabel("Out-of-fold R²")
    ax.set_ylim(-0.1, 1.0)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    ax.set_title("Decodability of hidden physics parameters from the blind policy")
    fig.tight_layout()
    fig.savefig(out_dir / "probe_r2_over_time.png", dpi=150)
    print(f"Saved {out_dir / 'probe_r2.csv'} and probe_r2_over_time.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

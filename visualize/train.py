#!/usr/bin/env python3
"""
Simple PPO training script — adaptive suspension agent.

Trains a blind MLP policy to drive a car to a target position by scheduling
PID gains (Kp, Ki, Kd) at each timestep.

Designed to run on a laptop in ~5-10 minutes (100k timesteps, single env).

Usage:
    python train.py                        # default settings
    python train.py --timesteps 200000     # longer run, better policy
    python train.py --target 3.0           # shorter target distance
    python train.py --run-name my_run      # custom output folder name
"""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal

from environment.agents.model import Agent
from environment.envs.adaptive_suspension import AdaptiveSuspensionEnv
from utils.wrappers import ObservationFeatureSelectWrapper


# ---------------------------------------------------------------------------
# Environment factory
# ---------------------------------------------------------------------------

def make_env(target_pos: float, max_episode_steps: int, hold_steps: int,
             obs_dims: int, stack_size: int) -> gym.Env:
    env = AdaptiveSuspensionEnv(
        target_pos=target_pos,
        hold_steps=hold_steps,
        max_episode_steps=max_episode_steps,
        stop_tolerance=0.05,
        # Gain schedule (thesis_v4_cliff protocol)
        gain_base_kp=1.8, gain_base_ki=0.7, gain_base_kd=0.5,
        gain_delta_kp=1.0, gain_delta_ki=0.6, gain_delta_kd=2.0,
        gain_range_kp=(0.0, 6.0), gain_range_ki=(0.0, 3.0), gain_range_kd=(0.0, 5.0),
        approach_progress_cutoff_m=2.0,
        near_approach_zone_m=2.0,
        brake_zone_vel_sq_coef=10.0,
        near_target_zone_m=0.8,
        near_target_coef=0.0,
        near_approach_coef=0.0,
        decel_bonus_coef=10.0,
        terminal_hold_bonus=50.0,
        terminal_hold_velocity_threshold=0.08,
        safety_speed_governor_enabled=False,
        safety_hard_overshoot_m=-1.0,
        safety_overshoot_penalty=0.0,
    )
    if obs_dims > 0 and obs_dims < int(env.observation_space.shape[0]):
        env = ObservationFeatureSelectWrapper(env, keep_dims=obs_dims)
    env = gym.wrappers.TimeLimit(env, max_episode_steps=max_episode_steps)
    env = gym.wrappers.FrameStackObservation(env, stack_size=stack_size)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    return env


# ---------------------------------------------------------------------------
# PPO training
# ---------------------------------------------------------------------------

def train(args: argparse.Namespace, out_dir: Path) -> Agent:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    env = make_env(
        target_pos=args.target,
        max_episode_steps=args.max_episode_steps,
        hold_steps=args.hold_steps,
        obs_dims=args.obs_dims,
        stack_size=args.stack_size,
    )
    obs, _ = env.reset(seed=args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Obs shape: {env.observation_space.shape}  Action shape: {env.action_space.shape}")

    agent = Agent(env, recurrent=False).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.lr, eps=1e-5)

    # PPO rollout buffers (num_steps steps × 1 env)
    N = args.num_steps
    obs_buf    = torch.zeros((N,) + env.observation_space.shape).to(device)
    act_buf    = torch.zeros((N,) + env.action_space.shape).to(device)
    logp_buf   = torch.zeros(N).to(device)
    rew_buf    = torch.zeros(N).to(device)
    done_buf   = torch.zeros(N).to(device)
    val_buf    = torch.zeros(N).to(device)

    # Logging
    episode_rewards: list[float] = []
    episode_lengths: list[int]   = []
    update_steps:    list[int]   = []

    total_steps = 0
    num_updates = args.timesteps // N
    minibatch_size = N // args.num_minibatches

    print(f"\nTraining for {args.timesteps:,} timesteps  ({num_updates} updates × {N} steps)")
    print(f"Output: {out_dir}\n")

    start_time = time.time()
    next_obs = torch.from_numpy(np.asarray(obs, dtype=np.float32)).to(device)
    next_done = torch.zeros(1).to(device)

    for update in range(1, num_updates + 1):

        # ── Collect rollout ──────────────────────────────────────────────
        for step in range(N):
            obs_buf[step] = next_obs
            done_buf[step] = next_done

            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(
                    next_obs.unsqueeze(0)
                )
                val_buf[step] = value.squeeze()

            act_buf[step]  = action.squeeze(0)
            logp_buf[step] = logprob.squeeze()

            obs_np, reward, terminated, truncated, info = env.step(
                action.squeeze(0).cpu().numpy()
            )
            rew_buf[step] = torch.tensor(float(reward))
            next_done = torch.tensor(float(terminated or truncated)).to(device)
            next_obs  = torch.from_numpy(np.asarray(obs_np, dtype=np.float32)).to(device)

            if terminated or truncated:
                ep_r = info.get("episode", {}).get("r", float("nan"))
                ep_l = info.get("episode", {}).get("l", 0)
                episode_rewards.append(float(ep_r))
                episode_lengths.append(int(ep_l))
                update_steps.append(total_steps + step)

            total_steps += 1

        # ── GAE advantage estimation ─────────────────────────────────────
        with torch.no_grad():
            next_value = agent.get_value(next_obs.unsqueeze(0)).squeeze()
            advantages = torch.zeros_like(rew_buf)
            last_gae = 0.0
            for t in reversed(range(N)):
                if t == N - 1:
                    next_non_terminal = 1.0 - next_done.item()
                    next_val = next_value
                else:
                    next_non_terminal = 1.0 - done_buf[t + 1].item()
                    next_val = val_buf[t + 1]
                delta = rew_buf[t] + args.gamma * next_val * next_non_terminal - val_buf[t]
                last_gae = float(delta) + args.gamma * args.gae_lambda * next_non_terminal * last_gae
                advantages[t] = last_gae
            returns = advantages + val_buf

        # ── PPO update ───────────────────────────────────────────────────
        flat_obs  = obs_buf.reshape((-1,) + env.observation_space.shape)
        flat_act  = act_buf.reshape((-1,) + env.action_space.shape)
        flat_logp = logp_buf.reshape(-1)
        flat_adv  = advantages.reshape(-1)
        flat_ret  = returns.reshape(-1)

        flat_adv = (flat_adv - flat_adv.mean()) / (flat_adv.std() + 1e-8)

        for _ in range(args.update_epochs):
            perm = torch.randperm(N)
            for start in range(0, N, minibatch_size):
                idx = perm[start : start + minibatch_size]
                mb_obs  = flat_obs[idx]
                mb_act  = flat_act[idx]
                mb_logp = flat_logp[idx]
                mb_adv  = flat_adv[idx]
                mb_ret  = flat_ret[idx]

                _, new_logp, entropy, new_val = agent.get_action_and_value(
                    mb_obs, mb_act
                )
                new_val = new_val.squeeze()

                log_ratio = new_logp - mb_logp
                ratio = log_ratio.exp()

                # Policy loss (clipped surrogate)
                pg_loss = torch.max(
                    -mb_adv * ratio,
                    -mb_adv * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef),
                ).mean()

                # Value loss
                v_loss = 0.5 * ((new_val - mb_ret) ** 2).mean()

                loss = pg_loss + args.vf_coef * v_loss - args.ent_coef * entropy.mean()

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

        # ── Progress log ─────────────────────────────────────────────────
        if update % 10 == 0 or update == num_updates:
            elapsed = time.time() - start_time
            sps = total_steps / elapsed
            recent = episode_rewards[-10:] if episode_rewards else [float("nan")]
            print(
                f"  update {update:4d}/{num_updates}  "
                f"steps {total_steps:7,}  "
                f"SPS {sps:5.0f}  "
                f"ep_reward (last 10) {np.mean(recent):7.1f}"
            )

    env.close()
    print(f"\nTraining done in {time.time() - start_time:.1f}s")

    # Save model
    model_path = out_dir / "model.pth"
    torch.save(agent.state_dict(), model_path)
    print(f"Model saved → {model_path}")

    # Save training curve
    import pandas as pd
    df = pd.DataFrame({
        "step":           update_steps,
        "episode_reward": episode_rewards,
        "episode_length": episode_lengths,
    })
    df.to_csv(out_dir / "training_curve.csv", index=False)

    # Plot training curve
    if len(episode_rewards) > 1:
        _plot_training_curve(episode_rewards, update_steps, out_dir)

    return agent


def _plot_training_curve(rewards: list[float], steps: list[int], out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(steps, rewards, alpha=0.3, color="steelblue", linewidth=0.8, label="Episode reward")
    # Rolling mean
    window = max(1, len(rewards) // 20)
    if len(rewards) >= window:
        roll = np.convolve(rewards, np.ones(window) / window, mode="valid")
        ax.plot(steps[window - 1 :], roll, color="steelblue", linewidth=2, label=f"Rolling mean ({window})")
    ax.set_xlabel("Environment steps")
    ax.set_ylabel("Episode reward")
    ax.set_title("Training curve")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "training_curve.png", dpi=120)
    plt.close(fig)
    print(f"Training curve saved → {out_dir / 'training_curve.png'}")


# ---------------------------------------------------------------------------
# Evaluation + plots
# ---------------------------------------------------------------------------

EVAL_SCENARIOS = [
    {"name": "Standard",      "mass": 10.0, "friction": 1.0},
    {"name": "Heavy+Slippery","mass": 20.0, "friction": 0.2},
    {"name": "Light+Grippy",  "mass":  5.0, "friction": 2.0},
]


def evaluate(agent: Agent, args: argparse.Namespace, out_dir: Path) -> None:
    """Run 3 eval scenarios, collect trajectories, save CSV + plots."""
    import pandas as pd

    device = next(agent.parameters()).device
    agent.eval()

    records = []
    trajectories = {}  # scenario_name -> {time, pos, vel}

    for scenario in EVAL_SCENARIOS:
        env = make_env(
            target_pos=args.target,
            max_episode_steps=args.max_episode_steps,
            hold_steps=args.hold_steps,
            obs_dims=args.obs_dims,
            stack_size=args.stack_size,
        )
        # Set fixed physics
        base_env = env.unwrapped
        nominal_mass     = float(base_env.model.body_mass[1])
        nominal_friction = float(base_env.model.geom_friction[0, 0])
        base_env.model.body_mass[1]       = scenario["mass"]
        base_env.model.geom_friction[0, 0] = scenario["friction"]
        base_env.set_disturbance_context(
            mass_scale=scenario["mass"] / max(nominal_mass, 1e-6),
            friction_scale=scenario["friction"] / max(nominal_friction, 1e-6),
        )

        obs, _ = env.reset(seed=args.seed)
        dt = float(base_env.model.opt.timestep) * base_env.frame_skip

        times, positions, velocities = [], [], []
        t = 0.0
        done = False

        with torch.no_grad():
            while not done:
                obs_t = torch.from_numpy(np.asarray(obs, dtype=np.float32)).unsqueeze(0).to(device)
                action = agent.actor_mean(obs_t.reshape(1, -1))
                action = torch.clamp(action, -1.0, 1.0)
                obs, reward, terminated, truncated, info = env.step(action.squeeze(0).cpu().numpy())
                done = terminated or truncated
                times.append(t)
                positions.append(float(info["state"]["pos"]))
                velocities.append(float(info["state"]["vel"]))
                t += dt

        trajectories[scenario["name"]] = {
            "time": times, "pos": positions, "vel": velocities
        }

        final_error = abs(positions[-1] - args.target) if positions else float("nan")
        overshoot   = max(0.0, max(positions) - args.target) if positions else float("nan")
        settled     = final_error < args.tolerance
        records.append({
            "scenario":    scenario["name"],
            "mass_kg":     scenario["mass"],
            "friction":    scenario["friction"],
            "steps":       len(positions),
            "duration_s":  round(t, 3),
            "final_error_m": round(final_error, 4),
            "overshoot_m":   round(overshoot, 4),
            "settled":     settled,
        })

        env.close()
        print(f"  {scenario['name']:20s}  steps={len(positions):4d}  "
              f"final_err={final_error:.4f}m  overshoot={overshoot:.4f}m  "
              f"settled={'YES' if settled else 'NO'}")

    # Save CSV
    df = pd.DataFrame(records)
    df.to_csv(out_dir / "eval_results.csv", index=False)
    print(f"Eval results saved → {out_dir / 'eval_results.csv'}")

    # Plots
    _plot_trajectories(trajectories, args.target, out_dir)


def _plot_trajectories(
    trajectories: dict, target: float, out_dir: Path
) -> None:
    colors = ["steelblue", "tomato", "seagreen"]
    names  = list(trajectories.keys())

    # Displacement-time
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, name in enumerate(names):
        traj = trajectories[name]
        ax.plot(traj["time"], traj["pos"], color=colors[i], linewidth=2, label=name)
    ax.axhline(target, color="black", linestyle="--", linewidth=1.2, label=f"Target ({target}m)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Position (m)")
    ax.set_title("Displacement–Time")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "displacement_time.png", dpi=120)
    plt.close(fig)
    print(f"Displacement-time plot saved → {out_dir / 'displacement_time.png'}")

    # Velocity-time
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, name in enumerate(names):
        traj = trajectories[name]
        ax.plot(traj["time"], traj["vel"], color=colors[i], linewidth=2, label=name)
    ax.axhline(0, color="black", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Velocity (m/s)")
    ax.set_title("Velocity–Time")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "velocity_time.png", dpi=120)
    plt.close(fig)
    print(f"Velocity-time plot saved → {out_dir / 'velocity_time.png'}")


# ---------------------------------------------------------------------------
# Config save
# ---------------------------------------------------------------------------

def save_config(args: argparse.Namespace, out_dir: Path) -> None:
    config = vars(args).copy()
    config["timestamp"] = datetime.now().isoformat()
    with open(out_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train adaptive suspension PPO agent (laptop-friendly)")

    p.add_argument("--run-name", default=None,
                   help="Output folder name under training_runs/. Defaults to timestamp.")
    p.add_argument("--timesteps", type=int, default=100_000,
                   help="Total training timesteps (default: 100000, ~5-10 min on CPU)")
    p.add_argument("--target", type=float, default=5.0,
                   help="Target position in metres (default: 5.0)")
    p.add_argument("--seed", type=int, default=42)

    # Model / env
    p.add_argument("--obs-dims", type=int, default=6,
                   help="Obs dims per step: 6=blind, 8=context-aware (default: 6, faster to train)")
    p.add_argument("--stack-size", type=int, default=10)
    p.add_argument("--max-episode-steps", type=int, default=500)
    p.add_argument("--hold-steps", type=int, default=25)
    p.add_argument("--tolerance", type=float, default=0.05)

    # PPO hyperparams
    p.add_argument("--num-steps", type=int, default=512,
                   help="Rollout length per update (default: 512)")
    p.add_argument("--num-minibatches", type=int, default=4)
    p.add_argument("--update-epochs", type=int, default=4)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--clip-coef", type=float, default=0.2)
    p.add_argument("--ent-coef", type=float, default=0.0)
    p.add_argument("--vf-coef", type=float, default=0.5)
    p.add_argument("--max-grad-norm", type=float, default=0.5)

    return p.parse_args()


def main() -> None:
    args = parse_args()

    run_name = args.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path("training_runs") / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    save_config(args, out_dir)
    print(f"\n{'='*56}")
    print(f"  Run: {run_name}")
    print(f"  Target: {args.target}m   Obs dims: {args.obs_dims}   Stack: {args.stack_size}")
    print(f"  Timesteps: {args.timesteps:,}   Seed: {args.seed}")
    print(f"{'='*56}\n")

    agent = train(args, out_dir)

    print("\nRunning evaluation...")
    evaluate(agent, args, out_dir)

    print(f"\nAll outputs saved to: {out_dir.resolve()}")
    print("\nTo visualize the trained model:")
    print(f"  python run_agent.py --model {out_dir / 'model.pth'} --obs-dims {args.obs_dims}")


if __name__ == "__main__":
    main()

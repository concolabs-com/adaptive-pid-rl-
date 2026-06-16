import argparse
import json
from pathlib import Path

import gymnasium as gym
import matplotlib.pyplot as plt
import mujoco
import numpy as np
import torch

from agents.model import Agent
from envs.adaptive_suspension import AdaptiveSuspensionEnv


class ObservationFeatureSelectWrapper(gym.ObservationWrapper):
    def __init__(self, env: gym.Env, keep_dims: int):
        super().__init__(env)
        source_space = env.observation_space
        if keep_dims <= 0 or keep_dims > int(source_space.shape[0]):
            raise ValueError(f"Invalid keep_dims={keep_dims} for source shape {source_space.shape}")
        self.keep_dims = int(keep_dims)
        self.observation_space = gym.spaces.Box(
            low=source_space.low[: self.keep_dims],
            high=source_space.high[: self.keep_dims],
            shape=(self.keep_dims,),
            dtype=source_space.dtype,
        )

    def observation(self, observation):
        return np.asarray(observation)[: self.keep_dims]


def infer_expected_obs_dim_per_step(state_dict: dict, stack_size: int) -> int:
    if "actor_mean.0.weight" in state_dict:
        input_dim = int(state_dict["actor_mean.0.weight"].shape[1])
    elif "encoder.0.weight" in state_dict:
        input_dim = int(state_dict["encoder.0.weight"].shape[1])
    else:
        raise ValueError("Could not infer input dimension from checkpoint state_dict.")
    if input_dim % stack_size != 0:
        raise ValueError(f"Checkpoint input dim {input_dim} is not divisible by stack_size={stack_size}.")
    return int(input_dim // stack_size)


def load_agent(model_path: Path, stack_size: int) -> tuple[Agent, int]:
    state_dict = torch.load(model_path, map_location="cpu")
    expected_obs_dim_per_step = infer_expected_obs_dim_per_step(state_dict, stack_size)

    temp_env = AdaptiveSuspensionEnv(target_pos=5.0, hold_steps=25, max_episode_steps=5000, stop_tolerance=0.05)
    if expected_obs_dim_per_step < int(temp_env.observation_space.shape[0]):
        temp_env = ObservationFeatureSelectWrapper(temp_env, keep_dims=expected_obs_dim_per_step)
    temp_env = gym.wrappers.TimeLimit(temp_env, max_episode_steps=5000)
    temp_env = gym.wrappers.FrameStackObservation(temp_env, stack_size=stack_size)

    agent = Agent(temp_env)
    agent.load_state_dict(state_dict)
    agent.eval()
    temp_env.close()
    return agent, expected_obs_dim_per_step


def make_env(cfg: dict, target_distance: float, max_steps: int):
    env = AdaptiveSuspensionEnv(
        target_pos=target_distance,
        hold_steps=int(cfg["hold_steps"]),
        max_episode_steps=max_steps,
        stop_tolerance=float(cfg["tolerance"]),
        gain_base_kp=float(cfg["gain_base_kp"]),
        gain_base_ki=float(cfg["gain_base_ki"]),
        gain_base_kd=float(cfg["gain_base_kd"]),
        gain_delta_kp=float(cfg["gain_delta_kp"]),
        gain_delta_ki=float(cfg["gain_delta_ki"]),
        gain_delta_kd=float(cfg["gain_delta_kd"]),
        gain_range_kp=tuple(cfg["gain_range_kp"]),
        gain_range_ki=tuple(cfg["gain_range_ki"]),
        gain_range_kd=tuple(cfg["gain_range_kd"]),
        terminal_hold_bonus=float(cfg.get("terminal_hold_bonus", 0.0)),
        terminal_hold_velocity_threshold=float(cfg.get("terminal_hold_velocity_threshold", 0.05)),
        action_slew_limit=float(cfg.get("action_slew_limit", 1.0)),
        action_rate_penalty_coef=float(cfg.get("action_rate_penalty_coef", 0.0)),
        safety_speed_governor_enabled=bool(cfg.get("safety_speed_governor_enabled", False)),
        safety_brake_margin_m=float(cfg.get("safety_brake_margin_m", 0.3)),
        safety_max_decel_mps2=float(cfg.get("safety_max_decel_mps2", 2.0)),
        safety_brake_k=float(cfg.get("safety_brake_k", 3.5)),
        safety_hard_overshoot_m=float(cfg.get("safety_hard_overshoot_m", -1.0)),
        safety_overshoot_penalty=float(cfg.get("safety_overshoot_penalty", 200.0)),
    )
    return gym.wrappers.TimeLimit(env, max_episode_steps=max_steps)


def run_trace(
    agent: Agent, expected_obs_dim: int, cfg: dict, scenario: dict, target_distance: float, seed: int, max_steps: int
) -> dict:
    env = make_env(cfg, target_distance=target_distance, max_steps=max_steps)
    if expected_obs_dim < int(env.observation_space.shape[0]):
        env = ObservationFeatureSelectWrapper(env, keep_dims=expected_obs_dim)
    env = gym.wrappers.FrameStackObservation(env, stack_size=int(cfg["stack_size"]))

    env.unwrapped.model.body_mass[1] = float(scenario["mass"])
    env.unwrapped.model.geom_friction[0, 0] = float(scenario["friction"])
    mujoco.mj_forward(env.unwrapped.model, env.unwrapped.data)

    obs, _ = env.reset(seed=seed)
    dt = float(env.unwrapped.dt)

    t, pos, vel, err, rew = [], [], [], [], []
    kp, ki, kd = [], [], []

    for step_idx in range(max_steps):
        with torch.no_grad():
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
            action = agent.actor_mean(obs_tensor.reshape(obs_tensor.shape[0], -1))
            action = torch.clamp(action, -1.0, 1.0)

        obs, reward, terminated, truncated, info = env.step(action.squeeze(0).cpu().numpy())

        state = info.get("state", {})
        gains = info.get("gains", {})
        p = float(state.get("pos", np.nan))
        v = float(state.get("vel", np.nan))

        t.append((step_idx + 1) * dt)
        pos.append(p)
        vel.append(v)
        err.append(float(target_distance - p))
        rew.append(float(reward))

        kp.append(float(gains.get("kp", np.nan)))
        ki.append(float(gains.get("ki", np.nan)))
        kd.append(float(gains.get("kd", np.nan)))

        if terminated or truncated:
            break

    env.close()
    return {
        "name": scenario["name"],
        "t": np.asarray(t, dtype=np.float64),
        "pos": np.asarray(pos, dtype=np.float64),
        "vel": np.asarray(vel, dtype=np.float64),
        "err": np.asarray(err, dtype=np.float64),
        "reward": np.asarray(rew, dtype=np.float64),
        "kp": np.asarray(kp, dtype=np.float64),
        "ki": np.asarray(ki, dtype=np.float64),
        "kd": np.asarray(kd, dtype=np.float64),
    }


def plot_main_traces(traces: list[dict], target_distance: float, tolerance: float, out_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex="col")

    for tr in traces:
        label = tr["name"]
        axes[0, 0].plot(tr["t"], tr["pos"], linewidth=2.0, label=label)
        axes[1, 0].plot(tr["t"], tr["vel"], linewidth=1.8, label=label)
        axes[0, 1].plot(tr["t"], tr["err"], linewidth=1.8, label=label)
        axes[1, 1].plot(tr["t"], tr["reward"], linewidth=1.6, label=label)

    axes[0, 0].axhline(target_distance, color="black", linestyle="--", linewidth=1.2, label="Target")
    axes[0, 0].axhline(target_distance + tolerance, color="gray", linestyle=":", linewidth=1.0)
    axes[0, 0].axhline(target_distance - tolerance, color="gray", linestyle=":", linewidth=1.0)
    axes[0, 0].set_title("Displacement vs Time")
    axes[0, 0].set_ylabel("Position (m)")
    axes[0, 0].grid(alpha=0.25)

    axes[1, 0].set_title("Velocity vs Time")
    axes[1, 0].set_xlabel("Time (s)")
    axes[1, 0].set_ylabel("Velocity (m/s)")
    axes[1, 0].grid(alpha=0.25)

    axes[0, 1].axhline(0.0, color="black", linestyle="--", linewidth=1.2)
    axes[0, 1].axhline(tolerance, color="gray", linestyle=":", linewidth=1.0)
    axes[0, 1].axhline(-tolerance, color="gray", linestyle=":", linewidth=1.0)
    axes[0, 1].set_title("Tracking Error vs Time")
    axes[0, 1].set_ylabel("Error (m)")
    axes[0, 1].grid(alpha=0.25)

    axes[1, 1].set_title("Instant Reward vs Time")
    axes[1, 1].set_xlabel("Time (s)")
    axes[1, 1].set_ylabel("Reward")
    axes[1, 1].grid(alpha=0.25)

    axes[0, 0].legend(loc="best")
    axes[1, 0].legend(loc="best")
    axes[0, 1].legend(loc="best")

    fig.suptitle("Stage2 V4 Safety Model Diagnostics", fontsize=14)
    fig.tight_layout(rect=(0.0, 0.02, 1.0, 0.95))
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_gain_diagnostics(traces: list[dict], out_path: Path) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(12.5, 9.0), sharex=True)

    for tr in traces:
        label = tr["name"]
        axes[0].plot(tr["t"], tr["kp"], linewidth=1.8, label=label)
        axes[1].plot(tr["t"], tr["ki"], linewidth=1.8, label=label)
        axes[2].plot(tr["t"], tr["kd"], linewidth=1.8, label=label)

    axes[0].set_title("Adaptive Gain Kp")
    axes[0].set_ylabel("Kp")
    axes[0].grid(alpha=0.25)

    axes[1].set_title("Adaptive Gain Ki")
    axes[1].set_ylabel("Ki")
    axes[1].grid(alpha=0.25)

    axes[2].set_title("Adaptive Gain Kd")
    axes[2].set_ylabel("Kd")
    axes[2].set_xlabel("Time (s)")
    axes[2].grid(alpha=0.25)

    axes[0].legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot displacement, velocity, and diagnostics for a Stage2 model.")
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("benchmark_results/stage2_safety_finetune_v4/seed_7/models/meta_rl_agent.pth"),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("benchmark_results/stage2_safety_finetune_v4/stage2_config.json"),
    )
    parser.add_argument("--seed", type=int, default=70008)
    parser.add_argument("--target-distance", type=float, default=5.0)
    parser.add_argument("--max-steps", type=int, default=5000)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmark_results/stage2_safety_finetune_v4/plots"),
    )
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    agent, expected_obs_dim = load_agent(args.model, int(cfg["stack_size"]))
    scenarios = cfg["scenarios"]

    traces = []
    for scenario in scenarios:
        traces.append(
            run_trace(
                agent,
                expected_obs_dim,
                cfg,
                scenario,
                target_distance=args.target_distance,
                seed=args.seed,
                max_steps=args.max_steps,
            )
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    main_plot = args.output_dir / "v4_displacement_velocity_diagnostics.png"
    gain_plot = args.output_dir / "v4_gain_diagnostics.png"

    plot_main_traces(traces, args.target_distance, float(cfg["tolerance"]), main_plot)
    plot_gain_diagnostics(traces, gain_plot)

    print(f"Saved: {main_plot}")
    print(f"Saved: {gain_plot}")


if __name__ == "__main__":
    main()

import argparse
import json
from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np
import pandas as pd
import torch
from gymnasium import spaces

from agents.model import Agent
from envs.adaptive_suspension import AdaptiveSuspensionEnv

SCENARIOS = [
    {"name": "Standard", "mass": 10.0, "friction": 1.0},
    {"name": "Heavy and Slippery", "mass": 20.0, "friction": 0.2},
    {"name": "Light and Grippy", "mass": 5.0, "friction": 2.0},
]


class ObservationFeatureSelectWrapper(gym.ObservationWrapper):
    """Keep only the first N per-step observation features for checkpoint compatibility."""

    def __init__(self, env: gym.Env, keep_dims: int):
        super().__init__(env)
        source_space = env.observation_space
        if not isinstance(source_space, spaces.Box) or len(source_space.shape) != 1:
            raise ValueError("ObservationFeatureSelectWrapper requires 1D Box observations.")

        if keep_dims <= 0 or keep_dims > int(source_space.shape[0]):
            raise ValueError(f"Invalid keep_dims={keep_dims} for source shape {source_space.shape}")

        self.keep_dims = int(keep_dims)
        self.observation_space = spaces.Box(
            low=source_space.low[: self.keep_dims],
            high=source_space.high[: self.keep_dims],
            shape=(self.keep_dims,),
            dtype=source_space.dtype,
        )

    def observation(self, observation):
        return np.asarray(observation)[: self.keep_dims]


class MidEpisodeFrictionPatchWrapper(gym.Wrapper):
    """Create a friction patch and optional tiny mass shift after a sampled trigger distance."""

    def __init__(
        self,
        env: gym.Env,
        patch_friction_scale: float,
        patch_length: float,
        patch_mass_scale: float,
        trigger_distance_range: tuple[float, float],
    ):
        super().__init__(env)
        self.patch_friction_scale = float(patch_friction_scale)
        self.patch_length = float(patch_length)
        self.patch_mass_scale = float(patch_mass_scale)
        self.trigger_distance_range = parse_range_pair(",".join(str(value) for value in trigger_distance_range))
        self._disturbance_fired = False
        self._start_x = 0.0
        self._trigger_x = 0.0
        self._patch_x_min = 0.0
        self._patch_x_max = 0.0
        self._base_friction = 0.0
        self._car_body_id = mujoco.mj_name2id(self.unwrapped.model, mujoco.mjtObj.mjOBJ_BODY, "car")
        if self._car_body_id < 0:
            self._car_body_id = 1
        self._nominal_mass = float(self.unwrapped.model.body_mass[self._car_body_id])
        self._nominal_inertia = self.unwrapped.model.body_inertia[self._car_body_id].copy()
        self._base_mass = self._nominal_mass
        self._trigger_distance = 0.0

    def reset(self, **kwargs):
        # Restore nominal mass each episode before sampling a new disturbance.
        self.unwrapped.model.body_mass[self._car_body_id] = float(self._nominal_mass)
        self.unwrapped.model.body_inertia[self._car_body_id, :] = self._nominal_inertia
        obs, info = self.env.reset(**kwargs)
        rng = getattr(self.unwrapped, "np_random", np.random)
        self._start_x = float(self.unwrapped.data.qpos[0])
        self._base_friction = float(self.unwrapped.model.geom_friction[0, 0])
        self._base_mass = float(self.unwrapped.model.body_mass[self._car_body_id])
        self._trigger_distance = float(rng.uniform(*self.trigger_distance_range))
        self._trigger_x = self._start_x + self._trigger_distance
        # Patch will be positioned at 0.2m after the trigger point
        patch_start = self._trigger_x + 0.2
        self._patch_x_min = patch_start
        self._patch_x_max = patch_start + self.patch_length
        self._disturbance_fired = False
        return obs, info

    def step(self, action):
        # Activate patch when vehicle crosses trigger point
        if not self._disturbance_fired and float(self.unwrapped.data.qpos[0]) >= float(self._trigger_x):
            self._disturbance_fired = True
            disturbed_mass = float(max(self._base_mass * self.patch_mass_scale, 1e-6))
            self.unwrapped.model.body_mass[self._car_body_id] = disturbed_mass
            self.unwrapped.model.body_inertia[self._car_body_id, :] = self._nominal_inertia * self.patch_mass_scale

        obs, reward, terminated, truncated, info = self.env.step(action)

        # Apply patch friction when car is within patch bounds
        if self._disturbance_fired:
            car_x = float(self.unwrapped.data.qpos[0])
            if self._patch_x_min <= car_x <= self._patch_x_max:
                self.unwrapped.model.geom_friction[0, 0] = float(self._base_friction * self.patch_friction_scale)
            else:
                self.unwrapped.model.geom_friction[0, 0] = float(self._base_friction)
            mujoco.mj_forward(self.unwrapped.model, self.unwrapped.data)

        info = dict(info)
        info["friction_shift"] = {
            "fired": bool(self._disturbance_fired),
            "start_x": float(self._start_x),
            "trigger_distance": float(self._trigger_distance),
            "trigger_x": float(self._trigger_x),
            "patch_x_min": float(self._patch_x_min),
            "patch_x_max": float(self._patch_x_max),
            "patch_friction_scale": float(self.patch_friction_scale),
            "patch_mass_scale": float(self.patch_mass_scale),
            "disturbed_mass": float(self.unwrapped.model.body_mass[self._car_body_id]),
        }
        return obs, reward, terminated, truncated, info


def compute_settling_time(errors: np.ndarray, dt: float, tolerance_m: float, timeout_s: float) -> tuple[float, int]:
    within = np.abs(errors) <= tolerance_m
    if not np.any(within):
        return float(timeout_s), 0

    suffix_all_within = np.flip(np.cumprod(np.flip(within).astype(int))).astype(bool)
    settled_indices = np.flatnonzero(suffix_all_within)
    if len(settled_indices) == 0:
        return float(timeout_s), 0

    return float(settled_indices[0] * dt), 1


def compute_hold_success(errors: np.ndarray, tolerance_m: float, hold_steps: int) -> int:
    if len(errors) < hold_steps:
        return 0
    return int(np.all(np.abs(errors[-hold_steps:]) <= tolerance_m))


def parse_range_pair(text: str) -> tuple[float, float]:
    parts = [part.strip() for part in text.split(",") if part.strip()]
    if len(parts) != 2:
        raise ValueError(f"Expected exactly two comma-separated values, got: {text}")
    first = float(parts[0])
    second = float(parts[1])
    return (first, second) if first <= second else (second, first)


def infer_expected_obs_dim_per_step(state_dict: dict, stack_size: int) -> int:
    if "encoder.0.weight" in state_dict:
        input_dim = int(state_dict["encoder.0.weight"].shape[1])
    elif "actor_mean.0.weight" in state_dict:
        input_dim = int(state_dict["actor_mean.0.weight"].shape[1])
    else:
        raise ValueError("Could not infer input dimension from checkpoint state_dict.")

    if stack_size <= 0:
        raise ValueError("stack_size must be > 0")
    if input_dim % stack_size != 0:
        raise ValueError(f"Checkpoint input dim {input_dim} is not divisible by stack_size={stack_size}.")
    return int(input_dim // stack_size)


def make_env_for_scenario(args: argparse.Namespace, scenario: dict, expected_obs_dim_per_step: int) -> gym.Env:
    env = AdaptiveSuspensionEnv(
        target_pos=args.target_pos,
        hold_steps=args.hold_steps,
        max_episode_steps=args.max_eval_steps,
        stop_tolerance=args.tolerance,
        gain_base_kp=args.gain_base_kp,
        gain_base_ki=args.gain_base_ki,
        gain_base_kd=args.gain_base_kd,
        gain_delta_kp=args.gain_delta_kp,
        gain_delta_ki=args.gain_delta_ki,
        gain_delta_kd=args.gain_delta_kd,
        gain_range_kp=args.gain_range_kp,
        gain_range_ki=args.gain_range_ki,
        gain_range_kd=args.gain_range_kd,
        terminal_hold_bonus=args.terminal_hold_bonus,
        terminal_hold_velocity_threshold=args.terminal_hold_velocity_threshold,
        action_slew_limit=args.action_slew_limit,
        action_rate_penalty_coef=args.action_rate_penalty_coef,
    )

    env.model.body_mass[1] = float(scenario["mass"])
    env.model.geom_friction[0, 0] = float(scenario["friction"])
    mujoco.mj_forward(env.model, env.data)

    base_obs_dim = int(env.observation_space.shape[0])
    if expected_obs_dim_per_step < base_obs_dim:
        env = ObservationFeatureSelectWrapper(env, keep_dims=expected_obs_dim_per_step)
    elif expected_obs_dim_per_step > base_obs_dim:
        raise ValueError(
            f"Checkpoint expects {expected_obs_dim_per_step} obs features per step, but env provides {base_obs_dim}."
        )

    env = MidEpisodeFrictionPatchWrapper(
        env,
        patch_friction_scale=args.patch_friction_scale,
        patch_length=args.patch_length,
        patch_mass_scale=args.patch_mass_scale,
        trigger_distance_range=args.disturbance_distance_range,
    )
    env = gym.wrappers.TimeLimit(env, max_episode_steps=args.max_eval_steps)
    env = gym.wrappers.FrameStackObservation(env, stack_size=args.stack_size)
    return env


def load_agent(env: gym.Env, state_dict: dict, recurrent_policy: bool, recurrent_hidden_size: int) -> Agent:
    agent = Agent(env, recurrent=recurrent_policy, recurrent_hidden_size=recurrent_hidden_size)
    agent.load_state_dict(state_dict)
    agent.eval()
    return agent


def select_action(agent: Agent, obs: np.ndarray, args: argparse.Namespace, hidden_state, done_tensor):
    with torch.no_grad():
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        if args.recurrent_policy:
            action, _, _, _, hidden_state = agent.get_action_and_value(
                obs_tensor,
                hidden_state=hidden_state,
                done=done_tensor,
                deterministic=True,
            )
        else:
            action = agent.actor_mean(obs_tensor.reshape(obs_tensor.shape[0], -1))
        action = torch.clamp(action, -1.0, 1.0)
    return action, hidden_state


def summarize_episode(
    positions: list[float],
    rewards: list[float],
    base_env: AdaptiveSuspensionEnv,
    args: argparse.Namespace,
    scenario: dict,
    episode_seed: int,
    shift_info: dict | None,
) -> dict:
    positions_arr = np.asarray(positions, dtype=np.float64)
    errors_arr = base_env.target_pos - positions_arr
    dt = float(base_env.dt)
    timeout_s = args.max_eval_steps * dt
    settling_time_s, settled = compute_settling_time(errors_arr, dt, args.tolerance, timeout_s)
    overshoot = max(float(np.max(positions_arr) - base_env.target_pos), 0.0) if len(positions_arr) else float("nan")
    iae = float(np.sum(np.abs(errors_arr)) * dt) if len(positions_arr) else float("nan")
    final_abs_error = float(abs(errors_arr[-1])) if len(errors_arr) else float("nan")
    success = compute_hold_success(errors_arr, args.tolerance, args.hold_steps) if len(errors_arr) else 0

    return {
        "scenario": scenario["name"],
        "episode": int(episode_seed % 1_000_000),
        "seed": int(episode_seed),
        "base_mass": float(scenario["mass"]),
        "base_friction": float(scenario["friction"]),
        "disturbance_distance_min": float(args.disturbance_distance_range[0]),
        "disturbance_distance_max": float(args.disturbance_distance_range[1]),
        "disturbance_distance_sampled": (
            float(shift_info.get("trigger_distance", float("nan"))) if shift_info else float("nan")
        ),
        "disturbance_trigger_x": float(shift_info.get("trigger_x", float("nan"))) if shift_info else float("nan"),
        "patch_x_min": float(shift_info.get("patch_x_min", float("nan"))) if shift_info else float("nan"),
        "patch_x_max": float(shift_info.get("patch_x_max", float("nan"))) if shift_info else float("nan"),
        "friction_scale": float(args.friction_scale),
        "patch_friction_scale": float(args.patch_friction_scale),
        "patch_mass_scale": float(args.patch_mass_scale),
        "patch_length": float(args.patch_length),
        "final_friction": float(scenario["friction"] * args.patch_friction_scale),
        "final_mass": float(scenario["mass"] * args.patch_mass_scale),
        "steps": int(len(positions_arr)),
        "mean_reward": float(np.mean(rewards)) if rewards else float("nan"),
        "final_position": float(positions_arr[-1]) if len(positions_arr) else float("nan"),
        "final_error": float(errors_arr[-1]) if len(errors_arr) else float("nan"),
        "settling_time_s": settling_time_s,
        "settled": settled,
        "overshoot": overshoot,
        "iae": iae,
        "final_abs_error": final_abs_error,
        "hold_steps": int(args.hold_steps),
        "success": success,
    }


def run_episode(env: gym.Env, agent: Agent, scenario: dict, episode_seed: int, args: argparse.Namespace) -> dict:
    obs, _ = env.reset(seed=episode_seed)
    base_env: AdaptiveSuspensionEnv = env.unwrapped
    positions = []
    rewards = []
    shift_info: dict | None = None
    device = next(agent.parameters()).device
    hidden_state = agent.get_initial_state(1, device=device) if args.recurrent_policy else None
    done_tensor = torch.zeros(1, device=device) if args.recurrent_policy else None

    for _ in range(args.max_eval_steps):
        action, hidden_state = select_action(agent, obs, args, hidden_state, done_tensor)

        obs, reward, terminated, truncated, info = env.step(action.squeeze(0).cpu().numpy())
        positions.append(float(info["state"]["pos"]))
        rewards.append(float(reward))
        if shift_info is None and isinstance(info, dict):
            shift_info = info.get("friction_shift")

        if terminated or truncated:
            break

    return summarize_episode(positions, rewards, base_env, args, scenario, episode_seed, shift_info)


def aggregate_results(df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        df.groupby(["scenario"], as_index=False)
        .agg(
            episodes=("episode", "count"),
            settling_time_mean=("settling_time_s", "mean"),
            settling_time_std=("settling_time_s", "std"),
            settled_rate=("settled", "mean"),
            overshoot_mean=("overshoot", "mean"),
            overshoot_std=("overshoot", "std"),
            iae_mean=("iae", "mean"),
            iae_std=("iae", "std"),
            final_abs_error_mean=("final_abs_error", "mean"),
            final_abs_error_std=("final_abs_error", "std"),
            success_rate=("success", "mean"),
            mean_reward=("mean_reward", "mean"),
        )
        .sort_values(["scenario"])
    )
    summary["settled_rate"] = 100.0 * summary["settled_rate"]
    summary["success_rate"] = 100.0 * summary["success_rate"]
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Mid-episode friction probe for the Stage 2 adaptive suspension model."
    )
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("benchmark_results") / "mid_episode_friction_probe")
    parser.add_argument("--seeds", type=str, default="7")
    parser.add_argument("--stack-size", type=int, default=10)
    parser.add_argument("--target-pos", type=float, default=3.0)
    parser.add_argument("--hold-steps", type=int, default=10)
    parser.add_argument("--tolerance", type=float, default=0.05)
    parser.add_argument("--max-eval-steps", type=int, default=500)
    parser.add_argument("--eval-episodes", type=int, default=30)
    parser.add_argument("--disturbance-step", type=int, default=150)
    parser.add_argument("--disturbance-distance-range", type=str, default="1.5,2.5")
    parser.add_argument("--friction-scale", type=float, default=0.98)
    parser.add_argument("--patch-friction-scale", type=float, default=0.25, help="Friction scaling on the patch (0-1)")
    parser.add_argument("--patch-length", type=float, default=0.8, help="Length of the friction patch in meters")
    parser.add_argument("--patch-mass-scale", type=float, default=1.0, help="Mass scaling applied after patch trigger")
    parser.add_argument("--mass-scale-fixed", type=float, default=1.0)
    parser.add_argument("--friction-scale-fixed", type=float, default=1.0)
    parser.add_argument("--recurrent-policy", action="store_true", default=False)
    parser.add_argument("--recurrent-hidden-size", type=int, default=128)
    parser.add_argument("--gain-base-kp", type=float, default=1.8)
    parser.add_argument("--gain-base-ki", type=float, default=0.7)
    parser.add_argument("--gain-base-kd", type=float, default=0.0)
    parser.add_argument("--gain-delta-kp", type=float, default=1.0)
    parser.add_argument("--gain-delta-ki", type=float, default=0.6)
    parser.add_argument("--gain-delta-kd", type=float, default=2.0)
    parser.add_argument("--gain-range-kp", type=str, default="0.0,6.0")
    parser.add_argument("--gain-range-ki", type=str, default="0.0,3.0")
    parser.add_argument("--gain-range-kd", type=str, default="0.0,5.0")
    parser.add_argument("--terminal-hold-bonus", type=float, default=0.0)
    parser.add_argument("--terminal-hold-velocity-threshold", type=float, default=0.05)
    parser.add_argument("--action-slew-limit", type=float, default=1.0)
    parser.add_argument("--action-rate-penalty-coef", type=float, default=0.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.seeds = [int(seed.strip()) for seed in args.seeds.split(",") if seed.strip()]
    args.gain_range_kp = parse_range_pair(args.gain_range_kp)
    args.gain_range_ki = parse_range_pair(args.gain_range_ki)
    args.gain_range_kd = parse_range_pair(args.gain_range_kd)
    args.disturbance_distance_range = parse_range_pair(args.disturbance_distance_range)

    config_path = args.output_dir / "mid_episode_friction_probe_config.json"
    config_payload = {
        "model_path": str(args.model_path),
        "seeds": args.seeds,
        "stack_size": args.stack_size,
        "target_pos": args.target_pos,
        "hold_steps": args.hold_steps,
        "tolerance": args.tolerance,
        "max_eval_steps": args.max_eval_steps,
        "eval_episodes": args.eval_episodes,
        "disturbance_trigger_mode": "position_patch",
        "disturbance_distance_range": list(args.disturbance_distance_range),
        "friction_scale": args.friction_scale,
        "patch_friction_scale": args.patch_friction_scale,
        "patch_mass_scale": args.patch_mass_scale,
        "patch_length": args.patch_length,
        "mass_scale_fixed": args.mass_scale_fixed,
        "friction_scale_fixed": args.friction_scale_fixed,
        "recurrent_policy": args.recurrent_policy,
        "recurrent_hidden_size": args.recurrent_hidden_size,
        "gain_base_kp": args.gain_base_kp,
        "gain_base_ki": args.gain_base_ki,
        "gain_base_kd": args.gain_base_kd,
        "gain_delta_kp": args.gain_delta_kp,
        "gain_delta_ki": args.gain_delta_ki,
        "gain_delta_kd": args.gain_delta_kd,
        "gain_range_kp": list(args.gain_range_kp),
        "gain_range_ki": list(args.gain_range_ki),
        "gain_range_kd": list(args.gain_range_kd),
        "terminal_hold_bonus": args.terminal_hold_bonus,
        "terminal_hold_velocity_threshold": args.terminal_hold_velocity_threshold,
        "action_slew_limit": args.action_slew_limit,
        "action_rate_penalty_coef": args.action_rate_penalty_coef,
    }
    config_path.write_text(json.dumps(config_payload, indent=2), encoding="utf-8")

    state_dict = torch.load(args.model_path, map_location="cpu")
    expected_obs_dim_per_step = infer_expected_obs_dim_per_step(state_dict, args.stack_size)
    print(f"Checkpoint expects {expected_obs_dim_per_step} features/step with stack_size={args.stack_size}.")

    records: list[dict] = []
    for seed in args.seeds:
        for scenario in SCENARIOS:
            env = make_env_for_scenario(args, scenario, expected_obs_dim_per_step)
            try:
                agent = load_agent(env, state_dict, args.recurrent_policy, args.recurrent_hidden_size)
                for episode_idx in range(args.eval_episodes):
                    episode_seed = seed * 10_000 + episode_idx
                    row = run_episode(env, agent, scenario, episode_seed, args)
                    row["seed_group"] = int(seed)
                    row["episode_index"] = int(episode_idx)
                    records.append(row)
            finally:
                env.close()

    raw_df = pd.DataFrame(records)
    raw_path = args.output_dir / "mid_episode_friction_probe_raw.csv"
    summary_path = args.output_dir / "mid_episode_friction_probe_summary.csv"
    raw_df.to_csv(raw_path, index=False)
    summary_df = aggregate_results(raw_df)
    summary_df.to_csv(summary_path, index=False)

    print(f"Saved config to {config_path}")
    print(f"Saved raw results to {raw_path}")
    print(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    main()

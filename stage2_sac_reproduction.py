import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor

from agents.domain_randomization import DomainRandomizationWrapper
from envs.adaptive_suspension import AdaptiveSuspensionEnv

DEFAULT_STACK_SIZE = 10
DEFAULT_NUM_ENVS = 1
DEFAULT_TOTAL_TIMESTEPS = 80_000
DEFAULT_EVAL_EPISODES = 10
DEFAULT_MAX_EVAL_STEPS = 500
DEFAULT_TRAIN_EPISODE_STEPS = 2000
DEFAULT_TARGET_POS = 1.0
DEFAULT_TARGET_MIN = 1.0
DEFAULT_TARGET_MAX = 10.0
DEFAULT_HOLD_STEPS = 25
DEFAULT_TOLERANCE = 0.1
DEFAULT_SEEDS = [7, 21, 42, 84, 123]


@dataclass(frozen=True)
class EvalScenario:
    name: str
    mass: float
    friction: float


@dataclass(frozen=True)
class Stage2SACConfig:
    output_dir: Path
    seeds: list[int]
    total_timesteps: int
    num_envs: int
    init_model_path: Path | None
    train_episode_steps: int
    target_pos: float
    target_min: float
    target_max: float
    randomize_target: bool
    hold_steps: int
    train_stop_tolerance: float
    stack_size: int
    eval_episodes: int
    max_eval_steps: int
    tolerance: float
    gain_base_kp: float
    gain_base_ki: float
    gain_base_kd: float
    gain_delta_kp: float
    gain_delta_ki: float
    gain_delta_kd: float
    gain_range_kp: tuple[float, float]
    gain_range_ki: tuple[float, float]
    gain_range_kd: tuple[float, float]
    mid_episode_disturbance_enabled: bool
    disturbance_mode: str
    disturbance_step_range: tuple[int, int]
    disturbance_time_range_s: tuple[float, float]
    disturbance_mass_scale_range: tuple[float, float]
    disturbance_friction_scale_range: tuple[float, float]
    position_patch_enabled: bool
    patch_x_range: tuple[float, float]
    patch_friction_scale: float
    terminal_hold_bonus: float
    terminal_hold_velocity_threshold: float
    action_slew_limit: float
    action_rate_penalty_coef: float
    disturbance_in_eval: bool
    learning_rate: float
    buffer_size: int
    learning_starts: int
    batch_size: int
    tau: float
    gamma: float
    train_freq: int
    gradient_steps: int


SCENARIOS = [
    EvalScenario("Standard", mass=10.0, friction=1.0),
    EvalScenario("Heavy and Slippery", mass=20.0, friction=0.2),
    EvalScenario("Light and Grippy", mass=5.0, friction=2.0),
]


class TargetRandomizationWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, target_min: float, target_max: float):
        super().__init__(env)
        self.target_min = float(target_min)
        self.target_max = float(target_max)

    def reset(self, **kwargs):
        rng = getattr(self.unwrapped, "np_random", np.random)
        sampled_target = float(rng.uniform(self.target_min, self.target_max))
        if hasattr(self.unwrapped, "set_target_pos"):
            self.unwrapped.set_target_pos(sampled_target)
        else:
            self.unwrapped.target_pos = sampled_target
        return self.env.reset(**kwargs)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def parse_range_pair(text: str, cast):
    parts = [p.strip() for p in text.split(",") if p.strip()]
    if len(parts) != 2:
        raise ValueError(f"Range must have exactly two comma-separated values: {text}")
    a = cast(parts[0])
    b = cast(parts[1])
    lo = a if a <= b else b
    hi = b if a <= b else a
    return lo, hi


def parse_seed_list(seed_text: str) -> list[int]:
    seeds = [int(part.strip()) for part in seed_text.split(",") if part.strip()]
    if not seeds:
        raise ValueError("At least one seed must be provided.")
    return seeds


def build_domain_randomization_config(config: Stage2SACConfig, for_eval: bool = False) -> dict:
    return {
        "mass_range": (5.0, 20.0),
        "friction_range": (0.1, 2.0),
        "initial_randomization_enabled": not for_eval,
        "mid_episode_disturbance_enabled": config.mid_episode_disturbance_enabled,
        "disturbance_mode": config.disturbance_mode,
        "disturbance_step_range": config.disturbance_step_range,
        "disturbance_time_range_s": config.disturbance_time_range_s,
        "disturbance_mass_scale_range": config.disturbance_mass_scale_range,
        "disturbance_friction_scale_range": config.disturbance_friction_scale_range,
        "position_patch_enabled": config.position_patch_enabled,
        "patch_x_range": config.patch_x_range,
        "patch_friction_scale": config.patch_friction_scale,
    }


def make_train_env(config: Stage2SACConfig, seed: int, idx: int):
    def thunk():
        env = AdaptiveSuspensionEnv(
            target_pos=config.target_pos,
            hold_steps=config.hold_steps,
            max_episode_steps=config.train_episode_steps,
            stop_tolerance=config.train_stop_tolerance,
            gain_base_kp=config.gain_base_kp,
            gain_base_ki=config.gain_base_ki,
            gain_base_kd=config.gain_base_kd,
            gain_delta_kp=config.gain_delta_kp,
            gain_delta_ki=config.gain_delta_ki,
            gain_delta_kd=config.gain_delta_kd,
            gain_range_kp=config.gain_range_kp,
            gain_range_ki=config.gain_range_ki,
            gain_range_kd=config.gain_range_kd,
            terminal_hold_bonus=config.terminal_hold_bonus,
            terminal_hold_velocity_threshold=config.terminal_hold_velocity_threshold,
            action_slew_limit=config.action_slew_limit,
            action_rate_penalty_coef=config.action_rate_penalty_coef,
        )
        if config.randomize_target:
            env = TargetRandomizationWrapper(env, config.target_min, config.target_max)
        env = DomainRandomizationWrapper(
            env, randomization_config=build_domain_randomization_config(config, for_eval=False)
        )
        env = gym.wrappers.TimeLimit(env, max_episode_steps=config.train_episode_steps)
        env = gym.wrappers.FrameStackObservation(env, stack_size=config.stack_size)
        env = gym.wrappers.FlattenObservation(env)
        env.action_space.seed(seed + idx)
        env.observation_space.seed(seed + idx)
        return env

    return thunk


def make_eval_env(config: Stage2SACConfig):
    env = AdaptiveSuspensionEnv(
        target_pos=config.target_pos,
        hold_steps=config.hold_steps,
        max_episode_steps=config.max_eval_steps,
        stop_tolerance=config.tolerance,
        gain_base_kp=config.gain_base_kp,
        gain_base_ki=config.gain_base_ki,
        gain_base_kd=config.gain_base_kd,
        gain_delta_kp=config.gain_delta_kp,
        gain_delta_ki=config.gain_delta_ki,
        gain_delta_kd=config.gain_delta_kd,
        gain_range_kp=config.gain_range_kp,
        gain_range_ki=config.gain_range_ki,
        gain_range_kd=config.gain_range_kd,
        terminal_hold_bonus=config.terminal_hold_bonus,
        terminal_hold_velocity_threshold=config.terminal_hold_velocity_threshold,
        action_slew_limit=config.action_slew_limit,
        action_rate_penalty_coef=config.action_rate_penalty_coef,
    )
    if config.disturbance_in_eval:
        env = DomainRandomizationWrapper(
            env, randomization_config=build_domain_randomization_config(config, for_eval=True)
        )
    env = gym.wrappers.TimeLimit(env, max_episode_steps=config.max_eval_steps)
    env = gym.wrappers.FrameStackObservation(env, stack_size=config.stack_size)
    env = gym.wrappers.FlattenObservation(env)
    return env


class EpisodeStatsCallback(BaseCallback):
    def __init__(self):
        super().__init__()
        self.rows: list[dict] = []

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            ep = info.get("episode")
            if ep is not None:
                self.rows.append(
                    {
                        "timesteps": int(self.num_timesteps),
                        "episode_return": float(ep.get("r", np.nan)),
                        "episode_length": float(ep.get("l", np.nan)),
                    }
                )
        return True


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


def run_eval_episode(eval_env, model: SAC, scenario: EvalScenario, seed: int, config: Stage2SACConfig) -> dict:
    eval_env.unwrapped.model.body_mass[1] = scenario.mass
    eval_env.unwrapped.model.geom_friction[0, 0] = scenario.friction
    obs, _ = eval_env.reset(seed=seed)

    positions = []
    rewards = []
    dt = float(eval_env.unwrapped.dt)
    timeout_s = config.max_eval_steps * dt

    for _ in range(config.max_eval_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = eval_env.step(action)
        positions.append(float(info["state"]["pos"]))
        rewards.append(float(reward))
        if terminated or truncated:
            break

    positions_arr = np.asarray(positions, dtype=np.float64)
    errors_arr = eval_env.unwrapped.target_pos - positions_arr
    settling_time_s, settled = compute_settling_time(errors_arr, dt, config.tolerance, timeout_s)
    overshoot = (
        max(float(np.max(positions_arr) - eval_env.unwrapped.target_pos), 0.0) if len(positions_arr) else float("nan")
    )
    iae = float(np.sum(np.abs(errors_arr)) * dt) if len(positions_arr) else float("nan")
    final_abs_error = float(abs(errors_arr[-1])) if len(positions_arr) else float("nan")
    success = compute_hold_success(errors_arr, config.tolerance, config.hold_steps) if len(positions_arr) else 0

    return {
        "seed": seed,
        "scenario": scenario.name,
        "episode": int(seed % 1_000_000),
        "mass": scenario.mass,
        "friction": scenario.friction,
        "steps": int(len(positions_arr)),
        "mean_reward": float(np.mean(rewards)) if rewards else float("nan"),
        "final_position": float(positions_arr[-1]) if len(positions_arr) else float("nan"),
        "final_error": float(errors_arr[-1]) if len(errors_arr) else float("nan"),
        "settling_time_s": settling_time_s,
        "settled": settled,
        "overshoot": overshoot,
        "iae": iae,
        "final_abs_error": final_abs_error,
        "hold_steps": int(config.hold_steps),
        "success": success,
    }


def evaluate_model(seed: int, model: SAC, config: Stage2SACConfig) -> pd.DataFrame:
    eval_env = make_eval_env(config)
    records = []
    for scenario in SCENARIOS:
        for episode in range(config.eval_episodes):
            episode_seed = seed * 10_000 + episode
            row = run_eval_episode(eval_env, model, scenario, episode_seed, config)
            row["episode"] = episode
            records.append(row)
    eval_env.close()
    return pd.DataFrame(records)


def aggregate_eval_results(df: pd.DataFrame) -> pd.DataFrame:
    return (
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
        .assign(success_rate=lambda x: x["success_rate"] * 100.0)
    )


def aggregate_seed_summary(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["seed", "scenario"], as_index=False)
        .agg(
            episodes=("episode", "count"),
            settling_time_mean=("settling_time_s", "mean"),
            settled_rate=("settled", "mean"),
            overshoot_mean=("overshoot", "mean"),
            iae_mean=("iae", "mean"),
            final_abs_error_mean=("final_abs_error", "mean"),
            success_rate=("success", "mean"),
            mean_reward=("mean_reward", "mean"),
        )
        .assign(success_rate=lambda x: x["success_rate"] * 100.0)
    )


def plot_training_curves(training_df: pd.DataFrame, output_path: Path) -> None:
    if training_df.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    for seed, seed_df in training_df.groupby("seed"):
        ax.plot(seed_df["timesteps"], seed_df["episode_return"], alpha=0.3, label=f"Seed {seed}")
    mean_curve = training_df.groupby("timesteps", as_index=False)["episode_return"].mean()
    ax.plot(mean_curve["timesteps"], mean_curve["episode_return"], color="black", linewidth=2, label="Mean")
    ax.set_xlabel("Timesteps")
    ax.set_ylabel("Episode Return")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)


def plot_eval_summary(summary_df: pd.DataFrame, output_path: Path) -> None:
    scenarios = list(summary_df["scenario"].drop_duplicates())
    fig, axes = plt.subplots(3, len(scenarios), figsize=(6 * len(scenarios), 11), squeeze=False)
    metric_specs = [
        ("iae_mean", "IAE"),
        ("settling_time_mean", "Settling Time (s)"),
        ("success_rate", "Success Rate (%)"),
    ]

    for col, scenario in enumerate(scenarios):
        subset = summary_df[summary_df["scenario"] == scenario].iloc[0]
        for row, (metric_key, ylabel) in enumerate(metric_specs):
            axes[row][col].bar(["SAC"], [float(subset[metric_key])], color="#1E7A59")
            axes[row][col].set_title(f"{scenario} - {ylabel}")
            axes[row][col].set_ylabel(ylabel)
            axes[row][col].set_ylim(bottom=0)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)


def apply_protocol_preset(args: argparse.Namespace) -> None:
    if args.protocol_preset == "thesis_v1":
        args.randomize_target = True
        args.target_pos = 5.0
        args.target_min = 3.0
        args.target_max = 7.0
        args.train_episode_steps = 1200
        args.max_eval_steps = 1200
        args.hold_steps = 25
        args.tolerance = 0.05
        if args.train_stop_tolerance is None:
            args.train_stop_tolerance = 0.05
        args.mid_episode_disturbance_enabled = True
        args.disturbance_mode = "step"
        args.disturbance_step_range = "120,220"
        args.disturbance_mass_scale_range = "0.9,1.3"
        args.disturbance_friction_scale_range = "0.5,1.4"
        args.position_patch_enabled = True
        args.patch_x_range = "1.5,2.4"
        args.patch_friction_scale = 0.35


def build_config(args: argparse.Namespace) -> Stage2SACConfig:
    if args.train_stop_tolerance is None:
        args.train_stop_tolerance = 0.05

    disturbance_mode = args.disturbance_mode.strip().lower()
    if disturbance_mode not in {"step", "time"}:
        raise ValueError("disturbance-mode must be one of: step, time")

    return Stage2SACConfig(
        output_dir=Path(args.output_dir),
        seeds=parse_seed_list(args.seeds),
        total_timesteps=args.total_timesteps,
        num_envs=args.num_envs,
        init_model_path=Path(args.init_model_path) if args.init_model_path else None,
        train_episode_steps=args.train_episode_steps,
        target_pos=args.target_pos,
        target_min=args.target_min,
        target_max=args.target_max,
        randomize_target=args.randomize_target,
        hold_steps=args.hold_steps,
        train_stop_tolerance=args.train_stop_tolerance,
        stack_size=args.stack_size,
        eval_episodes=args.eval_episodes,
        max_eval_steps=args.max_eval_steps,
        tolerance=args.tolerance,
        gain_base_kp=args.gain_base_kp,
        gain_base_ki=args.gain_base_ki,
        gain_base_kd=args.gain_base_kd,
        gain_delta_kp=args.gain_delta_kp,
        gain_delta_ki=args.gain_delta_ki,
        gain_delta_kd=args.gain_delta_kd,
        gain_range_kp=parse_range_pair(args.gain_range_kp, float),
        gain_range_ki=parse_range_pair(args.gain_range_ki, float),
        gain_range_kd=parse_range_pair(args.gain_range_kd, float),
        mid_episode_disturbance_enabled=args.mid_episode_disturbance_enabled,
        disturbance_mode=disturbance_mode,
        disturbance_step_range=parse_range_pair(args.disturbance_step_range, int),
        disturbance_time_range_s=parse_range_pair(args.disturbance_time_range_s, float),
        disturbance_mass_scale_range=parse_range_pair(args.disturbance_mass_scale_range, float),
        disturbance_friction_scale_range=parse_range_pair(args.disturbance_friction_scale_range, float),
        position_patch_enabled=args.position_patch_enabled,
        patch_x_range=parse_range_pair(args.patch_x_range, float),
        patch_friction_scale=args.patch_friction_scale,
        terminal_hold_bonus=args.terminal_hold_bonus,
        terminal_hold_velocity_threshold=args.terminal_hold_velocity_threshold,
        action_slew_limit=args.action_slew_limit,
        action_rate_penalty_coef=args.action_rate_penalty_coef,
        disturbance_in_eval=args.disturbance_in_eval,
        learning_rate=args.learning_rate,
        buffer_size=args.buffer_size,
        learning_starts=args.learning_starts,
        batch_size=args.batch_size,
        tau=args.tau,
        gamma=args.gamma,
        train_freq=args.train_freq,
        gradient_steps=args.gradient_steps,
    )


def save_json_config(config: Stage2SACConfig) -> None:
    serializable = asdict(config)
    serializable["output_dir"] = str(config.output_dir)
    serializable["init_model_path"] = str(config.init_model_path) if config.init_model_path is not None else None
    serializable["scenarios"] = [asdict(s) for s in SCENARIOS]
    with open(config.output_dir / "stage2_config.json", "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2)


def train_one_seed(seed: int, config: Stage2SACConfig) -> tuple[SAC, pd.DataFrame, Path]:
    set_seed(seed)
    seed_dir = config.output_dir / f"seed_{seed}"
    model_dir = seed_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    vec_env = DummyVecEnv([make_train_env(config, seed, idx) for idx in range(config.num_envs)])
    vec_env = VecMonitor(vec_env)

    callback = EpisodeStatsCallback()
    if config.init_model_path is not None:
        model = SAC.load(str(config.init_model_path), env=vec_env, device="cpu")
    else:
        model = SAC(
            policy="MlpPolicy",
            env=vec_env,
            learning_rate=config.learning_rate,
            buffer_size=config.buffer_size,
            learning_starts=config.learning_starts,
            batch_size=config.batch_size,
            tau=config.tau,
            gamma=config.gamma,
            train_freq=config.train_freq,
            gradient_steps=config.gradient_steps,
            verbose=0,
            seed=seed,
        )
    model.learn(total_timesteps=config.total_timesteps, callback=callback)

    model_path = model_dir / "meta_rl_agent_sac"
    model.save(model_path)
    vec_env.close()

    training_df = pd.DataFrame(callback.rows)
    if training_df.empty:
        training_df = pd.DataFrame(
            [{"timesteps": config.total_timesteps, "episode_return": np.nan, "episode_length": np.nan}]
        )
    training_df["seed"] = seed
    training_df.to_csv(seed_dir / "training_curve.csv", index=False)
    return model, training_df, model_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 2 SAC reproduction with multi-seed training and evaluation.")
    parser.add_argument("--protocol-preset", choices=["custom", "thesis_v1"], default="custom")
    parser.add_argument("--output-dir", default="benchmark_results/stage2_sac_reproduction")
    parser.add_argument("--seeds", default=",".join(str(seed) for seed in DEFAULT_SEEDS))
    parser.add_argument("--total-timesteps", type=int, default=DEFAULT_TOTAL_TIMESTEPS)
    parser.add_argument("--num-envs", type=int, default=DEFAULT_NUM_ENVS)
    parser.add_argument("--train-episode-steps", type=int, default=DEFAULT_TRAIN_EPISODE_STEPS)
    parser.add_argument("--target-pos", type=float, default=DEFAULT_TARGET_POS)
    parser.add_argument("--target-min", type=float, default=DEFAULT_TARGET_MIN)
    parser.add_argument("--target-max", type=float, default=DEFAULT_TARGET_MAX)
    parser.add_argument("--randomize-target", action="store_true", default=False)
    parser.add_argument("--hold-steps", type=int, default=DEFAULT_HOLD_STEPS)
    parser.add_argument("--train-stop-tolerance", type=float, default=None)
    parser.add_argument("--stack-size", type=int, default=DEFAULT_STACK_SIZE)
    parser.add_argument("--eval-episodes", type=int, default=DEFAULT_EVAL_EPISODES)
    parser.add_argument("--max-eval-steps", type=int, default=DEFAULT_MAX_EVAL_STEPS)
    parser.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE)

    parser.add_argument("--gain-base-kp", type=float, default=1.8)
    parser.add_argument("--gain-base-ki", type=float, default=0.7)
    parser.add_argument("--gain-base-kd", type=float, default=0.0)
    parser.add_argument("--gain-delta-kp", type=float, default=1.0)
    parser.add_argument("--gain-delta-ki", type=float, default=0.6)
    parser.add_argument("--gain-delta-kd", type=float, default=0.25)
    parser.add_argument("--gain-range-kp", type=str, default="0.0,6.0")
    parser.add_argument("--gain-range-ki", type=str, default="0.0,3.0")
    parser.add_argument("--gain-range-kd", type=str, default="0.0,5.0")

    parser.add_argument("--mid-episode-disturbance-enabled", action="store_true", default=False)
    parser.add_argument("--disturbance-mode", type=str, default="step", choices=["step", "time"])
    parser.add_argument("--disturbance-step-range", type=str, default="100,300")
    parser.add_argument("--disturbance-time-range-s", type=str, default="1.0,3.0")
    parser.add_argument("--disturbance-mass-scale-range", type=str, default="0.8,1.4")
    parser.add_argument("--disturbance-friction-scale-range", type=str, default="0.4,1.6")
    parser.add_argument("--position-patch-enabled", action="store_true", default=False)
    parser.add_argument("--patch-x-range", type=str, default="1.5,2.0")
    parser.add_argument("--patch-friction-scale", type=float, default=0.35)
    parser.add_argument("--disturbance-in-eval", action="store_true", default=False)

    parser.add_argument("--terminal-hold-bonus", type=float, default=0.0)
    parser.add_argument("--terminal-hold-velocity-threshold", type=float, default=0.05)
    parser.add_argument("--action-slew-limit", type=float, default=1.0)
    parser.add_argument("--action-rate-penalty-coef", type=float, default=0.0)

    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--buffer-size", type=int, default=200000)
    parser.add_argument("--learning-starts", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--train-freq", type=int, default=1)
    parser.add_argument("--gradient-steps", type=int, default=1)
    parser.add_argument("--init-model-path", type=str, default=None)
    args = parser.parse_args()

    apply_protocol_preset(args)
    config = build_config(args)
    config.output_dir.mkdir(parents=True, exist_ok=True)

    all_training_rows = []
    all_eval_rows = []
    seed_results = []

    for seed in config.seeds:
        model, training_df, model_path = train_one_seed(seed, config)
        all_training_rows.append(training_df)

        eval_df = evaluate_model(seed, model, config)
        seed_dir = config.output_dir / f"seed_{seed}"
        eval_df.to_csv(seed_dir / "eval_raw.csv", index=False)
        seed_summary = aggregate_seed_summary(eval_df)
        seed_summary.to_csv(seed_dir / "eval_seed_summary.csv", index=False)

        all_eval_rows.append(eval_df)
        seed_results.append(
            {
                "seed": seed,
                "model_path": str(model_path),
                "training_updates": int(training_df["timesteps"].max()) if not training_df.empty else 0,
                "eval_rows": int(len(eval_df)),
                "mean_eval_reward": float(eval_df["mean_reward"].mean()),
            }
        )

    training_all_df = pd.concat(all_training_rows, ignore_index=True)
    eval_all_df = pd.concat(all_eval_rows, ignore_index=True)
    eval_summary_df = aggregate_eval_results(eval_all_df)
    seed_summary_df = aggregate_seed_summary(eval_all_df)
    seed_results_df = pd.DataFrame(seed_results)

    training_all_df.to_csv(config.output_dir / "stage2_training_curves.csv", index=False)
    eval_all_df.to_csv(config.output_dir / "stage2_eval_raw.csv", index=False)
    eval_summary_df.to_csv(config.output_dir / "stage2_eval_summary.csv", index=False)
    seed_summary_df.to_csv(config.output_dir / "stage2_eval_seed_summary.csv", index=False)
    seed_results_df.to_csv(config.output_dir / "stage2_seed_results.csv", index=False)

    plot_training_curves(training_all_df, config.output_dir / "stage2_learning_curve.png")
    plot_eval_summary(eval_summary_df, config.output_dir / "stage2_eval_summary.png")
    save_json_config(config)

    print("Stage 2 SAC reproduction completed.")
    print(f"Saved training curves to {config.output_dir / 'stage2_training_curves.csv'}")
    print(f"Saved evaluation raw data to {config.output_dir / 'stage2_eval_raw.csv'}")
    print(f"Saved evaluation summary to {config.output_dir / 'stage2_eval_summary.csv'}")
    print(f"Saved learning curve plot to {config.output_dir / 'stage2_learning_curve.png'}")
    print(f"Saved evaluation plot to {config.output_dir / 'stage2_eval_summary.png'}")


if __name__ == "__main__":
    main()

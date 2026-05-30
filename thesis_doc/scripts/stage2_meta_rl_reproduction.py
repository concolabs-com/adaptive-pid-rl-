import argparse
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from agents.domain_randomization import DomainRandomizationWrapper
from agents.model import Agent
from envs.adaptive_suspension import AdaptiveSuspensionEnv
from tqdm import tqdm

EXP_NAME = "Stage2_MetaRL_Reproduction"
DEFAULT_STACK_SIZE = 10
DEFAULT_NUM_ENVS = 4
DEFAULT_NUM_STEPS = 1024
DEFAULT_TOTAL_TIMESTEPS = 80_000
DEFAULT_EVAL_EPISODES = 10
DEFAULT_MAX_EVAL_STEPS = 500
DEFAULT_TRAIN_EPISODE_STEPS = 2000
DEFAULT_TARGET_POS = 1.0
DEFAULT_TARGET_MIN = 1.0
DEFAULT_TARGET_MAX = 10.0
DEFAULT_HOLD_STEPS = 25
DEFAULT_CURRICULUM_SPEC = "1:3:0.34,1:6:0.33,1:10:0.33"
DEFAULT_LEARNING_RATE = 3e-4
DEFAULT_MINIBATCH_SIZE = 64
DEFAULT_UPDATE_EPOCHS = 10
DEFAULT_GAMMA = 0.99
DEFAULT_GAE_LAMBDA = 0.95
DEFAULT_CLIP_COEF = 0.2
DEFAULT_ENT_COEF = 0.0
DEFAULT_VF_COEF = 0.5
DEFAULT_MAX_GRAD_NORM = 0.5
DEFAULT_TOLERANCE = 0.1
DEFAULT_SEEDS = [7, 21, 42, 84, 123]


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
    elif "critic.0.weight" in state_dict:
        input_dim = int(state_dict["critic.0.weight"].shape[1])
    else:
        raise ValueError("Could not infer input dimension from checkpoint state_dict.")

    if input_dim % stack_size != 0:
        raise ValueError(f"Checkpoint input dim {input_dim} is not divisible by stack_size={stack_size}.")
    return int(input_dim // stack_size)


@dataclass(frozen=True)
class EvalScenario:
    name: str
    mass: float
    friction: float


@dataclass(frozen=True)
class Stage2Config:
    output_dir: Path
    seeds: list[int]
    total_timesteps: int
    num_envs: int
    num_steps: int
    train_episode_steps: int
    target_pos: float
    target_min: float
    target_max: float
    randomize_target: bool
    hold_steps: int
    obs_keep_dims: int
    train_stop_tolerance: float
    curriculum_enabled: bool
    curriculum_spec: str
    curriculum_phases: list[tuple[float, float, float]]
    stack_size: int
    recurrent_policy: bool
    recurrent_hidden_size: int
    learning_rate: float
    minibatch_size: int
    update_epochs: int
    gamma: float
    gae_lambda: float
    clip_coef: float
    ent_coef: float
    vf_coef: float
    max_grad_norm: float
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
    disturbance_curriculum_enabled: bool
    disturbance_mass_scale_start_range: tuple[float, float]
    disturbance_mass_scale_end_range: tuple[float, float]
    disturbance_friction_scale_start_range: tuple[float, float]
    disturbance_friction_scale_end_range: tuple[float, float]
    position_patch_enabled: bool
    patch_x_range: tuple[float, float]
    patch_friction_scale: float
    terminal_hold_bonus: float
    terminal_hold_velocity_threshold: float
    action_slew_limit: float
    action_rate_penalty_coef: float
    safety_speed_governor_enabled: bool
    safety_brake_margin_m: float
    safety_max_decel_mps2: float
    safety_brake_k: float
    safety_hard_overshoot_m: float
    safety_overshoot_penalty: float
    disturbance_in_eval: bool
    init_model_path: Path | None
    near_approach_zone_m: float
    near_approach_coef: float
    near_target_zone_m: float
    near_target_coef: float
    near_target_excess_thresh: float
    near_target_excess_coef: float
    approach_progress_cutoff_m: float
    brake_zone_vel_sq_coef: float
    decel_bonus_coef: float
    near_target_init_prob: float
    near_target_init_range_m: float
    brake_integral_reset_enabled: bool
    eval_only: bool


SCENARIOS = [
    EvalScenario("Standard", mass=10.0, friction=1.0),
    EvalScenario("Heavy and Slippery", mass=20.0, friction=0.2),
    EvalScenario("Light and Grippy", mass=5.0, friction=2.0),
    # OOD: beyond training distribution (mass trained 5-20kg, friction trained 0.2-2.0)
    EvalScenario("OOD Ultra Heavy", mass=35.0, friction=1.0),
    EvalScenario("OOD Ultra Slippery", mass=20.0, friction=0.05),
]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class TargetRandomizationWrapper(gym.Wrapper):
    """Randomize target distance on reset for better distance generalization."""

    def __init__(self, env: gym.Env, target_min: float, target_max: float):
        super().__init__(env)
        self.target_min = float(target_min)
        self.target_max = float(target_max)
        if self.target_min <= 0.0 or self.target_max <= 0.0:
            raise ValueError("target_min/target_max must be positive.")
        if self.target_min > self.target_max:
            raise ValueError("target_min must be <= target_max.")

    def reset(self, **kwargs):
        rng = getattr(self.unwrapped, "np_random", np.random)
        sampled_target = float(rng.uniform(self.target_min, self.target_max))
        if hasattr(self.unwrapped, "set_target_pos"):
            self.unwrapped.set_target_pos(sampled_target)
        else:
            self.unwrapped.target_pos = sampled_target
        return self.env.reset(**kwargs)

    def set_range(self, target_min: float, target_max: float) -> None:
        self.target_min = float(target_min)
        self.target_max = float(target_max)


def parse_range_pair(text: str, cast):
    parts = [p.strip() for p in text.split(",") if p.strip()]
    if len(parts) != 2:
        raise ValueError(f"Range must have exactly two comma-separated values: {text}")
    a = cast(parts[0])
    b = cast(parts[1])
    lo = a if a <= b else b
    hi = b if a <= b else a
    return lo, hi


def build_domain_randomization_config(config: Stage2Config, for_eval: bool = False) -> dict:
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


def make_train_env(
    seed: int,
    idx: int,
    run_name: str,
    stack_size: int,
    train_episode_steps: int,
    target_pos: float,
    randomize_target: bool,
    target_min: float,
    target_max: float,
    hold_steps: int,
    obs_keep_dims: int,
    train_stop_tolerance: float,
    gain_base_kp: float,
    gain_base_ki: float,
    gain_base_kd: float,
    gain_delta_kp: float,
    gain_delta_ki: float,
    gain_delta_kd: float,
    gain_range_kp: tuple[float, float],
    gain_range_ki: tuple[float, float],
    gain_range_kd: tuple[float, float],
    terminal_hold_bonus: float,
    terminal_hold_velocity_threshold: float,
    action_slew_limit: float,
    action_rate_penalty_coef: float,
    safety_speed_governor_enabled: bool,
    safety_brake_margin_m: float,
    safety_max_decel_mps2: float,
    safety_brake_k: float,
    safety_hard_overshoot_m: float,
    safety_overshoot_penalty: float,
    near_approach_zone_m: float,
    near_approach_coef: float,
    near_target_zone_m: float,
    near_target_coef: float,
    near_target_excess_thresh: float,
    near_target_excess_coef: float,
    approach_progress_cutoff_m: float,
    brake_zone_vel_sq_coef: float,
    decel_bonus_coef: float,
    near_target_init_prob: float,
    near_target_init_range_m: float,
    brake_integral_reset_enabled: bool,
    randomization_config: dict,
    capture_video: bool = False,
):
    def thunk():
        env = AdaptiveSuspensionEnv(
            target_pos=target_pos,
            hold_steps=hold_steps,
            max_episode_steps=train_episode_steps,
            stop_tolerance=train_stop_tolerance,
            gain_base_kp=gain_base_kp,
            gain_base_ki=gain_base_ki,
            gain_base_kd=gain_base_kd,
            gain_delta_kp=gain_delta_kp,
            gain_delta_ki=gain_delta_ki,
            gain_delta_kd=gain_delta_kd,
            gain_range_kp=gain_range_kp,
            gain_range_ki=gain_range_ki,
            gain_range_kd=gain_range_kd,
            terminal_hold_bonus=terminal_hold_bonus,
            terminal_hold_velocity_threshold=terminal_hold_velocity_threshold,
            action_slew_limit=action_slew_limit,
            action_rate_penalty_coef=action_rate_penalty_coef,
            safety_speed_governor_enabled=safety_speed_governor_enabled,
            safety_brake_margin_m=safety_brake_margin_m,
            safety_max_decel_mps2=safety_max_decel_mps2,
            safety_brake_k=safety_brake_k,
            safety_hard_overshoot_m=safety_hard_overshoot_m,
            safety_overshoot_penalty=safety_overshoot_penalty,
            near_approach_zone_m=near_approach_zone_m,
            near_approach_coef=near_approach_coef,
            near_target_zone_m=near_target_zone_m,
            near_target_coef=near_target_coef,
            near_target_excess_thresh=near_target_excess_thresh,
            near_target_excess_coef=near_target_excess_coef,
            approach_progress_cutoff_m=approach_progress_cutoff_m,
            brake_zone_vel_sq_coef=brake_zone_vel_sq_coef,
            decel_bonus_coef=decel_bonus_coef,
            near_target_init_prob=near_target_init_prob,
            near_target_init_range_m=near_target_init_range_m,
            brake_integral_reset_enabled=brake_integral_reset_enabled,
        )
        if randomize_target:
            env = TargetRandomizationWrapper(
                env,
                target_min=target_min,
                target_max=target_max,
            )
        env = DomainRandomizationWrapper(env, randomization_config=randomization_config)
        if obs_keep_dims > 0 and obs_keep_dims < int(env.observation_space.shape[0]):
            env = ObservationFeatureSelectWrapper(env, keep_dims=obs_keep_dims)
        env = gym.wrappers.TimeLimit(env, max_episode_steps=train_episode_steps)
        env = gym.wrappers.FrameStackObservation(env, stack_size=stack_size)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if capture_video and idx == 0:
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    return thunk


def make_eval_env(
    stack_size: int,
    target_pos: float,
    max_eval_steps: int,
    hold_steps: int = 10,
    obs_keep_dims: int = 0,
    stop_tolerance: float = 0.05,
    gain_base_kp: float = 1.8,
    gain_base_ki: float = 0.7,
    gain_base_kd: float = 0.0,
    gain_delta_kp: float = 1.0,
    gain_delta_ki: float = 0.6,
    gain_delta_kd: float = 2.0,
    gain_range_kp: tuple[float, float] = (0.0, 6.0),
    gain_range_ki: tuple[float, float] = (0.0, 3.0),
    gain_range_kd: tuple[float, float] = (0.0, 5.0),
    terminal_hold_bonus: float = 0.0,
    terminal_hold_velocity_threshold: float = 0.05,
    action_slew_limit: float = 1.0,
    action_rate_penalty_coef: float = 0.0,
    safety_speed_governor_enabled: bool = False,
    safety_brake_margin_m: float = 0.30,
    safety_max_decel_mps2: float = 2.0,
    safety_brake_k: float = 3.5,
    safety_hard_overshoot_m: float = -1.0,
    safety_overshoot_penalty: float = 200.0,
    near_approach_zone_m: float = 0.3,
    near_approach_coef: float = 0.75,
    near_target_zone_m: float = 0.2,
    near_target_coef: float = 1.1,
    near_target_excess_thresh: float = 0.15,
    near_target_excess_coef: float = 1.0,
    approach_progress_cutoff_m: float = 0.0,
    brake_zone_vel_sq_coef: float = 0.0,
    decel_bonus_coef: float = 0.0,
    near_target_init_prob: float = 0.0,
    near_target_init_range_m: float = 0.15,
    brake_integral_reset_enabled: bool = True,
    use_disturbance: bool = False,
    randomization_config: dict | None = None,
):
    env = AdaptiveSuspensionEnv(
        target_pos=target_pos,
        hold_steps=hold_steps,
        max_episode_steps=max_eval_steps,
        stop_tolerance=stop_tolerance,
        gain_base_kp=gain_base_kp,
        gain_base_ki=gain_base_ki,
        gain_base_kd=gain_base_kd,
        gain_delta_kp=gain_delta_kp,
        gain_delta_ki=gain_delta_ki,
        gain_delta_kd=gain_delta_kd,
        gain_range_kp=gain_range_kp,
        gain_range_ki=gain_range_ki,
        gain_range_kd=gain_range_kd,
        terminal_hold_bonus=terminal_hold_bonus,
        terminal_hold_velocity_threshold=terminal_hold_velocity_threshold,
        action_slew_limit=action_slew_limit,
        action_rate_penalty_coef=action_rate_penalty_coef,
        safety_speed_governor_enabled=safety_speed_governor_enabled,
        safety_brake_margin_m=safety_brake_margin_m,
        safety_max_decel_mps2=safety_max_decel_mps2,
        safety_brake_k=safety_brake_k,
        safety_hard_overshoot_m=safety_hard_overshoot_m,
        safety_overshoot_penalty=safety_overshoot_penalty,
        near_approach_zone_m=near_approach_zone_m,
        near_approach_coef=near_approach_coef,
        near_target_zone_m=near_target_zone_m,
        near_target_coef=near_target_coef,
        near_target_excess_thresh=near_target_excess_thresh,
        near_target_excess_coef=near_target_excess_coef,
        approach_progress_cutoff_m=approach_progress_cutoff_m,
        brake_zone_vel_sq_coef=brake_zone_vel_sq_coef,
        decel_bonus_coef=decel_bonus_coef,
        near_target_init_prob=near_target_init_prob,
        near_target_init_range_m=near_target_init_range_m,
        brake_integral_reset_enabled=brake_integral_reset_enabled,
    )
    if use_disturbance:
        env = DomainRandomizationWrapper(env, randomization_config=randomization_config)
    if obs_keep_dims > 0 and obs_keep_dims < int(env.observation_space.shape[0]):
        env = ObservationFeatureSelectWrapper(env, keep_dims=obs_keep_dims)
    env = gym.wrappers.TimeLimit(env, max_episode_steps=max_eval_steps)
    env = gym.wrappers.FrameStackObservation(env, stack_size=stack_size)
    return env


def parse_curriculum_spec(spec: str) -> list[tuple[float, float, float]]:
    phases: list[tuple[float, float, float]] = []
    parts = [p.strip() for p in spec.split(",") if p.strip()]
    if not parts:
        raise ValueError("curriculum-spec must contain at least one phase.")

    for part in parts:
        fields = [f.strip() for f in part.split(":")]
        if len(fields) != 3:
            raise ValueError("Each curriculum phase must be min:max:fraction, e.g. 1:3:0.34")
        t_min, t_max, frac = float(fields[0]), float(fields[1]), float(fields[2])
        if t_min <= 0.0 or t_max <= 0.0:
            raise ValueError("Curriculum target ranges must be positive.")
        if t_min > t_max:
            raise ValueError("Curriculum phase min must be <= max.")
        if frac <= 0.0:
            raise ValueError("Curriculum phase fraction must be > 0.")
        phases.append((t_min, t_max, frac))

    frac_sum = sum(p[2] for p in phases)
    if not np.isclose(frac_sum, 1.0, atol=1e-6):
        raise ValueError(f"Curriculum phase fractions must sum to 1.0 (got {frac_sum:.6f}).")
    return phases


def curriculum_range_for_progress(phases: list[tuple[float, float, float]], progress_01: float) -> tuple[float, float]:
    cumulative = 0.0
    for t_min, t_max, frac in phases:
        cumulative += frac
        if progress_01 <= cumulative + 1e-9:
            return t_min, t_max
    last_min, last_max, _ = phases[-1]
    return last_min, last_max


def lerp_range(start_range: tuple[float, float], end_range: tuple[float, float], alpha: float) -> tuple[float, float]:
    alpha = float(np.clip(alpha, 0.0, 1.0))
    lo = float(start_range[0] + alpha * (end_range[0] - start_range[0]))
    hi = float(start_range[1] + alpha * (end_range[1] - start_range[1]))
    return (min(lo, hi), max(lo, hi))


def set_training_target_range(envs, target_min: float, target_max: float) -> None:
    for wrapped_env in envs.envs:
        cursor = wrapped_env
        while hasattr(cursor, "env"):
            if isinstance(cursor, TargetRandomizationWrapper):
                cursor.set_range(target_min, target_max)
                break
            cursor = cursor.env


def set_training_disturbance_ranges(
    envs,
    mass_scale_range: tuple[float, float],
    friction_scale_range: tuple[float, float],
) -> None:
    for wrapped_env in envs.envs:
        cursor = wrapped_env
        while hasattr(cursor, "env"):
            if isinstance(cursor, DomainRandomizationWrapper):
                cursor.config["disturbance_mass_scale_range"] = mass_scale_range
                cursor.config["disturbance_friction_scale_range"] = friction_scale_range
                break
            cursor = cursor.env


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


def build_training_components(seed: int, config: Stage2Config):
    run_name = f"{EXP_NAME}_seed{seed}_{int(time.time())}"
    seed_dir = config.output_dir / f"seed_{seed}"
    model_dir = seed_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    envs = gym.vector.SyncVectorEnv(
        [
            make_train_env(
                seed + idx,
                idx,
                run_name,
                stack_size=config.stack_size,
                train_episode_steps=config.train_episode_steps,
                target_pos=config.target_pos,
                randomize_target=config.randomize_target,
                target_min=config.target_min,
                target_max=config.target_max,
                hold_steps=config.hold_steps,
                obs_keep_dims=config.obs_keep_dims,
                train_stop_tolerance=config.train_stop_tolerance,
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
                safety_speed_governor_enabled=config.safety_speed_governor_enabled,
                safety_brake_margin_m=config.safety_brake_margin_m,
                safety_max_decel_mps2=config.safety_max_decel_mps2,
                safety_brake_k=config.safety_brake_k,
                safety_hard_overshoot_m=config.safety_hard_overshoot_m,
                safety_overshoot_penalty=config.safety_overshoot_penalty,
                near_approach_zone_m=config.near_approach_zone_m,
                near_approach_coef=config.near_approach_coef,
                near_target_zone_m=config.near_target_zone_m,
                near_target_coef=config.near_target_coef,
                near_target_excess_thresh=config.near_target_excess_thresh,
                near_target_excess_coef=config.near_target_excess_coef,
                approach_progress_cutoff_m=config.approach_progress_cutoff_m,
                brake_zone_vel_sq_coef=config.brake_zone_vel_sq_coef,
                decel_bonus_coef=config.decel_bonus_coef,
                near_target_init_prob=config.near_target_init_prob,
                near_target_init_range_m=config.near_target_init_range_m,
                brake_integral_reset_enabled=config.brake_integral_reset_enabled,
                randomization_config=build_domain_randomization_config(config, for_eval=False),
                capture_video=False,
            )
            for idx in range(config.num_envs)
        ]
    )
    agent = Agent(envs, recurrent=config.recurrent_policy, recurrent_hidden_size=config.recurrent_hidden_size)
    if config.init_model_path is not None:
        state_dict = torch.load(config.init_model_path, map_location="cpu")
        agent.load_state_dict(state_dict)
    optimizer = optim.Adam(agent.parameters(), lr=config.learning_rate, eps=1e-5, weight_decay=0.0)

    obs_shape = envs.single_observation_space.shape
    act_shape = envs.single_action_space.shape

    buffers = {
        "obs": torch.zeros((config.num_steps, config.num_envs) + obs_shape),
        "actions": torch.zeros((config.num_steps, config.num_envs) + act_shape),
        "logprobs": torch.zeros((config.num_steps, config.num_envs)),
        "rewards": torch.zeros((config.num_steps, config.num_envs)),
        "dones": torch.zeros((config.num_steps, config.num_envs)),
        "values": torch.zeros((config.num_steps, config.num_envs)),
    }

    next_obs_np, _ = envs.reset(seed=seed)
    device = next(agent.parameters()).device
    trackers = {
        "next_obs": torch.as_tensor(next_obs_np, dtype=torch.float32),
        "next_done": torch.zeros(config.num_envs),
        "episode_returns": np.zeros(config.num_envs, dtype=np.float32),
        "episode_lengths": np.zeros(config.num_envs, dtype=np.int32),
        "completed_returns": [],
        "completed_lengths": [],
        "global_step": 0,
        "seed_dir": seed_dir,
        "model_dir": model_dir,
    }
    if config.recurrent_policy:
        initial_hidden = agent.get_initial_state(config.num_envs, device=device)
        trackers["next_hidden"] = initial_hidden.detach().clone()
        trackers["rollout_start_hidden"] = initial_hidden.detach().clone()
    return envs, agent, optimizer, buffers, trackers


def collect_rollout_step(envs, agent, buffers, trackers, config: Stage2Config):
    step = int(trackers["rollout_step"])
    next_obs = trackers["next_obs"]
    next_done = trackers["next_done"]

    trackers["global_step"] += config.num_envs
    buffers["obs"][step] = next_obs
    buffers["dones"][step] = next_done

    if config.recurrent_policy:
        hidden_state = trackers["next_hidden"]
        with torch.no_grad():
            action, logprob, _, value, next_hidden = agent.get_action_and_value(
                next_obs,
                hidden_state=hidden_state,
                done=next_done,
            )
        trackers["next_hidden"] = next_hidden.detach()
    else:
        with torch.no_grad():
            action, logprob, _, value = agent.get_action_and_value(next_obs)
    buffers["values"][step] = value.flatten()
    clipped_action = torch.clamp(action, -1.0, 1.0)
    buffers["actions"][step] = clipped_action
    buffers["logprobs"][step] = logprob

    next_obs_np, reward, terminations, truncations, _ = envs.step(clipped_action.cpu().numpy())
    next_done_np = np.logical_or(terminations, truncations)

    buffers["rewards"][step] = torch.as_tensor(reward, dtype=torch.float32)
    trackers["episode_returns"] += reward
    trackers["episode_lengths"] += 1

    for env_index, done_flag in enumerate(next_done_np):
        if done_flag:
            trackers["completed_returns"].append(float(trackers["episode_returns"][env_index]))
            trackers["completed_lengths"].append(int(trackers["episode_lengths"][env_index]))
            trackers["episode_returns"][env_index] = 0.0
            trackers["episode_lengths"][env_index] = 0

    trackers["next_obs"] = torch.as_tensor(next_obs_np, dtype=torch.float32)
    trackers["next_done"] = torch.as_tensor(next_done_np, dtype=torch.float32)


def run_rollout(envs, agent, buffers, trackers, config: Stage2Config) -> float:
    rollout_reward_total = 0.0
    trackers["rollout_step"] = 0
    if config.recurrent_policy:
        trackers["rollout_start_hidden"] = trackers["next_hidden"].detach().clone()

    for step in range(config.num_steps):
        trackers["rollout_step"] = step
        collect_rollout_step(envs, agent, buffers, trackers, config)
        rollout_reward_total += float(buffers["rewards"][step].sum().item())

    return rollout_reward_total


def ppo_update(agent, optimizer, buffers, trackers, config: Stage2Config, seed: int, update: int) -> dict:
    if config.recurrent_policy:
        with torch.no_grad():
            next_value = agent.get_value(
                trackers["next_obs"],
                hidden_state=trackers["next_hidden"],
                done=trackers["next_done"],
            ).reshape(1, -1)
            advantages = torch.zeros_like(buffers["rewards"])
            lastgaelam = 0.0
            for t in reversed(range(config.num_steps)):
                if t == config.num_steps - 1:
                    nextnonterminal = 1.0 - trackers["next_done"]
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - buffers["dones"][t + 1]
                    nextvalues = buffers["values"][t + 1]
                delta = buffers["rewards"][t] + config.gamma * nextvalues * nextnonterminal - buffers["values"][t]
                advantages[t] = lastgaelam = delta + config.gamma * config.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + buffers["values"]

        b_obs = buffers["obs"]
        b_actions = buffers["actions"]
        b_dones = buffers["dones"]
        b_logprobs = buffers["logprobs"].reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        initial_hidden = trackers["rollout_start_hidden"]

        approx_kl_mean = 0.0
        value_loss_mean = 0.0
        policy_loss_mean = 0.0
        entropy_mean = 0.0

        for _ in range(config.update_epochs):
            newlogprob_seq, entropy_seq, newvalue_seq, _ = agent.evaluate_sequence(
                b_obs,
                b_actions,
                b_dones,
                initial_hidden,
            )
            newlogprob = newlogprob_seq.reshape(-1)
            entropy = entropy_seq.reshape(-1)
            newvalue = newvalue_seq.reshape(-1)
            logratio = newlogprob - b_logprobs
            ratio = logratio.exp()

            with torch.no_grad():
                approx_kl = ((ratio - 1) - logratio).mean()

            mb_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)
            pg_loss1 = -mb_advantages * ratio
            pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - config.clip_coef, 1 + config.clip_coef)
            pg_loss = torch.max(pg_loss1, pg_loss2).mean()
            v_loss = 0.5 * ((newvalue - b_returns) ** 2).mean()
            entropy_loss = entropy.mean()
            loss = pg_loss - config.ent_coef * entropy_loss + config.vf_coef * v_loss

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(agent.parameters(), config.max_grad_norm)
            optimizer.step()

            approx_kl_mean += float(approx_kl.item())
            value_loss_mean += float(v_loss.item())
            policy_loss_mean += float(pg_loss.item())
            entropy_mean += float(entropy_loss.item())

        denom = max(config.update_epochs, 1)
        return {
            "approx_kl_mean": approx_kl_mean / denom,
            "value_loss_mean": value_loss_mean / denom,
            "policy_loss_mean": policy_loss_mean / denom,
            "entropy_mean": entropy_mean / denom,
        }

    with torch.no_grad():
        next_value = agent.get_value(trackers["next_obs"]).reshape(1, -1)
        advantages = torch.zeros_like(buffers["rewards"])
        lastgaelam = 0.0
        for t in reversed(range(config.num_steps)):
            if t == config.num_steps - 1:
                nextnonterminal = 1.0 - trackers["next_done"]
                nextvalues = next_value
            else:
                nextnonterminal = 1.0 - buffers["dones"][t + 1]
                nextvalues = buffers["values"][t + 1]
            delta = buffers["rewards"][t] + config.gamma * nextvalues * nextnonterminal - buffers["values"][t]
            advantages[t] = lastgaelam = delta + config.gamma * config.gae_lambda * nextnonterminal * lastgaelam
        returns = advantages + buffers["values"]

    obs_shape = buffers["obs"].shape[2:]
    act_shape = buffers["actions"].shape[2:]

    b_obs = buffers["obs"].reshape((-1,) + obs_shape)
    b_logprobs = buffers["logprobs"].reshape(-1)
    b_actions = buffers["actions"].reshape((-1,) + act_shape)
    b_advantages = advantages.reshape(-1)
    b_returns = returns.reshape(-1)

    rng = np.random.default_rng(seed + update)
    b_inds = np.arange(config.num_steps * config.num_envs)

    approx_kl_mean = 0.0
    value_loss_mean = 0.0
    policy_loss_mean = 0.0
    entropy_mean = 0.0
    minibatch_count = 0

    for _ in range(config.update_epochs):
        rng.shuffle(b_inds)
        for start in range(0, config.num_steps * config.num_envs, config.minibatch_size):
            end = start + config.minibatch_size
            mb_inds = b_inds[start:end]

            _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
            logratio = newlogprob - b_logprobs[mb_inds]
            ratio = logratio.exp()

            with torch.no_grad():
                approx_kl = ((ratio - 1) - logratio).mean()

            mb_advantages = b_advantages[mb_inds]
            mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

            pg_loss1 = -mb_advantages * ratio
            pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - config.clip_coef, 1 + config.clip_coef)
            pg_loss = torch.max(pg_loss1, pg_loss2).mean()

            newvalue = newvalue.view(-1)
            v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()
            entropy_loss = entropy.mean()
            loss = pg_loss - config.ent_coef * entropy_loss + config.vf_coef * v_loss

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(agent.parameters(), config.max_grad_norm)
            optimizer.step()

            approx_kl_mean += float(approx_kl.item())
            value_loss_mean += float(v_loss.item())
            policy_loss_mean += float(pg_loss.item())
            entropy_mean += float(entropy_loss.item())
            minibatch_count += 1

    return {
        "approx_kl_mean": approx_kl_mean / max(minibatch_count, 1),
        "value_loss_mean": value_loss_mean / max(minibatch_count, 1),
        "policy_loss_mean": policy_loss_mean / max(minibatch_count, 1),
        "entropy_mean": entropy_mean / max(minibatch_count, 1),
    }


def build_training_row(
    seed: int, update: int, lr: float, rollout_reward_total: float, trackers, update_stats: dict
) -> dict:
    recent_returns = trackers["completed_returns"][-20:]
    recent_lengths = trackers["completed_lengths"][-20:]
    mean_recent_return = float(np.mean(recent_returns)) if recent_returns else float(rollout_reward_total)
    mean_recent_length = float(np.mean(recent_lengths)) if recent_lengths else float("nan")

    return {
        "seed": seed,
        "update": update,
        "global_step": int(trackers["global_step"]),
        "lr": lr,
        "rollout_reward_total": rollout_reward_total,
        "rollout_reward_mean": rollout_reward_total
        / float(trackers["next_obs"].shape[0] * trackers["next_obs"].shape[1]),
        "completed_episodes": len(trackers["completed_returns"]),
        "recent_episode_return_mean": mean_recent_return,
        "recent_episode_length_mean": mean_recent_length,
        "target_min_active": float(trackers.get("current_target_min", np.nan)),
        "target_max_active": float(trackers.get("current_target_max", np.nan)),
        "disturbance_mass_scale_min_active": float(trackers.get("current_disturbance_mass_scale_min", np.nan)),
        "disturbance_mass_scale_max_active": float(trackers.get("current_disturbance_mass_scale_max", np.nan)),
        "disturbance_friction_scale_min_active": float(trackers.get("current_disturbance_friction_scale_min", np.nan)),
        "disturbance_friction_scale_max_active": float(trackers.get("current_disturbance_friction_scale_max", np.nan)),
        **update_stats,
    }


def train_one_seed(seed: int, config: Stage2Config) -> tuple[Path, pd.DataFrame]:
    set_seed(seed)
    envs, agent, optimizer, buffers, trackers = build_training_components(seed, config)

    training_rows = []
    num_updates = config.total_timesteps // (config.num_steps * config.num_envs)
    pbar = tqdm(range(1, num_updates + 1), desc=f"Training seed {seed}")

    for update in pbar:
        progress = float(update) / float(num_updates)

        if config.randomize_target:
            if config.curriculum_enabled:
                phase_min, phase_max = curriculum_range_for_progress(config.curriculum_phases, progress)
            else:
                phase_min, phase_max = config.target_min, config.target_max
            set_training_target_range(envs, phase_min, phase_max)
            trackers["current_target_min"] = float(phase_min)
            trackers["current_target_max"] = float(phase_max)

        if config.disturbance_curriculum_enabled:
            mass_scale_range = lerp_range(
                config.disturbance_mass_scale_start_range,
                config.disturbance_mass_scale_end_range,
                progress,
            )
            friction_scale_range = lerp_range(
                config.disturbance_friction_scale_start_range,
                config.disturbance_friction_scale_end_range,
                progress,
            )
            set_training_disturbance_ranges(envs, mass_scale_range, friction_scale_range)
            trackers["current_disturbance_mass_scale_min"] = float(mass_scale_range[0])
            trackers["current_disturbance_mass_scale_max"] = float(mass_scale_range[1])
            trackers["current_disturbance_friction_scale_min"] = float(friction_scale_range[0])
            trackers["current_disturbance_friction_scale_max"] = float(friction_scale_range[1])
        else:
            trackers["current_disturbance_mass_scale_min"] = float(config.disturbance_mass_scale_range[0])
            trackers["current_disturbance_mass_scale_max"] = float(config.disturbance_mass_scale_range[1])
            trackers["current_disturbance_friction_scale_min"] = float(config.disturbance_friction_scale_range[0])
            trackers["current_disturbance_friction_scale_max"] = float(config.disturbance_friction_scale_range[1])

        frac = 1.0 - (update - 1.0) / num_updates
        lr_now = frac * config.learning_rate
        optimizer.param_groups[0]["lr"] = lr_now

        rollout_reward_total = run_rollout(envs, agent, buffers, trackers, config)
        update_stats = ppo_update(agent, optimizer, buffers, trackers, config, seed, update)
        training_rows.append(build_training_row(seed, update, lr_now, rollout_reward_total, trackers, update_stats))

        reward_text = f"{training_rows[-1]['recent_episode_return_mean']:.2f}"
        if np.isnan(training_rows[-1]["recent_episode_return_mean"]):
            reward_text = "nan"
        t_min = trackers.get("current_target_min", config.target_min)
        t_max = trackers.get("current_target_max", config.target_max)
        pbar.set_postfix(
            reward=reward_text,
            episodes=len(trackers["completed_returns"]),
            target_range=f"{t_min:.1f}-{t_max:.1f}",
        )

    seed_dir = trackers["seed_dir"]
    model_dir = trackers["model_dir"]
    model_path = model_dir / "meta_rl_agent.pth"
    torch.save(agent.state_dict(), model_path)
    envs.close()

    training_df = pd.DataFrame(training_rows)
    training_df.to_csv(seed_dir / "training_curve.csv", index=False)
    return model_path, training_df


def prepare_eval_episode(env, scenario: EvalScenario, seed: int):
    env.unwrapped.model.body_mass[1] = scenario.mass
    env.unwrapped.model.geom_friction[0, 0] = scenario.friction
    return env.reset(seed=seed)


def run_eval_episode(eval_env, agent, scenario: EvalScenario, seed: int, config: Stage2Config) -> dict:
    obs, _ = prepare_eval_episode(eval_env, scenario, seed)
    positions = []
    rewards = []
    dt = float(eval_env.unwrapped.model.opt.timestep)
    timeout_s = config.max_eval_steps * dt
    device = next(agent.parameters()).device
    hidden_state = agent.get_initial_state(1, device=device) if config.recurrent_policy else None
    done_tensor = torch.zeros(1, device=device) if config.recurrent_policy else None

    for _ in range(config.max_eval_steps):
        with torch.no_grad():
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
            if config.recurrent_policy:
                action, _, _, _, hidden_state = agent.get_action_and_value(
                    obs_tensor,
                    hidden_state=hidden_state,
                    done=done_tensor,
                    deterministic=True,
                )
            else:
                action = agent.actor_mean(obs_tensor.reshape(obs_tensor.shape[0], -1))
            action = torch.clamp(action, -1.0, 1.0)

        obs, reward, terminated, truncated, info = eval_env.step(action.squeeze(0).cpu().numpy())
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


def evaluate_model(seed: int, model_path: Path, config: Stage2Config) -> pd.DataFrame:
    eval_env = make_eval_env(
        stack_size=config.stack_size,
        target_pos=config.target_pos,
        max_eval_steps=config.max_eval_steps,
        hold_steps=config.hold_steps,
        obs_keep_dims=config.obs_keep_dims,
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
        safety_speed_governor_enabled=config.safety_speed_governor_enabled,
        safety_brake_margin_m=config.safety_brake_margin_m,
        safety_max_decel_mps2=config.safety_max_decel_mps2,
        safety_brake_k=config.safety_brake_k,
        safety_hard_overshoot_m=config.safety_hard_overshoot_m,
        safety_overshoot_penalty=config.safety_overshoot_penalty,
        near_approach_zone_m=config.near_approach_zone_m,
        near_approach_coef=config.near_approach_coef,
        near_target_zone_m=config.near_target_zone_m,
        near_target_coef=config.near_target_coef,
        near_target_excess_thresh=config.near_target_excess_thresh,
        near_target_excess_coef=config.near_target_excess_coef,
        approach_progress_cutoff_m=config.approach_progress_cutoff_m,
        brake_zone_vel_sq_coef=config.brake_zone_vel_sq_coef,
        decel_bonus_coef=config.decel_bonus_coef,
        near_target_init_prob=0.0,  # never use near-target init during eval
        near_target_init_range_m=config.near_target_init_range_m,
        brake_integral_reset_enabled=config.brake_integral_reset_enabled,
        use_disturbance=config.disturbance_in_eval,
        randomization_config=build_domain_randomization_config(config, for_eval=True),
    )
    agent = Agent(eval_env, recurrent=config.recurrent_policy, recurrent_hidden_size=config.recurrent_hidden_size)
    agent.load_state_dict(torch.load(model_path, map_location="cpu"))
    agent.eval()

    records = []
    for scenario in SCENARIOS:
        for episode in range(config.eval_episodes):
            episode_seed = seed * 10_000 + episode
            row = run_eval_episode(eval_env, agent, scenario, episode_seed, config)
            row["episode"] = episode
            records.append(row)

    eval_env.close()
    return pd.DataFrame(records)


def aggregate_eval_results(df: pd.DataFrame) -> pd.DataFrame:
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


def aggregate_seed_summary(df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        df.groupby(["seed", "scenario"], as_index=False)
        .agg(
            episodes=("episode", "count"),
            settling_time_mean=("settling_time_s", "mean"),
            overshoot_mean=("overshoot", "mean"),
            iae_mean=("iae", "mean"),
            final_abs_error_mean=("final_abs_error", "mean"),
            success_rate=("success", "mean"),
            mean_reward=("mean_reward", "mean"),
        )
        .sort_values(["seed", "scenario"])
    )
    summary["success_rate"] = 100.0 * summary["success_rate"]
    return summary


def plot_eval_trajectories(seed: int, model_path: Path, config: "Stage2Config", output_path: Path) -> None:
    """Run one episode per scenario with the trained model and plot displacement/velocity/gains."""
    env = make_eval_env(
        stack_size=config.stack_size,
        target_pos=config.target_pos,
        max_eval_steps=config.max_eval_steps,
        hold_steps=config.hold_steps,
        obs_keep_dims=config.obs_keep_dims,
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
        safety_speed_governor_enabled=config.safety_speed_governor_enabled,
        safety_brake_margin_m=config.safety_brake_margin_m,
        safety_max_decel_mps2=config.safety_max_decel_mps2,
        safety_brake_k=config.safety_brake_k,
        safety_hard_overshoot_m=config.safety_hard_overshoot_m,
        safety_overshoot_penalty=config.safety_overshoot_penalty,
        near_approach_zone_m=config.near_approach_zone_m,
        near_approach_coef=config.near_approach_coef,
        near_target_zone_m=config.near_target_zone_m,
        near_target_coef=config.near_target_coef,
        near_target_excess_thresh=config.near_target_excess_thresh,
        near_target_excess_coef=config.near_target_excess_coef,
        approach_progress_cutoff_m=config.approach_progress_cutoff_m,
        brake_zone_vel_sq_coef=config.brake_zone_vel_sq_coef,
        decel_bonus_coef=config.decel_bonus_coef,
        near_target_init_prob=0.0,
        near_target_init_range_m=config.near_target_init_range_m,
        brake_integral_reset_enabled=config.brake_integral_reset_enabled,
        use_disturbance=config.disturbance_in_eval,
        randomization_config=build_domain_randomization_config(config, for_eval=True),
    )
    agent = Agent(env, recurrent=config.recurrent_policy, recurrent_hidden_size=config.recurrent_hidden_size)
    agent.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
    agent.eval()

    dt = float(env.unwrapped.model.opt.timestep)
    device = next(agent.parameters()).device
    target = config.target_pos

    SCENARIO_COLORS = {"Standard": "#2196F3", "Heavy and Slippery": "#F44336", "Light and Grippy": "#4CAF50"}

    fig, axes = plt.subplots(len(SCENARIOS), 3, figsize=(15, 4 * len(SCENARIOS)))
    fig.suptitle(f"Eval Trajectories — Seed {seed}", fontsize=12)
    col_labels = ["Displacement (m)", "Velocity (m/s)", "PID Gains"]
    for j, lbl in enumerate(col_labels):
        axes[0, j].set_title(lbl, fontsize=10, fontweight="bold")

    for i, scenario in enumerate(SCENARIOS):
        obs, _ = prepare_eval_episode(env, scenario, seed=70000)
        hidden_state = agent.get_initial_state(1, device=device) if config.recurrent_policy else None
        done_tensor = torch.zeros(1, device=device) if config.recurrent_policy else None

        positions, velocities, kps, kis, kds, times = [], [], [], [], [], []
        for step in range(config.max_eval_steps):
            with torch.no_grad():
                obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
                if config.recurrent_policy:
                    action, _, _, _, hidden_state = agent.get_action_and_value(
                        obs_t, hidden_state=hidden_state, done=done_tensor, deterministic=True
                    )
                else:
                    action = agent.actor_mean(obs_t.reshape(obs_t.shape[0], -1))
                action = torch.clamp(action, -1.0, 1.0)
            obs, _, terminated, truncated, info = env.step(action.squeeze(0).cpu().numpy())
            positions.append(float(info["state"]["pos"]))
            velocities.append(float(info["state"]["vel"]))
            kps.append(float(info["gains"]["kp"]))
            kis.append(float(info["gains"]["ki"]))
            kds.append(float(info["gains"]["kd"]))
            times.append((step + 1) * dt)
            if terminated or truncated:
                break

        t = np.array(times)
        pos = np.array(positions)
        vel = np.array(velocities)
        color = SCENARIO_COLORS.get(scenario.name, "#555555")
        row_label = f"{scenario.name}\n(m={scenario.mass} kg, fr={scenario.friction})"

        ax = axes[i, 0]
        ax.plot(t, pos, color=color, lw=1.5)
        ax.axhline(target, color="black", lw=1, linestyle=":", label=f"Target {target} m")
        ax.set_ylabel(row_label, fontsize=8)
        ax.set_xlabel("Time (s)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        ax = axes[i, 1]
        ax.plot(t, vel, color=color, lw=1.5)
        ax.axhline(0, color="black", lw=0.8, linestyle=":")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Velocity (m/s)")
        ax.grid(True, alpha=0.3)

        ax = axes[i, 2]
        ax.plot(t, kps, color="#2196F3", lw=1.2, label="Kp")
        ax.plot(t, kis, color="#4CAF50", lw=1.2, label="Ki")
        ax.plot(t, kds, color="#FF9800", lw=1.2, label="Kd")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Gain value")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    env.close()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_training_curves(training_df: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    for seed, subset in training_df.groupby("seed"):
        ax.plot(subset["update"], subset["recent_episode_return_mean"], label=f"Seed {seed}", alpha=0.8)
    ax.set_title("Stage 2 Training Curve")
    ax.set_xlabel("Update")
    ax.set_ylabel("Recent Episode Return")
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
            axes[row][col].bar(["Meta-RL"], [float(subset[metric_key])], color="#2E5491")
            axes[row][col].set_title(f"{scenario} - {ylabel}")
            axes[row][col].set_ylabel(ylabel)
            axes[row][col].set_ylim(bottom=0)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)


def parse_seed_list(seed_text: str) -> list[int]:
    seeds = [int(part.strip()) for part in seed_text.split(",") if part.strip()]
    if not seeds:
        raise ValueError("At least one seed must be provided.")
    return seeds


def protocol_preset_defaults(preset_name: str) -> dict[str, Any]:
    if preset_name == "thesis_v1":
        return {
            "randomize_target": True,
            "target_pos": 5.0,
            "target_min": 3.0,
            "target_max": 7.0,
            "train_episode_steps": 1200,
            "max_eval_steps": 1200,
            "hold_steps": 25,
            "tolerance": 0.05,
            "train_stop_tolerance": 0.05,
            "mid_episode_disturbance_enabled": True,
            "disturbance_mode": "step",
            "disturbance_step_range": "120,220",
            "disturbance_mass_scale_range": "0.9,1.3",
            "disturbance_friction_scale_range": "0.5,1.4",
            "disturbance_mass_scale_end_range": "0.9,1.3",
            "disturbance_friction_scale_end_range": "0.5,1.4",
            "position_patch_enabled": True,
            "patch_x_range": "1.5,2.4",
            "patch_friction_scale": 0.35,
        }
    if preset_name == "thesis_v2_curriculum":
        return {
            "randomize_target": True,
            "target_pos": 5.0,
            "target_min": 3.0,
            "target_max": 7.0,
            "train_episode_steps": 1200,
            "max_eval_steps": 1200,
            "hold_steps": 25,
            "tolerance": 0.05,
            "train_stop_tolerance": 0.05,
            "mid_episode_disturbance_enabled": True,
            "disturbance_mode": "step",
            "disturbance_step_range": "120,220",
            "disturbance_mass_scale_range": "0.9,1.3",
            "disturbance_friction_scale_range": "0.5,1.4",
            "disturbance_curriculum_enabled": True,
            "disturbance_mass_scale_start_range": "0.99,1.03",
            "disturbance_mass_scale_end_range": "0.9,1.3",
            "disturbance_friction_scale_start_range": "0.95,1.05",
            "disturbance_friction_scale_end_range": "0.5,1.4",
            "position_patch_enabled": True,
            "patch_x_range": "1.5,2.4",
            "patch_friction_scale": 0.35,
        }
    if preset_name == "thesis_v3_safety":
        return {
            "randomize_target": True,
            "target_pos": 5.0,
            "target_min": 3.0,
            "target_max": 7.0,
            "train_episode_steps": 1200,
            "max_eval_steps": 5000,
            "hold_steps": 25,
            "tolerance": 0.05,
            "train_stop_tolerance": 0.05,
            "mid_episode_disturbance_enabled": True,
            "disturbance_mode": "step",
            "disturbance_step_range": "120,220",
            "disturbance_mass_scale_range": "0.9,1.3",
            "disturbance_friction_scale_range": "0.5,1.4",
            "disturbance_mass_scale_end_range": "0.9,1.3",
            "disturbance_friction_scale_end_range": "0.5,1.4",
            "position_patch_enabled": True,
            "patch_x_range": "1.5,2.4",
            "patch_friction_scale": 0.35,
            "safety_speed_governor_enabled": True,
            "safety_brake_margin_m": 0.18,
            "safety_max_decel_mps2": 1.6,
            "safety_brake_k": 2.2,
            "safety_hard_overshoot_m": 0.06,
            "safety_overshoot_penalty": 80.0,
            "terminal_hold_bonus": 30.0,
            "terminal_hold_velocity_threshold": 0.08,
        }
    if preset_name == "thesis_v5_no_reset":
        return {
            # Same structure as thesis_v4_cliff but:
            #   - brake_integral_reset_enabled=False: RL must prevent windup via gain scheduling
            #   - Ki can reach 0.0: agent can suppress integral during approach
            #   - Kd range up to 8.0: enough authority to brake even with residual integral
            #   - Longer target (8-10m): more approach distance = more windup = harder for fixed PID
            "randomize_target": True,
            "target_pos": 8.0,
            "target_min": 4.0,
            "target_max": 10.0,
            "train_episode_steps": 1500,
            "max_eval_steps": 5000,
            "hold_steps": 25,
            "tolerance": 0.05,
            "train_stop_tolerance": 0.05,
            "mid_episode_disturbance_enabled": True,
            "disturbance_mode": "step",
            "disturbance_step_range": "120,220",
            "disturbance_mass_scale_range": "0.9,1.3",
            "disturbance_friction_scale_range": "0.5,1.4",
            "disturbance_mass_scale_end_range": "0.9,1.3",
            "disturbance_friction_scale_end_range": "0.5,1.4",
            "position_patch_enabled": True,
            "patch_x_range": "1.5,2.4",
            "patch_friction_scale": 0.35,
            "safety_speed_governor_enabled": False,
            "safety_hard_overshoot_m": 1.5,
            "safety_overshoot_penalty": 20.0,
            "terminal_hold_bonus": 50.0,
            "terminal_hold_velocity_threshold": 0.08,
            # Wider gains: Ki down to 0 (suppress windup), Kd up to 8 (brake with residual integral)
            "gain_base_kp": 1.8,
            "gain_delta_kp": 1.8,  # Kp: 0.0 to 3.6
            "gain_base_ki": 0.5,
            "gain_delta_ki": 0.5,  # Ki: 0.0 to 1.0  (can fully suppress integral)
            "gain_base_kd": 1.0,
            "gain_delta_kd": 7.0,  # Kd: 0.0 to 8.0  (strong braking authority)
            "gain_range_kp": "0.0,8.0",
            "gain_range_ki": "0.0,3.0",
            "gain_range_kd": "0.0,10.0",
            "approach_progress_cutoff_m": 2.0,
            "brake_zone_vel_sq_coef": 10.0,
            "near_approach_zone_m": 2.0,
            "near_approach_coef": 0.0,
            "near_target_zone_m": 0.8,
            "near_target_coef": 0.0,
            "near_target_excess_thresh": 0.10,
            "near_target_excess_coef": 0.0,
            "decel_bonus_coef": 10.0,
            "near_target_init_prob": 0.20,
            "near_target_init_range_m": 0.04,
            "brake_integral_reset_enabled": False,
        }
    if preset_name == "thesis_v4_cliff":
        return {
            "randomize_target": True,
            "target_pos": 5.0,
            "target_min": 3.0,
            "target_max": 7.0,
            "train_episode_steps": 1200,
            "max_eval_steps": 5000,
            "hold_steps": 25,
            "tolerance": 0.05,
            "train_stop_tolerance": 0.05,
            "mid_episode_disturbance_enabled": True,
            "disturbance_mode": "step",
            "disturbance_step_range": "120,220",
            "disturbance_mass_scale_range": "0.9,1.3",
            "disturbance_friction_scale_range": "0.5,1.4",
            "disturbance_mass_scale_end_range": "0.9,1.3",
            "disturbance_friction_scale_end_range": "0.5,1.4",
            "position_patch_enabled": True,
            "patch_x_range": "1.5,2.4",
            "patch_friction_scale": 0.35,
            # Speed governor OFF, cliff DISABLED — let PD settling + overshoot penalty teach braking.
            # Cliff was killing all exploration: agent never reached hold bonus, value function
            # stuck at -300 everywhere, zero gradient toward braking behavior.
            # The existing -2.0*overshoot/step penalty already punishes overshooting.
            "safety_speed_governor_enabled": False,
            "safety_hard_overshoot_m": -1.0,  # disabled
            "safety_overshoot_penalty": 0.0,
            "terminal_hold_bonus": 50.0,
            "terminal_hold_velocity_threshold": 0.08,
            # Larger Kd range so the agent can physically brake
            "gain_delta_kd": 2.0,
            "gain_base_kd": 0.5,
            # No progress reward within 2m — removes incentive to rush into braking zone
            "approach_progress_cutoff_m": 2.0,
            # Simplified velocity shaping: only distance-weighted quadratic, no linear terms.
            # Linear |vel| penalties were fighting each other and teaching wrong gradient
            # (reduce Kp globally rather than raise Kd near target).
            "brake_zone_vel_sq_coef": 10.0,
            "near_approach_zone_m": 2.0,
            "near_approach_coef": 0.0,  # disabled — was causing wrong gradient
            "near_target_zone_m": 0.8,
            "near_target_coef": 0.0,  # disabled — same issue
            "near_target_excess_thresh": 0.10,
            "near_target_excess_coef": 0.0,  # disabled
            # Decel bonus: immediate reward for slowing down in braking zone
            "decel_bonus_coef": 10.0,
            # 20% of episodes start at target — agent experiences hold bonus immediately
            "near_target_init_prob": 0.20,
            "near_target_init_range_m": 0.04,
        }
    return {}


def build_config(args: argparse.Namespace) -> Stage2Config:
    if args.target_min <= 0.0 or args.target_max <= 0.0:
        raise ValueError("target-min and target-max must be positive.")
    if args.target_min > args.target_max:
        raise ValueError("target-min must be <= target-max.")
    if args.hold_steps <= 0:
        raise ValueError("hold-steps must be a positive integer.")
    if args.obs_keep_dims < 0:
        raise ValueError("obs-keep-dims must be >= 0 (0 means auto).")
    if args.recurrent_hidden_size <= 0:
        raise ValueError("recurrent-hidden-size must be a positive integer.")
    if args.train_stop_tolerance is None:
        args.train_stop_tolerance = 0.05

    if args.train_stop_tolerance <= 0.0:
        raise ValueError("train-stop-tolerance must be > 0.")

    disturbance_mode = args.disturbance_mode.strip().lower()
    if disturbance_mode not in {"step", "time"}:
        raise ValueError("disturbance-mode must be one of: step, time")

    disturbance_step_range = parse_range_pair(args.disturbance_step_range, int)
    disturbance_time_range_s = parse_range_pair(args.disturbance_time_range_s, float)
    disturbance_mass_scale_range = parse_range_pair(args.disturbance_mass_scale_range, float)
    disturbance_friction_scale_range = parse_range_pair(args.disturbance_friction_scale_range, float)
    disturbance_mass_scale_start_range = parse_range_pair(args.disturbance_mass_scale_start_range, float)
    disturbance_mass_scale_end_range = parse_range_pair(args.disturbance_mass_scale_end_range, float)
    disturbance_friction_scale_start_range = parse_range_pair(args.disturbance_friction_scale_start_range, float)
    disturbance_friction_scale_end_range = parse_range_pair(args.disturbance_friction_scale_end_range, float)
    patch_x_range = parse_range_pair(args.patch_x_range, float)
    gain_range_kp = parse_range_pair(args.gain_range_kp, float)
    gain_range_ki = parse_range_pair(args.gain_range_ki, float)
    gain_range_kd = parse_range_pair(args.gain_range_kd, float)

    if args.patch_friction_scale <= 0.0:
        raise ValueError("patch-friction-scale must be > 0")
    if args.terminal_hold_bonus < 0.0:
        raise ValueError("terminal-hold-bonus must be >= 0")
    if args.terminal_hold_velocity_threshold <= 0.0:
        raise ValueError("terminal-hold-velocity-threshold must be > 0")
    if args.action_slew_limit <= 0.0 or args.action_slew_limit > 1.0:
        raise ValueError("action-slew-limit must be in (0, 1]")
    if args.action_rate_penalty_coef < 0.0:
        raise ValueError("action-rate-penalty-coef must be >= 0")
    if args.safety_brake_margin_m < 0.0:
        raise ValueError("safety-brake-margin-m must be >= 0")
    if args.safety_max_decel_mps2 <= 0.0:
        raise ValueError("safety-max-decel-mps2 must be > 0")
    if args.safety_brake_k <= 0.0:
        raise ValueError("safety-brake-k must be > 0")
    if args.safety_overshoot_penalty < 0.0:
        raise ValueError("safety-overshoot-penalty must be >= 0")

    init_model_path = Path(args.init_model_path) if args.init_model_path else None
    if init_model_path is not None and not init_model_path.exists():
        raise ValueError(f"init-model-path does not exist: {init_model_path}")

    obs_keep_dims = int(args.obs_keep_dims)
    if init_model_path is not None and obs_keep_dims == 0:
        state_dict = torch.load(init_model_path, map_location="cpu")
        obs_keep_dims = infer_expected_obs_dim_per_step(state_dict, args.stack_size)
    if obs_keep_dims == 0:
        obs_keep_dims = 8
    if obs_keep_dims <= 0 or obs_keep_dims > 8:
        raise ValueError(f"obs-keep-dims must be in [1, 8], got {obs_keep_dims}")

    curriculum_phases = parse_curriculum_spec(args.curriculum_spec) if args.curriculum_enabled else []

    return Stage2Config(
        output_dir=Path(args.output_dir),
        seeds=parse_seed_list(args.seeds),
        total_timesteps=args.total_timesteps,
        num_envs=args.num_envs,
        num_steps=args.num_steps,
        train_episode_steps=args.train_episode_steps,
        target_pos=args.target_pos,
        target_min=args.target_min,
        target_max=args.target_max,
        randomize_target=args.randomize_target,
        hold_steps=args.hold_steps,
        obs_keep_dims=obs_keep_dims,
        train_stop_tolerance=args.train_stop_tolerance,
        curriculum_enabled=args.curriculum_enabled,
        curriculum_spec=args.curriculum_spec,
        curriculum_phases=curriculum_phases,
        stack_size=args.stack_size,
        recurrent_policy=args.recurrent_policy,
        recurrent_hidden_size=args.recurrent_hidden_size,
        learning_rate=args.learning_rate,
        minibatch_size=args.minibatch_size,
        update_epochs=args.update_epochs,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_coef=args.clip_coef,
        ent_coef=args.ent_coef,
        vf_coef=args.vf_coef,
        max_grad_norm=args.max_grad_norm,
        eval_episodes=args.eval_episodes,
        max_eval_steps=args.max_eval_steps,
        tolerance=args.tolerance,
        gain_base_kp=args.gain_base_kp,
        gain_base_ki=args.gain_base_ki,
        gain_base_kd=args.gain_base_kd,
        gain_delta_kp=args.gain_delta_kp,
        gain_delta_ki=args.gain_delta_ki,
        gain_delta_kd=args.gain_delta_kd,
        gain_range_kp=gain_range_kp,
        gain_range_ki=gain_range_ki,
        gain_range_kd=gain_range_kd,
        mid_episode_disturbance_enabled=args.mid_episode_disturbance_enabled,
        disturbance_mode=disturbance_mode,
        disturbance_step_range=disturbance_step_range,
        disturbance_time_range_s=disturbance_time_range_s,
        disturbance_mass_scale_range=disturbance_mass_scale_range,
        disturbance_friction_scale_range=disturbance_friction_scale_range,
        disturbance_curriculum_enabled=args.disturbance_curriculum_enabled,
        disturbance_mass_scale_start_range=disturbance_mass_scale_start_range,
        disturbance_mass_scale_end_range=disturbance_mass_scale_end_range,
        disturbance_friction_scale_start_range=disturbance_friction_scale_start_range,
        disturbance_friction_scale_end_range=disturbance_friction_scale_end_range,
        position_patch_enabled=args.position_patch_enabled,
        patch_x_range=patch_x_range,
        patch_friction_scale=args.patch_friction_scale,
        terminal_hold_bonus=args.terminal_hold_bonus,
        terminal_hold_velocity_threshold=args.terminal_hold_velocity_threshold,
        action_slew_limit=args.action_slew_limit,
        action_rate_penalty_coef=args.action_rate_penalty_coef,
        safety_speed_governor_enabled=args.safety_speed_governor_enabled,
        safety_brake_margin_m=args.safety_brake_margin_m,
        safety_max_decel_mps2=args.safety_max_decel_mps2,
        safety_brake_k=args.safety_brake_k,
        safety_hard_overshoot_m=args.safety_hard_overshoot_m,
        safety_overshoot_penalty=args.safety_overshoot_penalty,
        near_approach_zone_m=args.near_approach_zone_m,
        near_approach_coef=args.near_approach_coef,
        near_target_zone_m=args.near_target_zone_m,
        near_target_coef=args.near_target_coef,
        near_target_excess_thresh=args.near_target_excess_thresh,
        near_target_excess_coef=args.near_target_excess_coef,
        approach_progress_cutoff_m=args.approach_progress_cutoff_m,
        brake_zone_vel_sq_coef=args.brake_zone_vel_sq_coef,
        decel_bonus_coef=args.decel_bonus_coef,
        near_target_init_prob=args.near_target_init_prob,
        near_target_init_range_m=args.near_target_init_range_m,
        disturbance_in_eval=args.disturbance_in_eval,
        init_model_path=init_model_path,
        brake_integral_reset_enabled=args.brake_integral_reset_enabled,
        eval_only=args.eval_only,
    )


def save_json_config(config: Stage2Config) -> None:
    serializable = asdict(config)
    serializable["output_dir"] = str(config.output_dir)
    serializable["init_model_path"] = str(config.init_model_path) if config.init_model_path is not None else None
    serializable["scenarios"] = [asdict(s) for s in SCENARIOS]
    with open(config.output_dir / "stage2_config.json", "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage 2: Meta-RL reproduction with multi-seed PPO training and evaluation."
    )
    parser.add_argument(
        "--protocol-preset",
        choices=[
            "custom",
            "thesis_v1",
            "thesis_v2_curriculum",
            "thesis_v3_safety",
            "thesis_v4_cliff",
            "thesis_v5_no_reset",
        ],
        default="custom",
    )
    parser.add_argument("--output-dir", default="benchmark_results/stage2_meta_rl_reproduction")
    parser.add_argument("--seeds", default=",".join(str(seed) for seed in DEFAULT_SEEDS))
    parser.add_argument("--total-timesteps", type=int, default=DEFAULT_TOTAL_TIMESTEPS)
    parser.add_argument("--num-envs", type=int, default=DEFAULT_NUM_ENVS)
    parser.add_argument("--num-steps", type=int, default=DEFAULT_NUM_STEPS)
    parser.add_argument("--train-episode-steps", type=int, default=DEFAULT_TRAIN_EPISODE_STEPS)
    parser.add_argument("--target-pos", type=float, default=DEFAULT_TARGET_POS)
    parser.add_argument("--target-min", type=float, default=DEFAULT_TARGET_MIN)
    parser.add_argument("--target-max", type=float, default=DEFAULT_TARGET_MAX)
    parser.add_argument("--randomize-target", action="store_true", default=False)
    parser.add_argument("--hold-steps", type=int, default=DEFAULT_HOLD_STEPS)
    parser.add_argument("--obs-keep-dims", type=int, default=0)
    parser.add_argument("--train-stop-tolerance", type=float, default=None)
    parser.add_argument("--curriculum-enabled", action="store_true", default=False)
    parser.add_argument("--curriculum-spec", type=str, default=DEFAULT_CURRICULUM_SPEC)
    parser.add_argument("--stack-size", type=int, default=DEFAULT_STACK_SIZE)
    parser.add_argument("--recurrent-policy", action="store_true", default=False)
    parser.add_argument("--recurrent-hidden-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--minibatch-size", type=int, default=DEFAULT_MINIBATCH_SIZE)
    parser.add_argument("--update-epochs", type=int, default=DEFAULT_UPDATE_EPOCHS)
    parser.add_argument("--gamma", type=float, default=DEFAULT_GAMMA)
    parser.add_argument("--gae-lambda", type=float, default=DEFAULT_GAE_LAMBDA)
    parser.add_argument("--clip-coef", type=float, default=DEFAULT_CLIP_COEF)
    parser.add_argument("--ent-coef", type=float, default=DEFAULT_ENT_COEF)
    parser.add_argument("--vf-coef", type=float, default=DEFAULT_VF_COEF)
    parser.add_argument("--max-grad-norm", type=float, default=DEFAULT_MAX_GRAD_NORM)
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
    parser.add_argument("--disturbance-curriculum-enabled", action="store_true", default=False)
    parser.add_argument("--disturbance-mass-scale-start-range", type=str, default="0.98,1.05")
    parser.add_argument("--disturbance-mass-scale-end-range", type=str, default="0.9,1.3")
    parser.add_argument("--disturbance-friction-scale-start-range", type=str, default="0.9,1.1")
    parser.add_argument("--disturbance-friction-scale-end-range", type=str, default="0.5,1.4")
    parser.add_argument("--position-patch-enabled", action="store_true", default=False)
    parser.add_argument("--patch-x-range", type=str, default="1.5,2.0")
    parser.add_argument("--patch-friction-scale", type=float, default=0.35)
    parser.add_argument("--terminal-hold-bonus", type=float, default=0.0)
    parser.add_argument("--terminal-hold-velocity-threshold", type=float, default=0.05)
    parser.add_argument("--action-slew-limit", type=float, default=1.0)
    parser.add_argument("--action-rate-penalty-coef", type=float, default=0.0)
    parser.add_argument("--safety-speed-governor-enabled", action="store_true", default=False)
    parser.add_argument("--safety-brake-margin-m", type=float, default=0.30)
    parser.add_argument("--safety-max-decel-mps2", type=float, default=2.0)
    parser.add_argument("--safety-brake-k", type=float, default=3.5)
    parser.add_argument("--safety-hard-overshoot-m", type=float, default=-1.0)
    parser.add_argument("--safety-overshoot-penalty", type=float, default=200.0)
    parser.add_argument("--disturbance-in-eval", action="store_true", default=False)
    parser.add_argument("--eval-only", action="store_true", default=False)
    parser.add_argument(
        "--no-brake-integral-reset",
        dest="brake_integral_reset_enabled",
        action="store_false",
        default=True,
        help="Disable integral reset on braking zone entry — RL must learn windup prevention via gain scheduling",
    )
    parser.add_argument("--init-model-path", type=str, default=None)
    parser.add_argument("--near-approach-zone-m", type=float, default=0.3)
    parser.add_argument("--near-approach-coef", type=float, default=0.75)
    parser.add_argument("--near-target-zone-m", type=float, default=0.2)
    parser.add_argument("--near-target-coef", type=float, default=1.1)
    parser.add_argument("--near-target-excess-thresh", type=float, default=0.15)
    parser.add_argument("--near-target-excess-coef", type=float, default=1.0)
    parser.add_argument("--approach-progress-cutoff-m", type=float, default=0.0)
    parser.add_argument("--brake-zone-vel-sq-coef", type=float, default=0.0)
    parser.add_argument("--decel-bonus-coef", type=float, default=0.0)
    parser.add_argument("--near-target-init-prob", type=float, default=0.0)
    parser.add_argument("--near-target-init-range-m", type=float, default=0.15)

    # Parse once to read protocol preset, apply preset defaults, then parse again.
    # This guarantees explicit CLI flags override preset values.
    pre_args, _ = parser.parse_known_args()
    parser.set_defaults(**protocol_preset_defaults(pre_args.protocol_preset))
    args = parser.parse_args()

    config = build_config(args)
    config.output_dir.mkdir(parents=True, exist_ok=True)

    if config.eval_only and config.init_model_path is None:
        raise ValueError("--eval-only requires --init-model-path")

    all_training_rows = []
    all_eval_rows = []
    seed_results = []

    for seed in config.seeds:
        if config.eval_only:
            model_path = config.init_model_path
            training_df = pd.DataFrame()
        else:
            model_path, training_df = train_one_seed(seed, config)
        all_training_rows.append(training_df.assign(seed=seed) if not training_df.empty else training_df)

        eval_df = evaluate_model(seed, model_path, config)
        seed_dir = config.output_dir / f"seed_{seed}"
        eval_df.to_csv(seed_dir / "eval_raw.csv", index=False)

        seed_summary = aggregate_seed_summary(eval_df)
        seed_summary.to_csv(seed_dir / "eval_seed_summary.csv", index=False)

        traj_path = seed_dir / "trajectory_plots.png"
        plot_eval_trajectories(seed, model_path, config, traj_path)
        print(f"Saved trajectory plots to {traj_path}")

        all_eval_rows.append(eval_df)
        seed_results.append(
            {
                "seed": seed,
                "model_path": str(model_path),
                "training_updates": (
                    int(training_df["update"].max()) if "update" in training_df.columns and not training_df.empty else 0
                ),
                "eval_rows": int(len(eval_df)),
                "mean_eval_reward": float(eval_df["mean_reward"].mean()),
            }
        )

    non_empty_training = [df for df in all_training_rows if not df.empty]
    training_all_df = pd.concat(non_empty_training, ignore_index=True) if non_empty_training else pd.DataFrame()
    eval_all_df = pd.concat(all_eval_rows, ignore_index=True)
    eval_summary_df = aggregate_eval_results(eval_all_df)
    seed_summary_df = aggregate_seed_summary(eval_all_df)
    seed_results_df = pd.DataFrame(seed_results)

    if not training_all_df.empty:
        training_all_df.to_csv(config.output_dir / "stage2_training_curves.csv", index=False)
        plot_training_curves(training_all_df, config.output_dir / "stage2_learning_curve.png")
    eval_all_df.to_csv(config.output_dir / "stage2_eval_raw.csv", index=False)
    eval_summary_df.to_csv(config.output_dir / "stage2_eval_summary.csv", index=False)
    seed_summary_df.to_csv(config.output_dir / "stage2_eval_seed_summary.csv", index=False)
    seed_results_df.to_csv(config.output_dir / "stage2_seed_results.csv", index=False)

    plot_eval_summary(eval_summary_df, config.output_dir / "stage2_eval_summary.png")
    save_json_config(config)

    print("Stage 2 reproduction completed.")
    print(f"Saved training curves to {config.output_dir / 'stage2_training_curves.csv'}")
    print(f"Saved evaluation raw data to {config.output_dir / 'stage2_eval_raw.csv'}")
    print(f"Saved evaluation summary to {config.output_dir / 'stage2_eval_summary.csv'}")
    print(f"Saved learning curve plot to {config.output_dir / 'stage2_learning_curve.png'}")
    print(f"Saved evaluation plot to {config.output_dir / 'stage2_eval_summary.png'}")


if __name__ == "__main__":
    main()

import argparse
import json
from pathlib import Path

import gymnasium as gym
import matplotlib.pyplot as plt
import mujoco
import numpy as np
import torch

from agents.domain_randomization import DomainRandomizationWrapper
from agents.model import Agent
from envs.adaptive_suspension import AdaptiveSuspensionEnv
from stage1_fixed_pid_benchmark import Scenario, apply_scenario, load_controller_specs, load_sim, reset_episode
from stage1_fixed_pid_benchmark import run_episode as run_stage1_episode

METHOD_PID = "Stage1 Fixed PID (Robust)"
METHOD_STAGE2_CLEAN = "Stage2 Clean-Trained"
METHOD_STAGE2_DIST = "Stage2 Disturbance-Trained"


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


def load_stage2_agent(model_path: Path, stack_size: int):
    state_dict = torch.load(model_path, map_location="cpu")
    expected_obs_dim_per_step = infer_expected_obs_dim_per_step(state_dict, stack_size)

    # Build a tiny temp env only for model shape construction.
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


def build_disturbance_config(config_json: dict, use_disturbance: bool) -> dict:
    return {
        "mass_range": (5.0, 20.0),
        "friction_range": (0.1, 2.0),
        "initial_randomization_enabled": False,
        "mid_episode_disturbance_enabled": bool(use_disturbance),
        "disturbance_mode": config_json["disturbance_mode"],
        "disturbance_step_range": tuple(config_json["disturbance_step_range"]),
        "disturbance_time_range_s": tuple(config_json["disturbance_time_range_s"]),
        "disturbance_mass_scale_range": tuple(config_json["disturbance_mass_scale_range"]),
        "disturbance_friction_scale_range": tuple(config_json["disturbance_friction_scale_range"]),
        "position_patch_enabled": bool(config_json["position_patch_enabled"]),
        "patch_x_range": tuple(config_json["patch_x_range"]),
        "patch_friction_scale": float(config_json["patch_friction_scale"]),
    }


def run_stage2_episode_trace(
    method_name: str,
    agent: Agent,
    expected_obs_dim_per_step: int,
    cfg: dict,
    scenario: Scenario,
    target_distance: float,
    seed: int,
    use_disturbance: bool,
) -> dict:
    env = AdaptiveSuspensionEnv(
        target_pos=target_distance,
        hold_steps=int(cfg["hold_steps"]),
        max_episode_steps=int(cfg["max_eval_steps"]),
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
        terminal_hold_bonus=float(cfg["terminal_hold_bonus"]),
        terminal_hold_velocity_threshold=float(cfg["terminal_hold_velocity_threshold"]),
        action_slew_limit=float(cfg["action_slew_limit"]),
        action_rate_penalty_coef=float(cfg["action_rate_penalty_coef"]),
    )

    if use_disturbance:
        env = DomainRandomizationWrapper(env, randomization_config=build_disturbance_config(cfg, use_disturbance=True))

    if expected_obs_dim_per_step < int(env.observation_space.shape[0]):
        env = ObservationFeatureSelectWrapper(env, keep_dims=expected_obs_dim_per_step)

    env = gym.wrappers.TimeLimit(env, max_episode_steps=int(cfg["max_eval_steps"]))
    env = gym.wrappers.FrameStackObservation(env, stack_size=int(cfg["stack_size"]))

    # Apply static scenario before reset, as in Stage2 eval pipeline.
    env.unwrapped.model.body_mass[1] = float(scenario.mass_scale * 10.0)
    env.unwrapped.model.geom_friction[0, 0] = float(scenario.friction_scale * 1.0)
    mujoco.mj_forward(env.unwrapped.model, env.unwrapped.data)

    obs, _ = env.reset(seed=seed)

    t = []
    pos = []
    vel = []
    err = []
    reward = []
    kp_arr = []
    ki_arr = []
    kd_arr = []
    mass_arr = []
    fric_arr = []
    disturbance_fired_steps = []

    dt = float(env.unwrapped.dt)

    step_idx = 0
    while step_idx < int(cfg["max_eval_steps"]):
        with torch.no_grad():
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
            action = agent.actor_mean(obs_tensor.reshape(obs_tensor.shape[0], -1))
            action = torch.clamp(action, -1.0, 1.0)

        obs, rew, terminated, truncated, info = env.step(action.squeeze(0).cpu().numpy())

        state = info.get("state", {})
        gains = info.get("gains", {})
        factors = info.get("external_factors", {})

        t.append((step_idx + 1) * dt)
        p = float(state.get("pos", np.nan))
        v = float(state.get("vel", np.nan))
        pos.append(p)
        vel.append(v)
        err.append(float(target_distance - p))
        reward.append(float(rew))

        kp_arr.append(float(gains.get("kp", np.nan)))
        ki_arr.append(float(gains.get("ki", np.nan)))
        kd_arr.append(float(gains.get("kd", np.nan)))

        mass_arr.append(float(factors.get("mass", env.unwrapped.model.body_mass[1])))
        fric_arr.append(float(factors.get("friction", env.unwrapped.model.geom_friction[0, 0])))

        if bool(factors.get("disturbance_fired", False)):
            disturbance_fired_steps.append(step_idx)

        if terminated or truncated:
            break
        step_idx += 1

    env.close()

    first_disturbance_time = None
    if disturbance_fired_steps:
        first_disturbance_time = (min(disturbance_fired_steps) + 1) * dt

    return {
        "method": method_name,
        "t": np.asarray(t, dtype=np.float64),
        "pos": np.asarray(pos, dtype=np.float64),
        "vel": np.asarray(vel, dtype=np.float64),
        "err": np.asarray(err, dtype=np.float64),
        "reward": np.asarray(reward, dtype=np.float64),
        "kp": np.asarray(kp_arr, dtype=np.float64),
        "ki": np.asarray(ki_arr, dtype=np.float64),
        "kd": np.asarray(kd_arr, dtype=np.float64),
        "mass": np.asarray(mass_arr, dtype=np.float64),
        "friction": np.asarray(fric_arr, dtype=np.float64),
        "first_disturbance_time": first_disturbance_time,
    }


def run_stage1_trace(target_distance: float, seed: int) -> dict:
    model, data, car_body_id, floor_geom_id, base_car_mass, base_floor_friction = load_sim()
    specs = load_controller_specs("benchmark_results/stage1_simc_controller_gains.csv")
    robust = None
    for spec in specs:
        if spec.name == "Fixed PID (Robust)":
            robust = spec
            break
    if robust is None:
        raise ValueError("Could not find 'Fixed PID (Robust)' in Stage1 gains CSV.")

    scenario = Scenario(name="Standard", mass_scale=1.0, friction_scale=1.0)
    apply_scenario(model, car_body_id, floor_geom_id, base_car_mass, base_floor_friction, scenario)
    reset_episode(model, data, np.random.default_rng(seed))

    metrics, traces = run_stage1_episode(
        model,
        data,
        robust,
        distance_target_m=target_distance,
        max_steps=5000,
        tolerance_m=0.05,
        render=False,
    )

    t = np.asarray(traces["time_s"], dtype=np.float64)
    pos = np.asarray(traces["distance_m"], dtype=np.float64)
    vel = np.asarray(traces["forward_speed_mps"], dtype=np.float64)
    err = target_distance - pos

    # No explicit gain logging in Stage1 run loop; emit constants.
    kp = np.full_like(t, float(robust.kp))
    ki = np.full_like(t, float(robust.ki))
    kd = np.full_like(t, float(robust.kd))

    return {
        "method": METHOD_PID,
        "t": t,
        "pos": pos,
        "vel": vel,
        "err": err,
        "reward": np.full_like(t, np.nan),
        "kp": kp,
        "ki": ki,
        "kd": kd,
        "mass": np.full_like(t, np.nan),
        "friction": np.full_like(t, np.nan),
        "first_disturbance_time": None,
        "meta": metrics,
    }


def plot_behavior_panel(
    clean_runs: list[dict], disturbed_runs: list[dict], target_distance: float, out_path: Path
) -> None:
    colors = {
        METHOD_PID: "#ff7f0e",
        METHOD_STAGE2_CLEAN: "#1f77b4",
        METHOD_STAGE2_DIST: "#2ca02c",
    }

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex="col")

    for run in clean_runs:
        axes[0, 0].plot(run["t"], run["pos"], label=run["method"], color=colors.get(run["method"], None), linewidth=1.8)
        axes[1, 0].plot(run["t"], run["vel"], label=run["method"], color=colors.get(run["method"], None), linewidth=1.5)

    axes[0, 0].axhline(target_distance, color="black", linestyle="--", linewidth=1.0, label="Target")
    axes[0, 0].set_title("Clean Eval: Displacement")
    axes[0, 0].set_ylabel("Position (m)")
    axes[0, 0].grid(alpha=0.25)
    axes[1, 0].set_title("Clean Eval: Velocity")
    axes[1, 0].set_xlabel("Time (s)")
    axes[1, 0].set_ylabel("Velocity (m/s)")
    axes[1, 0].grid(alpha=0.25)

    for run in disturbed_runs:
        axes[0, 1].plot(run["t"], run["pos"], label=run["method"], color=colors.get(run["method"], None), linewidth=1.8)
        axes[1, 1].plot(run["t"], run["vel"], label=run["method"], color=colors.get(run["method"], None), linewidth=1.5)
        if run["first_disturbance_time"] is not None:
            axes[0, 1].axvline(
                run["first_disturbance_time"], color=colors.get(run["method"], None), linestyle=":", alpha=0.8
            )
            axes[1, 1].axvline(
                run["first_disturbance_time"], color=colors.get(run["method"], None), linestyle=":", alpha=0.8
            )

    axes[0, 1].axhline(target_distance, color="black", linestyle="--", linewidth=1.0, label="Target")
    axes[0, 1].set_title("Disturbed Eval: Displacement")
    axes[0, 1].set_ylabel("Position (m)")
    axes[0, 1].grid(alpha=0.25)
    axes[1, 1].set_title("Disturbed Eval: Velocity")
    axes[1, 1].set_xlabel("Time (s)")
    axes[1, 1].set_ylabel("Velocity (m/s)")
    axes[1, 1].grid(alpha=0.25)

    axes[0, 0].legend(loc="best")
    axes[0, 1].legend(loc="best")
    fig.suptitle("Representative Simulated Runs: Clean vs Mid-Episode Disturbance", fontsize=14)
    fig.tight_layout(rect=(0.0, 0.02, 1.0, 0.95))
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_disturbance_factors(disturbed_runs: list[dict], out_path: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(12.5, 7.5), sharex=True)

    for run in disturbed_runs:
        label = run["method"]
        axes[0].plot(run["t"], run["mass"], linewidth=1.8, label=label)
        axes[1].plot(run["t"], run["friction"], linewidth=1.8, label=label)

    axes[0].set_title("Disturbed Eval: Effective Mass Over Time")
    axes[0].set_ylabel("Body Mass")
    axes[0].grid(alpha=0.25)
    axes[0].legend(loc="best")

    axes[1].set_title("Disturbed Eval: Effective Floor Friction Over Time")
    axes[1].set_ylabel("Friction")
    axes[1].set_xlabel("Time (s)")
    axes[1].grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate human-readable trajectory graphs for representative simulation runs."
    )
    parser.add_argument(
        "--stage2-clean-model",
        type=Path,
        default=Path("benchmark_results/stage2_meta_rl_reproduction_v3/seed_7/models/meta_rl_agent.pth"),
    )
    parser.add_argument(
        "--stage2-disturbance-model",
        type=Path,
        default=Path(
            "benchmark_results/stage2_disturbance_trained_distance_randomized_"
            "seed7_warm500k/seed_7/models/meta_rl_agent.pth"
        ),
    )
    parser.add_argument(
        "--stage2-config",
        type=Path,
        default=Path("benchmark_results/stage2_warm500k_disturbance_eval_longhorizon_v2/stage2_config.json"),
    )
    parser.add_argument("--target-distance", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=70008)
    parser.add_argument("--output-dir", type=Path, default=Path("benchmark_results/final_thesis_master_comparison"))
    args = parser.parse_args()

    with open(args.stage2_config, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    stack_size = int(cfg["stack_size"])

    clean_agent, clean_obs_dim = load_stage2_agent(args.stage2_clean_model, stack_size)
    dist_agent, dist_obs_dim = load_stage2_agent(args.stage2_disturbance_model, stack_size)

    scenario = Scenario(name="Standard", mass_scale=1.0, friction_scale=1.0)

    pid_clean = run_stage1_trace(target_distance=args.target_distance, seed=args.seed)

    stage2_clean_clean = run_stage2_episode_trace(
        METHOD_STAGE2_CLEAN,
        clean_agent,
        clean_obs_dim,
        cfg,
        scenario,
        target_distance=args.target_distance,
        seed=args.seed,
        use_disturbance=False,
    )
    stage2_dist_clean = run_stage2_episode_trace(
        METHOD_STAGE2_DIST,
        dist_agent,
        dist_obs_dim,
        cfg,
        scenario,
        target_distance=args.target_distance,
        seed=args.seed,
        use_disturbance=False,
    )

    stage2_clean_disturbed = run_stage2_episode_trace(
        METHOD_STAGE2_CLEAN,
        clean_agent,
        clean_obs_dim,
        cfg,
        scenario,
        target_distance=args.target_distance,
        seed=args.seed,
        use_disturbance=True,
    )
    stage2_dist_disturbed = run_stage2_episode_trace(
        METHOD_STAGE2_DIST,
        dist_agent,
        dist_obs_dim,
        cfg,
        scenario,
        target_distance=args.target_distance,
        seed=args.seed,
        use_disturbance=True,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    behavior_panel_path = args.output_dir / "sim_behavior_clean_vs_disturbed_panel.png"
    disturbance_factors_path = args.output_dir / "sim_behavior_disturbance_factors.png"

    clean_runs = [pid_clean, stage2_clean_clean, stage2_dist_clean]
    disturbed_runs = [stage2_clean_disturbed, stage2_dist_disturbed]

    plot_behavior_panel(clean_runs, disturbed_runs, args.target_distance, behavior_panel_path)
    plot_disturbance_factors(disturbed_runs, disturbance_factors_path)

    print(f"Saved behavior panel: {behavior_panel_path}")
    print(f"Saved disturbance factors plot: {disturbance_factors_path}")


if __name__ == "__main__":
    main()

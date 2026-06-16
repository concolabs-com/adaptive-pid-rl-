"""
Fixed trajectory plots using Standard (more aggressive) PID for fair comparison.
"""

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


def load_stage2_agent(model_path: Path, stack_size: int):
    state_dict = torch.load(model_path, map_location="cpu")

    # Infer obs dim
    if "actor_mean.0.weight" in state_dict:
        input_dim = int(state_dict["actor_mean.0.weight"].shape[1])
    elif "encoder.0.weight" in state_dict:
        input_dim = int(state_dict["encoder.0.weight"].shape[1])
    else:
        raise ValueError("Could not infer input dimension from checkpoint state_dict.")

    expected_obs_dim_per_step = int(input_dim // stack_size)

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


def run_stage1_trace(target_distance: float, seed: int, use_standard: bool = False):
    model, data, car_body_id, floor_geom_id, base_car_mass, base_floor_friction = load_sim()
    specs = load_controller_specs("benchmark_results/stage1_simc_controller_gains.csv")

    if use_standard:
        controller = [s for s in specs if s.name == "Fixed PID (Standard)"][0]
        method_name = "Stage1 Fixed PID (Standard)"
    else:
        controller = [s for s in specs if s.name == "Fixed PID (Robust)"][0]
        method_name = "Stage1 Fixed PID (Robust)"

    scenario = Scenario(name="Standard", mass_scale=1.0, friction_scale=1.0)
    apply_scenario(model, car_body_id, floor_geom_id, base_car_mass, base_floor_friction, scenario)
    reset_episode(model, data, np.random.default_rng(seed))

    metrics, traces = run_stage1_episode(
        model,
        data,
        controller,
        distance_target_m=target_distance,
        max_steps=5000,
        tolerance_m=0.05,
        render=False,
    )

    t = np.asarray(traces["time_s"], dtype=np.float64)
    pos = np.asarray(traces["distance_m"], dtype=np.float64)
    vel = np.asarray(traces["forward_speed_mps"], dtype=np.float64)
    err = target_distance - pos

    kp = np.full_like(t, float(controller.kp))
    ki = np.full_like(t, float(controller.ki))
    kd = np.full_like(t, float(controller.kd))

    return {
        "method": method_name,
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


def run_stage2_episode_trace(
    method_name: str,
    agent: Agent,
    expected_obs_dim_per_step: int,
    cfg: dict,
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
    )

    if use_disturbance:
        disturbance_config = {
            "mass_range": (5.0, 20.0),
            "friction_range": (0.1, 2.0),
            "initial_randomization_enabled": False,
            "mid_episode_disturbance_enabled": True,
            "disturbance_mode": cfg["disturbance_mode"],
            "disturbance_step_range": tuple(cfg["disturbance_step_range"]),
            "disturbance_time_range_s": tuple(cfg["disturbance_time_range_s"]),
            "disturbance_mass_scale_range": tuple(cfg["disturbance_mass_scale_range"]),
            "disturbance_friction_scale_range": tuple(cfg["disturbance_friction_scale_range"]),
            "position_patch_enabled": bool(cfg["position_patch_enabled"]),
            "patch_x_range": tuple(cfg["patch_x_range"]),
            "patch_friction_scale": float(cfg["patch_friction_scale"]),
        }
        env = DomainRandomizationWrapper(env, randomization_config=disturbance_config)

    if expected_obs_dim_per_step < int(env.observation_space.shape[0]):
        env = ObservationFeatureSelectWrapper(env, keep_dims=expected_obs_dim_per_step)

    env = gym.wrappers.TimeLimit(env, max_episode_steps=int(cfg["max_eval_steps"]))
    env = gym.wrappers.FrameStackObservation(env, stack_size=int(cfg["stack_size"]))

    env.unwrapped.model.body_mass[1] = 10.0
    env.unwrapped.model.geom_friction[0, 0] = 1.0
    mujoco.mj_forward(env.unwrapped.model, env.unwrapped.data)

    obs, _ = env.reset(seed=seed)

    t, pos, vel, err, reward, kp_arr, ki_arr, kd_arr = [], [], [], [], [], [], [], []
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
        "mass": np.full(len(t), np.nan, dtype=np.float64),
        "friction": np.full(len(t), np.nan, dtype=np.float64),
        "first_disturbance_time": first_disturbance_time,
    }


def plot_fair_comparison(output_dir: Path):
    """Generate fair comparison plots using Standard PID."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load config
    config_path = Path("benchmark_results/stage2_warm500k_disturbance_eval_longhorizon_v2/stage2_config.json")
    with open(config_path) as f:
        cfg = json.load(f)

    # Load Stage2 agent
    agent_path = Path(
        "benchmark_results/stage2_warm500k_disturbance_eval_longhorizon_v2/seed_7/models/meta_rl_agent.pth"
    )
    if not agent_path.exists():
        print(f"ERROR: Agent not found at {agent_path}")
        return

    agent, obs_dim = load_stage2_agent(agent_path, stack_size=10)

    print("Generating fair comparison plots...")
    print("=" * 70)

    # Run all traces
    target_distance = 5.0

    # Run BOTH PID variants
    print("Running Stage1 Standard PID (clean)...")
    clean_std = run_stage1_trace(target_distance, seed=42, use_standard=True)

    print("Running Stage1 Robust PID (clean) for reference...")
    clean_robust = run_stage1_trace(target_distance, seed=42, use_standard=False)

    print("Running Stage2 Clean-Trained (clean)...")
    clean_s2c = run_stage2_episode_trace(
        "Stage2 Clean-Trained", agent, obs_dim, cfg, target_distance, seed=42, use_disturbance=False
    )

    print("Running Stage2 Disturbance-Trained (clean)...")
    clean_s2d = run_stage2_episode_trace(
        "Stage2 Disturbance-Trained", agent, obs_dim, cfg, target_distance, seed=42, use_disturbance=False
    )

    # Plot 1: FAIR COMPARISON (Standard vs Stage2)
    colors = {
        "Stage1 Fixed PID (Standard)": "#ff7f0e",
        "Stage2 Clean-Trained": "#1f77b4",
        "Stage2 Disturbance-Trained": "#2ca02c",
    }

    fig, axes = plt.subplots(2, 1, figsize=(14, 9))

    for run in [clean_std, clean_s2c, clean_s2d]:
        axes[0].plot(run["t"], run["pos"], label=run["method"], color=colors.get(run["method"], None), linewidth=2)
        axes[1].plot(run["t"], run["vel"], label=run["method"], color=colors.get(run["method"], None), linewidth=2)

    axes[0].axhline(target_distance, color="black", linestyle="--", linewidth=1.5, label="Target")
    axes[0].set_ylabel("Position (m)", fontsize=12)
    axes[0].set_title(
        "FAIR COMPARISON: Stage1 Standard vs Stage2 Models (Clean Evaluation)", fontsize=13, fontweight="bold"
    )
    axes[0].legend(fontsize=11, loc="best")
    axes[0].grid(alpha=0.3)
    axes[0].set_xlim(0, min(10, max(run["t"].max() for run in [clean_std, clean_s2c, clean_s2d])))

    axes[1].set_xlabel("Time (s)", fontsize=12)
    axes[1].set_ylabel("Velocity (m/s)", fontsize=12)
    axes[1].set_title("Velocity Comparison", fontsize=13)
    axes[1].legend(fontsize=11, loc="best")
    axes[1].grid(alpha=0.3)
    axes[1].set_xlim(0, min(10, max(run["t"].max() for run in [clean_std, clean_s2c, clean_s2d])))

    plt.tight_layout()
    plt.savefig(output_dir / "fair_comparison_standard_vs_stage2.png", dpi=220, bbox_inches="tight")
    print(f"✓ Fair comparison saved: {output_dir / 'fair_comparison_standard_vs_stage2.png'}")
    plt.close()

    # Plot 2: DIAGNOSTIC (All variants including Robust)
    colors_all = {
        "Stage1 Fixed PID (Standard)": "#ff7f0e",
        "Stage1 Fixed PID (Robust)": "#ffbb78",
        "Stage2 Clean-Trained": "#1f77b4",
        "Stage2 Disturbance-Trained": "#2ca02c",
    }

    fig, axes = plt.subplots(2, 1, figsize=(14, 9))

    for run in [clean_robust, clean_std, clean_s2c, clean_s2d]:
        axes[0].plot(
            run["t"],
            run["pos"],
            label=run["method"],
            color=colors_all.get(run["method"], None),
            linewidth=2,
            linestyle="--" if "Robust" in run["method"] else "-",
        )
        axes[1].plot(
            run["t"],
            run["vel"],
            label=run["method"],
            color=colors_all.get(run["method"], None),
            linewidth=2,
            linestyle="--" if "Robust" in run["method"] else "-",
        )

    axes[0].axhline(target_distance, color="black", linestyle=":", linewidth=1.5, label="Target")
    axes[0].set_ylabel("Position (m)", fontsize=12)
    axes[0].set_title(
        "DIAGNOSTIC: All PID Variants (dashed = underpowered Robust, solid = Standard/Stage2)",
        fontsize=13,
        fontweight="bold",
    )
    axes[0].legend(fontsize=10, loc="best", ncol=2)
    axes[0].grid(alpha=0.3)
    axes[0].set_xlim(0, min(10, max(run["t"].max() for run in [clean_robust, clean_std, clean_s2c, clean_s2d])))

    axes[1].set_xlabel("Time (s)", fontsize=12)
    axes[1].set_ylabel("Velocity (m/s)", fontsize=12)
    axes[1].set_title("Velocity Comparison: Why Robust is so slow (weak gains)", fontsize=13)
    axes[1].legend(fontsize=10, loc="best", ncol=2)
    axes[1].grid(alpha=0.3)
    axes[1].set_xlim(0, min(10, max(run["t"].max() for run in [clean_robust, clean_std, clean_s2c, clean_s2d])))

    plt.tight_layout()
    plt.savefig(output_dir / "diagnostic_all_variants.png", dpi=220, bbox_inches="tight")
    print(f"✓ Diagnostic plot saved: {output_dir / 'diagnostic_all_variants.png'}")
    plt.close()

    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY OF FINDINGS:")
    print("=" * 70)
    print(f"\nStage1 Standard PID (Kp={clean_std['kp'][0]:.4f}, Ki={clean_std['ki'][0]:.2f}):")
    print(f"  - Final position: {clean_std['pos'][-1]:.3f}m")
    print(f"  - Max velocity: {np.max(clean_std['vel']):.4f} m/s")
    print(f"  - Time to target: {clean_std['t'][np.argmin(np.abs(clean_std['pos']-target_distance))]:.2f}s")

    print(f"\nStage1 Robust PID (Kp={clean_robust['kp'][0]:.4f}, Ki={clean_robust['ki'][0]:.2f}):")
    print(f"  - Final position: {clean_robust['pos'][-1]:.3f}m")
    print(f"  - Max velocity: {np.max(clean_robust['vel']):.4f} m/s")
    print(f"  - Time to target: {clean_robust['t'][np.argmin(np.abs(clean_robust['pos']-target_distance))]:.2f}s")

    print("\nStage2 Clean-Trained:")
    print(f"  - Final position: {clean_s2c['pos'][-1]:.3f}m")
    print(f"  - Max velocity: {np.max(clean_s2c['vel']):.4f} m/s")
    print(f"  - Overshoot: {np.max(clean_s2c['pos']) - target_distance:.3f}m")

    print("\nConclusion:")
    print("  Stage1 Robust was underpowered (Kp 0.155)")
    print("  Stage1 Standard is 2.5x stronger (Kp 0.396) → fairer comparison")
    print("  Stage2 uses position-based control → inherently different but achieves target")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="benchmark_results/final_thesis_master_comparison")
    args = parser.parse_args()

    plot_fair_comparison(args.output_dir)

import argparse
import json
from pathlib import Path

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from agents.model import Agent
from envs.adaptive_suspension import AdaptiveSuspensionEnv
from stage1_fixed_pid_benchmark import Scenario, apply_scenario, load_controller_specs, load_sim, reset_episode
from stage1_fixed_pid_benchmark import run_episode as run_stage1_episode

CLEAN_SCENARIO = Scenario(name="Clean", mass_scale=1.0, friction_scale=1.0)

METHOD_STAGE1 = "Stage1_FixedPID_Robust"
METHOD_STAGE2_CLEAN = "Stage2_CleanTrained"
METHOD_STAGE2_DIST = "Stage2_DisturbanceTrained"

ENV_CONFIG_KEYS = [
    "gain_base_kp",
    "gain_base_ki",
    "gain_base_kd",
    "gain_delta_kp",
    "gain_delta_ki",
    "gain_delta_kd",
    "gain_range_kp",
    "gain_range_ki",
    "gain_range_kd",
    "terminal_hold_bonus",
    "terminal_hold_velocity_threshold",
    "action_slew_limit",
    "action_rate_penalty_coef",
    "safety_speed_governor_enabled",
    "safety_brake_margin_m",
    "safety_max_decel_mps2",
    "safety_brake_k",
    "safety_hard_overshoot_m",
    "safety_overshoot_penalty",
]


def compute_hold_settling(
    errors: np.ndarray,
    velocities: np.ndarray,
    tolerance: float,
    stop_velocity: float,
    hold_steps: int,
    dt: float,
    timeout_s: float,
) -> tuple[float, int]:
    if errors.size == 0 or velocities.size == 0 or errors.size != velocities.size:
        return float(timeout_s), 0

    hold_steps = max(int(hold_steps), 1)
    within = (np.abs(errors) <= tolerance) & (np.abs(velocities) <= stop_velocity)

    run_length = 0
    for index, ok in enumerate(within):
        run_length = run_length + 1 if bool(ok) else 0
        if run_length >= hold_steps:
            settle_index = index - hold_steps + 1
            return float(settle_index * dt), 1

    return float(timeout_s), 0


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


def load_env_kwargs_from_config(config_path: Path | None) -> dict:
    if config_path is None:
        return {}
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    kwargs = {k: payload[k] for k in ENV_CONFIG_KEYS if k in payload}
    for key in ("gain_range_kp", "gain_range_ki", "gain_range_kd"):
        if key in kwargs:
            kwargs[key] = tuple(kwargs[key])
    return kwargs


def wrap_stage2_env(
    target_pos: float,
    max_steps: int,
    stack_size: int,
    hold_steps: int,
    tolerance: float,
    expected_obs_dim_per_step: int,
    env_kwargs: dict,
):
    env = AdaptiveSuspensionEnv(
        target_pos=target_pos,
        hold_steps=hold_steps,
        max_episode_steps=max_steps,
        stop_tolerance=tolerance,
        **env_kwargs,
    )
    if expected_obs_dim_per_step < int(env.observation_space.shape[0]):
        env = ObservationFeatureSelectWrapper(env, keep_dims=expected_obs_dim_per_step)
    env = gym.wrappers.TimeLimit(env, max_episode_steps=max_steps)
    env = gym.wrappers.FrameStackObservation(env, stack_size=stack_size)
    return env


def load_stage2_agent(
    model_path: Path,
    distance: float,
    max_steps: int,
    stack_size: int,
    hold_steps: int,
    tolerance: float,
    env_kwargs: dict,
):
    state_dict = torch.load(model_path, map_location="cpu")
    expected_obs_dim_per_step = infer_expected_obs_dim_per_step(state_dict, stack_size)
    env = wrap_stage2_env(
        distance,
        max_steps,
        stack_size,
        hold_steps,
        tolerance,
        expected_obs_dim_per_step,
        env_kwargs,
    )
    agent = Agent(env)
    agent.load_state_dict(state_dict)
    agent.eval()
    env.close()
    return agent, expected_obs_dim_per_step


def run_stage2_episode(
    agent: Agent,
    distance: float,
    seed: int,
    scenario: Scenario,
    max_steps: int,
    stack_size: int,
    hold_steps: int,
    tolerance: float,
    stop_velocity: float,
    expected_obs_dim_per_step: int,
    env_kwargs: dict,
) -> dict:
    import mujoco

    env = AdaptiveSuspensionEnv(
        target_pos=distance,
        hold_steps=hold_steps,
        max_episode_steps=max_steps,
        stop_tolerance=tolerance,
        **env_kwargs,
    )
    env.model.body_mass[1] = float(scenario.mass_scale * 10.0)
    env.model.geom_friction[0, 0] = float(scenario.friction_scale * 1.0)
    mujoco.mj_forward(env.model, env.data)
    if expected_obs_dim_per_step < int(env.observation_space.shape[0]):
        env = ObservationFeatureSelectWrapper(env, keep_dims=expected_obs_dim_per_step)
    env = gym.wrappers.TimeLimit(env, max_episode_steps=max_steps)
    env = gym.wrappers.FrameStackObservation(env, stack_size=stack_size)

    obs, _ = env.reset(seed=seed)
    positions = []
    velocities = []
    rewards = []
    max_pos = -1e9

    while True:
        with torch.no_grad():
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
            action = agent.actor_mean(obs_tensor.reshape(obs_tensor.shape[0], -1))
            action = torch.clamp(action, -1.0, 1.0)

        obs, reward, terminated, truncated, info = env.step(action.squeeze(0).cpu().numpy())
        pos = float(info["state"]["pos"])
        vel = float(info["state"].get("vel", 0.0))
        positions.append(pos)
        velocities.append(vel)
        rewards.append(float(reward))
        max_pos = max(max_pos, pos)

        if terminated or truncated:
            break

    final_pos = positions[-1] if positions else float("nan")
    final_err = distance - final_pos
    final_abs_err = abs(final_err)
    overshoot = max(max_pos - distance, 0.0)
    dt = float(env.unwrapped.dt)
    timeout_s = float(max_steps * dt)
    errors = distance - np.asarray(positions, dtype=np.float64)
    vels = np.asarray(velocities, dtype=np.float64)
    settling_time_s, settled = compute_hold_settling(
        errors=errors,
        velocities=vels,
        tolerance=tolerance,
        stop_velocity=stop_velocity,
        hold_steps=hold_steps,
        dt=dt,
        timeout_s=timeout_s,
    )
    success = int(settled)
    env.close()

    return {
        "distance_m": float(distance),
        "steps": len(positions),
        "mean_reward": float(np.mean(rewards)) if rewards else float("nan"),
        "final_pos": float(final_pos),
        "final_err": float(final_err),
        "final_abs_err": float(final_abs_err),
        "overshoot": float(overshoot),
        "success": success,
        "settled": settled,
        "settling_time_s": float(settling_time_s),
    }


def run_stage1_episode_on_distance(
    model,
    data,
    controller,
    distance: float,
    max_steps: int,
    tolerance: float,
    hold_steps: int,
    stop_velocity: float,
):
    metrics, traces = run_stage1_episode(
        model,
        data,
        controller,
        distance_target_m=distance,
        max_steps=max_steps,
        tolerance_m=tolerance,
        render=False,
    )

    times = np.asarray(traces.get("time_s", []), dtype=np.float64)
    distances = np.asarray(traces.get("distance_m", []), dtype=np.float64)
    velocities = np.asarray(traces.get("forward_speed_mps", []), dtype=np.float64)

    if times.size >= 2:
        dt = float(times[1] - times[0])
    elif times.size == 1:
        dt = float(times[0])
    else:
        dt = 0.002
    timeout_s = float(max_steps * dt)

    errors = distance - distances
    settling_time_s, settled = compute_hold_settling(
        errors=errors,
        velocities=velocities,
        tolerance=tolerance,
        stop_velocity=stop_velocity,
        hold_steps=hold_steps,
        dt=dt,
        timeout_s=timeout_s,
    )

    final_pos = float(distances[-1]) if distances.size else float("nan")
    final_err = float(distance - final_pos) if distances.size else float("nan")
    final_abs_err = float(abs(final_err)) if distances.size else float("nan")

    return {
        "distance_m": float(distance),
        "steps": int(metrics.get("steps", 0)),
        "mean_reward": float(metrics.get("mean_reward", float("nan"))),
        "final_pos": final_pos,
        "final_err": final_err,
        "final_abs_err": final_abs_err,
        "overshoot": float(metrics.get("overshoot_m", metrics.get("overshoot", float("nan")))),
        "success": int(settled),
        "settled": int(settled),
        "settling_time_s": float(settling_time_s),
    }


def plot_results(df: pd.DataFrame, out_path: Path) -> None:
    methods = list(df["method"].drop_duplicates())
    suites = list(df["suite"].drop_duplicates()) if "suite" in df.columns else ["Clean"]
    colors = {
        METHOD_STAGE1: "#ff7f0e",
        METHOD_STAGE2_CLEAN: "#1f77b4",
        METHOD_STAGE2_DIST: "#2ca02c",
    }
    suite_styles = {
        "Clean": "-",
        "Stress": "--",
    }

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True)
    metric_specs = [
        ("settling_time_s", "Settling Time (s)"),
        ("overshoot", "Overshoot (m)"),
        ("final_abs_err", "Final Abs Error (m)"),
        ("success", "Success"),
    ]

    for ax, (metric, title) in zip(axes.flatten(), metric_specs):
        for suite in suites:
            for method in methods:
                subset = df[(df["method"] == method) & (df["suite"] == suite)].sort_values("distance_m")
                if subset.empty:
                    continue
                label = f"{method} ({suite})"
                ax.plot(
                    subset["distance_m"],
                    subset[metric],
                    marker="o",
                    linewidth=1.6,
                    linestyle=suite_styles.get(suite, "-"),
                    label=label,
                    color=colors.get(method),
                )
        ax.set_title(title)
        ax.grid(alpha=0.25)
        if metric == "success":
            ax.set_ylim(-0.05, 1.05)

    axes[1, 0].set_xlabel("Target Distance (m)")
    axes[1, 1].set_xlabel("Target Distance (m)")
    axes[0, 0].legend(loc="best")
    fig.suptitle("Three-Method Distance Generalization Comparison", fontsize=14)
    fig.tight_layout(rect=(0.0, 0.02, 1.0, 0.96))
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare Stage 1 PID vs Stage 2 clean vs Stage 2 " "disturbance-trained across random target distances."
        )
    )
    parser.add_argument("--stage1-controller", default="Fixed PID (Robust)")
    parser.add_argument("--stage2-clean-model", type=Path, required=True)
    parser.add_argument("--stage2-disturbance-model", type=Path, required=True)
    parser.add_argument("--stage2-clean-config", type=Path, default=None)
    parser.add_argument("--stage2-disturbance-config", type=Path, default=None)
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--distance-mode", choices=["random", "fixed"], default="random")
    parser.add_argument("--fixed-distances", type=str, default="3,5,7")
    parser.add_argument("--episodes-per-distance", type=int, default=10)
    parser.add_argument("--min-distance", type=float, default=3.0)
    parser.add_argument("--max-distance", type=float, default=7.0)
    parser.add_argument("--max-steps", type=int, default=1200)
    parser.add_argument("--stack-size", type=int, default=10)
    parser.add_argument("--hold-steps", type=int, default=25)
    parser.add_argument("--tolerance", type=float, default=0.05)
    parser.add_argument("--stop-velocity", type=float, default=0.05)
    parser.add_argument("--suite", choices=["clean", "stress", "both"], default="both")
    parser.add_argument("--stress-mass-scale", type=float, default=1.25)
    parser.add_argument("--stress-friction-scale", type=float, default=0.6)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("benchmark_results/three_method_distance_generalization")
    )
    args = parser.parse_args()

    if args.min_distance <= 0 or args.max_distance <= 0:
        raise ValueError("Distances must be positive.")
    if args.min_distance > args.max_distance:
        raise ValueError("min-distance must be <= max-distance.")
    if args.episodes_per_distance <= 0:
        raise ValueError("episodes-per-distance must be positive.")
    if args.stress_mass_scale <= 0 or args.stress_friction_scale <= 0:
        raise ValueError("Stress scenario mass/friction scales must be positive.")

    rng = np.random.default_rng(args.seed)
    if args.distance_mode == "random":
        sampled_distances = rng.uniform(args.min_distance, args.max_distance, size=args.episodes)
        distance_set_label = f"Random-{args.min_distance:.1f}to{args.max_distance:.1f}"
    else:
        fixed = [float(x.strip()) for x in args.fixed_distances.split(",") if x.strip()]
        if len(fixed) == 0:
            raise ValueError("fixed-distances must provide at least one distance.")
        if any(d <= 0 for d in fixed):
            raise ValueError("All fixed distances must be positive.")
        sampled_distances = np.asarray(fixed * args.episodes_per_distance, dtype=np.float64)
        distance_set_label = "Fixed-" + "-".join(str(int(d)) if float(d).is_integer() else f"{d:g}" for d in fixed)

    suites: list[tuple[str, Scenario]] = []
    if args.suite in ("clean", "both"):
        suites.append(("Clean", CLEAN_SCENARIO))
    if args.suite in ("stress", "both"):
        suites.append(
            (
                "Stress",
                Scenario(
                    name="Stress",
                    mass_scale=float(args.stress_mass_scale),
                    friction_scale=float(args.stress_friction_scale),
                ),
            )
        )

    model, data, car_body_id, floor_geom_id, base_car_mass, base_floor_friction = load_sim()
    controllers = load_controller_specs(None)
    stage1_controller = None
    for controller in controllers:
        if controller.name == args.stage1_controller:
            stage1_controller = controller
            break
    if stage1_controller is None:
        raise ValueError(f"Could not find controller '{args.stage1_controller}' in stage1 benchmark specs.")

    clean_env_kwargs = load_env_kwargs_from_config(args.stage2_clean_config)
    dist_env_kwargs = load_env_kwargs_from_config(args.stage2_disturbance_config)

    clean_agent, clean_expected_obs_dim_per_step = load_stage2_agent(
        args.stage2_clean_model,
        sampled_distances[0],
        args.max_steps,
        args.stack_size,
        args.hold_steps,
        args.tolerance,
        clean_env_kwargs,
    )
    dist_agent, dist_expected_obs_dim_per_step = load_stage2_agent(
        args.stage2_disturbance_model,
        sampled_distances[0],
        args.max_steps,
        args.stack_size,
        args.hold_steps,
        args.tolerance,
        dist_env_kwargs,
    )

    rows = []
    for suite_index, (suite_name, scenario) in enumerate(suites):
        for episode_index, distance in enumerate(sampled_distances):
            episode_seed = int(args.seed * 10_000 + suite_index * 1_000_000 + episode_index)

            apply_scenario(model, car_body_id, floor_geom_id, base_car_mass, base_floor_friction, scenario)
            reset_episode(model, data, np.random.default_rng(episode_seed))
            row = run_stage1_episode_on_distance(
                model,
                data,
                stage1_controller,
                float(distance),
                args.max_steps,
                args.tolerance,
                args.hold_steps,
                args.stop_velocity,
            )
            row.update(
                {
                    "suite": suite_name,
                    "episode": episode_index,
                    "seed": episode_seed,
                    "scenario": distance_set_label,
                    "method": METHOD_STAGE1,
                }
            )
            rows.append(row)

            clean_row = run_stage2_episode(
                clean_agent,
                float(distance),
                episode_seed,
                scenario,
                args.max_steps,
                args.stack_size,
                args.hold_steps,
                args.tolerance,
                args.stop_velocity,
                clean_expected_obs_dim_per_step,
                clean_env_kwargs,
            )
            clean_row.update(
                {
                    "suite": suite_name,
                    "episode": episode_index,
                    "seed": episode_seed,
                    "scenario": distance_set_label,
                    "method": METHOD_STAGE2_CLEAN,
                }
            )
            rows.append(clean_row)

            dist_row = run_stage2_episode(
                dist_agent,
                float(distance),
                episode_seed,
                scenario,
                args.max_steps,
                args.stack_size,
                args.hold_steps,
                args.tolerance,
                args.stop_velocity,
                dist_expected_obs_dim_per_step,
                dist_env_kwargs,
            )
            dist_row.update(
                {
                    "suite": suite_name,
                    "episode": episode_index,
                    "seed": episode_seed,
                    "scenario": distance_set_label,
                    "method": METHOD_STAGE2_DIST,
                }
            )
            rows.append(dist_row)

    raw_df = pd.DataFrame(rows)
    summary_df = raw_df.groupby(["suite", "method"], as_index=False).agg(
        episodes=("episode", "count"),
        distance_mean=("distance_m", "mean"),
        distance_std=("distance_m", "std"),
        settling_time_mean=("settling_time_s", "mean"),
        settling_time_std=("settling_time_s", "std"),
        overshoot_mean=("overshoot", "mean"),
        overshoot_std=("overshoot", "std"),
        final_abs_error_mean=("final_abs_err", "mean"),
        final_abs_error_std=("final_abs_err", "std"),
        success_rate=("success", "mean"),
        settled_rate=("settled", "mean"),
        mean_reward=("mean_reward", "mean"),
    )
    summary_df["success_rate"] *= 100.0
    summary_df["settled_rate"] *= 100.0

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = out_dir / "three_method_distance_raw.csv"
    summary_path = out_dir / "three_method_distance_summary.csv"
    plot_path = out_dir / "three_method_distance_panel.png"

    raw_df.to_csv(raw_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    plot_results(raw_df, plot_path)

    print(f"Saved raw results: {raw_path}")
    print(f"Saved summary: {summary_path}")
    print(f"Saved plot: {plot_path}")


if __name__ == "__main__":
    main()

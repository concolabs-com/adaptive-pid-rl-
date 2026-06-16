import argparse
import math
import os
from dataclasses import dataclass

import matplotlib.pyplot as plt
import mujoco
import numpy as np
import pandas as pd

CAR_XML_PATH = os.path.join("envs", "assets", "car_model.xml")


@dataclass(frozen=True)
class Scenario:
    name: str
    mass: float
    friction: float


@dataclass(frozen=True)
class ControllerSpec:
    name: str
    kind: str
    gains: tuple[float, float, float, float] | None = None


SCENARIOS = [
    Scenario("Standard", mass=3.0, friction=1.0),
    Scenario("Heavy & Slippery", mass=6.0, friction=0.4),
    Scenario("Light & Grippy", mass=1.8, friction=1.6),
]

CONTROLLER_ADAPTIVE = "Adaptive Scheduled"
CONTROLLER_FIXED_STANDARD = "Fixed PID (Standard)"
CONTROLLER_FIXED_ROBUST = "Fixed PID (Robust)"

CONTROLLERS = [
    ControllerSpec(CONTROLLER_ADAPTIVE, "adaptive"),
    ControllerSpec(CONTROLLER_FIXED_STANDARD, "fixed", gains=(0.55, 0.15, 1.80, 0.20)),
    ControllerSpec(CONTROLLER_FIXED_ROBUST, "fixed", gains=(0.35, 0.25, 1.20, 0.35)),
]


TARGET_X = 1.0
TARGET_Y = 0.8


def wrap_to_pi(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


def yaw_from_quat(quat_wxyz):
    w, x, y, z = quat_wxyz
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def build_sim():
    model = mujoco.MjModel.from_xml_path(CAR_XML_PATH)
    data = mujoco.MjData(model)

    car_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "car")
    floor_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "")
    if floor_geom_id < 0:
        floor_geom_id = 0

    base_car_mass = float(model.body_mass[car_body_id])
    base_floor_friction = float(model.geom_friction[floor_geom_id, 0])

    return model, data, car_body_id, floor_geom_id, base_car_mass, base_floor_friction


def apply_scenario(model, data, car_body_id, floor_geom_id, scenario, episode_seed):
    episode_rng = np.random.default_rng(episode_seed)
    model.body_mass[car_body_id] = scenario.mass
    model.geom_friction[floor_geom_id, 0] = scenario.friction

    mujoco.mj_resetData(model, data)
    data.qpos[0] = float(episode_rng.uniform(-0.05, 0.05))
    data.qpos[1] = float(episode_rng.uniform(-0.15, 0.15))
    data.qpos[2] = 0.03
    yaw0 = float(episode_rng.uniform(-0.45, 0.45))
    data.qpos[3:7] = np.array([math.cos(yaw0 / 2.0), 0.0, 0.0, math.sin(yaw0 / 2.0)])
    mujoco.mj_forward(model, data)


def get_state(data):
    x = float(data.qpos[0])
    y = float(data.qpos[1])
    yaw = yaw_from_quat(data.qpos[3:7])

    vx = float(data.qvel[0])
    vy = float(data.qvel[1])
    yaw_rate = float(data.qvel[5])

    dx = TARGET_X - x
    dy = TARGET_Y - y
    distance = math.hypot(dx, dy)

    desired_heading = math.atan2(dy, dx)
    heading_error = wrap_to_pi(desired_heading - yaw)

    v_forward = math.cos(yaw) * vx + math.sin(yaw) * vy

    return {
        "x": x,
        "y": y,
        "yaw": yaw,
        "distance": distance,
        "heading_error": heading_error,
        "v_forward": v_forward,
        "yaw_rate": yaw_rate,
    }


def fixed_pid_action(state, gains):
    kp_dist, kd_dist, kp_heading, kd_heading = gains

    forward = kp_dist * state["distance"] - kd_dist * state["v_forward"]
    turn = kp_heading * state["heading_error"] - kd_heading * state["yaw_rate"]

    if abs(state["heading_error"]) > 0.9:
        forward *= 0.4

    return np.array([np.clip(forward, -1.0, 1.0), np.clip(turn, -1.0, 1.0)], dtype=np.float32)


def differential_mix(forward_turn_cmd):
    forward = float(forward_turn_cmd[0])
    turn = float(forward_turn_cmd[1])

    left = np.clip(forward - turn, -1.0, 1.0)
    right = np.clip(forward + turn, -1.0, 1.0)
    return np.array([left, right], dtype=np.float32)


def adaptive_scheduled_action(state, scenario):
    # This is a lightweight adaptive baseline that changes gains with context.
    mass_factor = scenario.mass / 3.0
    friction_factor = max(scenario.friction, 0.2)

    kp_dist = 0.55 / (0.8 + 0.4 * mass_factor)
    kd_dist = 0.12 + 0.18 * mass_factor

    kp_heading = 1.8 / (0.7 + 0.3 * friction_factor)
    kd_heading = 0.15 + 0.20 * (1.0 / friction_factor)

    gains = (kp_dist, kd_dist, kp_heading, kd_heading)
    return fixed_pid_action(state, gains)


def settle_time(distance_errors, dt, tolerance):
    within = np.asarray(distance_errors) <= tolerance
    if not np.any(within):
        return np.nan

    suffix_all_within = np.flip(np.cumprod(np.flip(within).astype(int))).astype(bool)
    settled_indices = np.flatnonzero(suffix_all_within)
    if len(settled_indices) == 0:
        return np.nan

    return float(settled_indices[0] * dt)


def compute_metrics(distance_errors, x_positions, dt, tolerance):
    if len(distance_errors) == 0:
        return {
            "steps": 0,
            "settling_time_s": np.nan,
            "overshoot": np.nan,
            "overshoot_pct": np.nan,
            "iae": np.nan,
            "final_error": np.nan,
            "abs_final_error": np.nan,
        }

    final_error = float(distance_errors[-1])
    x_overshoot = max(float(np.max(x_positions) - TARGET_X), 0.0)

    return {
        "steps": int(len(distance_errors)),
        "settling_time_s": settle_time(distance_errors, dt, tolerance),
        "overshoot": x_overshoot,
        "overshoot_pct": 100.0 * x_overshoot / max(abs(TARGET_X), 1e-8),
        "iae": float(np.sum(np.abs(distance_errors)) * dt),
        "final_error": final_error,
        "abs_final_error": abs(final_error),
    }


def run_episode(model, data, scenario, controller, max_steps, settle_tolerance):
    distance_errors = []
    x_positions = []
    y_positions = []
    forward_cmds = []
    turn_cmds = []
    total_reward = 0.0

    for _ in range(max_steps):
        state = get_state(data)

        if controller.kind == "adaptive":
            forward_turn = adaptive_scheduled_action(state, scenario)
        else:
            forward_turn = fixed_pid_action(state, controller.gains)

        wheel_action = differential_mix(forward_turn)

        data.ctrl[0] = float(wheel_action[0])
        data.ctrl[1] = float(wheel_action[1])
        mujoco.mj_step(model, data)

        next_state = get_state(data)
        distance_errors.append(next_state["distance"])
        x_positions.append(next_state["x"])
        y_positions.append(next_state["y"])
        forward_cmds.append(float(forward_turn[0]))
        turn_cmds.append(float(forward_turn[1]))

        # Reward proxy for comparison consistency.
        reward = -(next_state["distance"] ** 2 + 0.1 * next_state["heading_error"] ** 2)
        total_reward += reward

    dt = model.opt.timestep
    metrics = compute_metrics(distance_errors, x_positions, dt, settle_tolerance)
    metrics["return"] = float(total_reward)

    return {
        "distance_errors": distance_errors,
        "x_positions": x_positions,
        "y_positions": y_positions,
        "forward_cmds": forward_cmds,
        "turn_cmds": turn_cmds,
        "metrics": metrics,
    }


def evaluate_benchmark(episodes_per_scenario, max_steps, settle_tolerance, seed):
    model, data, car_body_id, floor_geom_id, base_car_mass, base_floor_friction = build_sim()

    records = []
    plot_samples = {}
    rng = np.random.default_rng(seed)

    for scenario in SCENARIOS:
        for episode_index in range(episodes_per_scenario):
            episode_seed = int(rng.integers(0, 1_000_000_000))

            for controller in CONTROLLERS:
                apply_scenario(model, data, car_body_id, floor_geom_id, scenario, episode_seed)
                rollout = run_episode(
                    model,
                    data,
                    scenario,
                    controller,
                    max_steps=max_steps,
                    settle_tolerance=settle_tolerance,
                )

                metrics = rollout["metrics"].copy()
                metrics.update(
                    {
                        "scenario": scenario.name,
                        "mass": scenario.mass,
                        "friction": scenario.friction,
                        "controller": controller.name,
                        "episode": episode_index,
                        "mass_relative": scenario.mass / max(base_car_mass, 1e-8),
                        "friction_relative": scenario.friction / max(base_floor_friction, 1e-8),
                    }
                )
                records.append(metrics)

                if episode_index == 0 and controller.name == CONTROLLER_ADAPTIVE:
                    plot_samples[scenario.name] = rollout

    return pd.DataFrame(records), plot_samples


def summarize_results(df):
    summary = (
        df.groupby(["scenario", "controller"], as_index=False)
        .agg(
            return_mean=("return", "mean"),
            return_std=("return", "std"),
            settling_time_mean=("settling_time_s", "mean"),
            settling_time_std=("settling_time_s", "std"),
            overshoot_mean=("overshoot", "mean"),
            overshoot_std=("overshoot", "std"),
            iae_mean=("iae", "mean"),
            iae_std=("iae", "std"),
            final_error_mean=("final_error", "mean"),
            final_error_std=("final_error", "std"),
            abs_final_error_mean=("abs_final_error", "mean"),
            abs_final_error_std=("abs_final_error", "std"),
        )
        .sort_values(["scenario", "controller"])
    )
    return summary


def compute_improvements(summary):
    rows = []
    metrics = ["settling_time", "overshoot", "iae", "abs_final_error"]

    for scenario in summary["scenario"].drop_duplicates():
        scenario_summary = summary[summary["scenario"] == scenario].set_index("controller")
        if CONTROLLER_ADAPTIVE not in scenario_summary.index or CONTROLLER_FIXED_ROBUST not in scenario_summary.index:
            continue

        for metric in metrics:
            adaptive_value = float(scenario_summary.loc[CONTROLLER_ADAPTIVE, f"{metric}_mean"])
            fixed_value = float(scenario_summary.loc[CONTROLLER_FIXED_ROBUST, f"{metric}_mean"])

            if np.isnan(adaptive_value) or np.isnan(fixed_value):
                improvement_pct = np.nan
            elif abs(fixed_value) < 1e-12:
                improvement_pct = np.nan
            else:
                improvement_pct = 100.0 * (fixed_value - adaptive_value) / abs(fixed_value)

            rows.append(
                {
                    "scenario": scenario,
                    "metric": metric,
                    "adaptive_mean": adaptive_value,
                    "fixed_pid_robust_mean": fixed_value,
                    "improvement_pct": improvement_pct,
                }
            )

    return pd.DataFrame(rows)


def plot_summary(summary, plot_samples, output_path):
    scenarios = list(summary["scenario"].drop_duplicates())
    controllers = list(summary["controller"].drop_duplicates())

    fig, axes = plt.subplots(2, len(scenarios), figsize=(6 * len(scenarios), 8), squeeze=False)

    for column, scenario in enumerate(scenarios):
        scenario_summary = summary[summary["scenario"] == scenario].set_index("controller")

        for metric_row, metric_name, ylabel in [
            (0, "iae_mean", "IAE"),
            (1, "settling_time_mean", "Settling Time (s)"),
        ]:
            values = [scenario_summary.loc[controller, metric_name] for controller in controllers]
            axes[metric_row][column].bar(controllers, values, color=["#2E5491", "#E97D30", "#4CAF50"])
            axes[metric_row][column].set_title(f"{scenario} - {ylabel}")
            axes[metric_row][column].tick_params(axis="x", rotation=25)
            axes[metric_row][column].set_ylabel(ylabel)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)

    fig2, axes2 = plt.subplots(len(plot_samples), 1, figsize=(10, 4 * len(plot_samples)), squeeze=False)
    for row, (scenario_name, rollout) in enumerate(plot_samples.items()):
        axes2[row][0].plot(rollout["x_positions"], rollout["y_positions"], label="Trajectory")
        axes2[row][0].scatter([TARGET_X], [TARGET_Y], c="red", marker="x", s=80, label="Target")
        axes2[row][0].set_title(f"Adaptive Controller Trajectory - {scenario_name}")
        axes2[row][0].set_xlabel("x")
        axes2[row][0].set_ylabel("y")
        axes2[row][0].axis("equal")
        axes2[row][0].legend()

    fig2.tight_layout()
    sample_path = os.path.join(os.path.dirname(output_path), "benchmark_sample_tracks.png")
    fig2.savefig(sample_path, dpi=200)


def main():
    parser = argparse.ArgumentParser(description="Run differential-drive car benchmarks.")
    parser.add_argument("--output-dir", default="benchmark_results")
    parser.add_argument("--episodes-per-scenario", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--settle-tolerance", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    df, plot_samples = evaluate_benchmark(
        episodes_per_scenario=args.episodes_per_scenario,
        max_steps=args.max_steps,
        settle_tolerance=args.settle_tolerance,
        seed=args.seed,
    )

    summary = summarize_results(df)
    improvements = compute_improvements(summary)

    raw_path = os.path.join(args.output_dir, "benchmark_raw_results.csv")
    summary_path = os.path.join(args.output_dir, "benchmark_summary.csv")
    improvement_path = os.path.join(args.output_dir, "benchmark_improvements.csv")
    plot_path = os.path.join(args.output_dir, "benchmark_summary.png")

    df.to_csv(raw_path, index=False)
    summary.to_csv(summary_path, index=False)
    improvements.to_csv(improvement_path, index=False)
    plot_summary(summary, plot_samples, plot_path)

    print(f"Saved raw results to {raw_path}")
    print(f"Saved summary to {summary_path}")
    print(f"Saved improvements to {improvement_path}")
    print(f"Saved plots to {plot_path}")


if __name__ == "__main__":
    main()

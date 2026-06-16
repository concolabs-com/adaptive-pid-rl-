import argparse
import json
import math
import os
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass

import matplotlib.pyplot as plt
import mujoco
import mujoco.viewer
import numpy as np
import pandas as pd

from utils.pid import PIDController

CAR_XML_PATH = os.path.join("envs", "assets", "car_model.xml")

# Stage 1 requires two fixed PID baselines.
CONTROLLER_FIXED_STANDARD = "Fixed PID (Standard)"
CONTROLLER_FIXED_ROBUST = "Fixed PID (Robust)"


@dataclass(frozen=True)
class Scenario:
    name: str
    mass_scale: float
    friction_scale: float


@dataclass(frozen=True)
class ControllerSpec:
    name: str
    kp: float
    ki: float
    kd: float
    target_speed: float


SCENARIOS = [
    Scenario(name="Standard", mass_scale=1.00, friction_scale=1.00),
    Scenario(name="Heavy and Slippery", mass_scale=1.50, friction_scale=0.60),
    Scenario(name="Light and Grippy", mass_scale=0.70, friction_scale=1.40),
]

DEFAULT_CONTROLLERS = [
    ControllerSpec(name=CONTROLLER_FIXED_STANDARD, kp=1.8, ki=0.7, kd=0.0, target_speed=0.55),
    ControllerSpec(name=CONTROLLER_FIXED_ROBUST, kp=0.9, ki=0.18, kd=0.0, target_speed=0.55),
]

DECEL_ZONE_M = 0.60
STOP_SPEED_MPS = 0.05


def load_controller_specs(csv_path: str | None) -> list[ControllerSpec]:
    if not csv_path:
        return list(DEFAULT_CONTROLLERS)

    df = pd.read_csv(csv_path)
    required = {"controller", "kp", "ki", "kd"}
    missing = required.difference(df.columns)
    if missing:
        missing_text = ", ".join(sorted(missing))
        raise ValueError(f"Controller CSV is missing required columns: {missing_text}")

    if "target_speed" not in df.columns:
        df["target_speed"] = 0.55

    specs = []
    for _, row in df.iterrows():
        specs.append(
            ControllerSpec(
                name=str(row["controller"]),
                kp=float(row["kp"]),
                ki=float(row["ki"]),
                kd=float(row["kd"]),
                target_speed=float(row["target_speed"]),
            )
        )

    # Keep the expected Stage 1 ordering if both standard and robust are present.
    name_to_spec = {spec.name: spec for spec in specs}
    ordered = []
    for name in [CONTROLLER_FIXED_STANDARD, CONTROLLER_FIXED_ROBUST]:
        if name in name_to_spec:
            ordered.append(name_to_spec[name])
    for spec in specs:
        if spec.name not in {s.name for s in ordered}:
            ordered.append(spec)
    return ordered


def yaw_from_quat(quat_wxyz: np.ndarray) -> float:
    w, x, y, z = quat_wxyz
    return float(math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


def load_sim():
    model = mujoco.MjModel.from_xml_path(CAR_XML_PATH)
    data = mujoco.MjData(model)

    car_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "car")
    if car_body_id < 0:
        car_body_id = 1

    floor_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    if floor_geom_id < 0:
        floor_geom_id = 0

    base_car_mass = float(model.body_mass[car_body_id])
    base_floor_friction = float(model.geom_friction[floor_geom_id, 0])

    return model, data, car_body_id, floor_geom_id, base_car_mass, base_floor_friction


def apply_scenario(
    model, car_body_id: int, floor_geom_id: int, base_car_mass: float, base_floor_friction: float, scenario: Scenario
) -> None:
    model.body_mass[car_body_id] = base_car_mass * scenario.mass_scale
    model.geom_friction[floor_geom_id, 0] = base_floor_friction * scenario.friction_scale


def reset_episode(model, data, episode_rng: np.random.Generator) -> np.ndarray:
    mujoco.mj_resetData(model, data)

    # Keep starts nearly identical while still having realistic variation.
    data.qpos[0] = float(episode_rng.uniform(-0.02, 0.02))
    data.qpos[1] = float(episode_rng.uniform(-0.03, 0.03))
    data.qpos[2] = 0.03

    yaw0 = float(episode_rng.uniform(-0.05, 0.05))
    data.qpos[3:7] = np.array([math.cos(yaw0 / 2.0), 0.0, 0.0, math.sin(yaw0 / 2.0)])

    data.qvel[:] = 0.0
    data.ctrl[:] = 0.0
    mujoco.mj_forward(model, data)

    heading = np.array([math.cos(yaw0), math.sin(yaw0)], dtype=np.float64)
    return heading


def get_forward_speed(data, heading: np.ndarray) -> float:
    vx = float(data.qvel[0])
    vy = float(data.qvel[1])
    return float(vx * heading[0] + vy * heading[1])


def get_distance_along_heading(data, heading: np.ndarray) -> float:
    x = float(data.qpos[0])
    y = float(data.qpos[1])
    return float(x * heading[0] + y * heading[1])


def speed_setpoint(distance_remaining: float, cruise_speed: float) -> float:
    if distance_remaining <= 0.0:
        return 0.0
    if distance_remaining >= DECEL_ZONE_M:
        return cruise_speed
    return float(cruise_speed * (distance_remaining / DECEL_ZONE_M))


def detect_drive_signs(model, data, heading: np.ndarray) -> tuple[float, float]:
    start_qpos = data.qpos.copy()
    start_qvel = data.qvel.copy()

    def trial(left_sign: float, right_sign: float) -> float:
        data.qpos[:] = start_qpos
        data.qvel[:] = start_qvel
        mujoco.mj_forward(model, data)

        for _ in range(25):
            data.ctrl[0] = left_sign * 0.2
            data.ctrl[1] = right_sign * 0.2
            mujoco.mj_step(model, data)
        return get_distance_along_heading(data, heading)

    candidates = [(+1.0, +1.0), (-1.0, -1.0), (+1.0, -1.0), (-1.0, +1.0)]
    best_pair = candidates[0]
    best_progress = -np.inf

    for left_sign, right_sign in candidates:
        progress = trial(left_sign, right_sign)
        if progress > best_progress:
            best_progress = progress
            best_pair = (left_sign, right_sign)

    data.qpos[:] = start_qpos
    data.qvel[:] = start_qvel
    mujoco.mj_forward(model, data)
    return best_pair


def settling_time(errors: np.ndarray, dt: float, tolerance_m: float, timeout_s: float) -> tuple[float, int]:
    within = np.abs(errors) <= tolerance_m
    if not np.any(within):
        return float(timeout_s), 0

    suffix_all_within = np.flip(np.cumprod(np.flip(within).astype(int))).astype(bool)
    settled_indices = np.flatnonzero(suffix_all_within)
    if len(settled_indices) == 0:
        return float(timeout_s), 0

    return float(settled_indices[0] * dt), 1


def run_episode(
    model,
    data,
    controller: ControllerSpec,
    distance_target_m: float,
    max_steps: int,
    tolerance_m: float,
    render: bool = False,
):
    heading = np.array([1.0, 0.0], dtype=np.float64)
    heading[:] = heading / (np.linalg.norm(heading) + 1e-12)
    heading = heading.astype(np.float64)

    # Heading from current quaternion is more accurate than assuming +x.
    yaw0 = yaw_from_quat(data.qpos[3:7])
    heading = np.array([math.cos(yaw0), math.sin(yaw0)], dtype=np.float64)

    dt = float(model.opt.timestep)
    left_sign, right_sign = detect_drive_signs(model, data, heading)

    pid = PIDController(
        kp=controller.kp,
        ki=controller.ki,
        kd=controller.kd,
        setpoint=controller.target_speed,
        output_limits=(-0.35, 1.0),
    )

    times = []
    distances = []
    speed_errors = []
    forward_speeds = []

    success = False

    viewer_context = mujoco.viewer.launch_passive(model, data) if render else nullcontext(None)
    with viewer_context as viewer:
        for step in range(max_steps):
            dist = get_distance_along_heading(data, heading)
            distance_remaining = distance_target_m - dist

            pid.setpoint = speed_setpoint(max(distance_remaining, 0.0), controller.target_speed)
            speed_forward = get_forward_speed(data, heading)

            control, _ = pid.update(speed_forward, dt)
            control = float(np.clip(control, -0.35, 1.0))

            data.ctrl[0] = left_sign * control
            data.ctrl[1] = right_sign * control
            mujoco.mj_step(model, data)

            if viewer is not None:
                if not viewer.is_running():
                    break
                viewer.sync()
                time.sleep(dt)

            t = (step + 1) * dt
            dist_now = get_distance_along_heading(data, heading)
            err = distance_target_m - dist_now

            times.append(t)
            distances.append(dist_now)
            speed_errors.append(err)
            forward_speeds.append(speed_forward)

            if abs(err) <= tolerance_m:
                success = True

            # Stop early once close enough and almost stopped.
            if dist_now >= distance_target_m and abs(speed_forward) <= STOP_SPEED_MPS:
                break

    errors = np.asarray(speed_errors, dtype=np.float64)
    distances_arr = np.asarray(distances, dtype=np.float64)

    if len(errors) == 0:
        return {
            "settling_time_s": float("nan"),
            "overshoot_m": float("nan"),
            "iae": float("nan"),
            "final_abs_error_m": float("nan"),
            "success": 0,
            "steps": 0,
        }, {
            "time_s": times,
            "distance_m": distances,
            "forward_speed_mps": forward_speeds,
        }

    timeout_s = max_steps * dt
    settle_time_s, settled = settling_time(errors, dt=dt, tolerance_m=tolerance_m, timeout_s=timeout_s)

    overshoot_m = max(float(np.max(distances_arr) - distance_target_m), 0.0)
    iae = float(np.sum(np.abs(errors)) * dt)
    final_abs_error_m = float(abs(errors[-1]))

    metrics = {
        "settling_time_s": settle_time_s,
        "settled": settled,
        "overshoot_m": overshoot_m,
        "iae": iae,
        "final_abs_error_m": final_abs_error_m,
        "success": int(success),
        "steps": int(len(errors)),
    }

    traces = {
        "time_s": times,
        "distance_m": distances,
        "forward_speed_mps": forward_speeds,
    }
    return metrics, traces


def aggregate_results(df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        df.groupby(["scenario", "controller"], as_index=False)
        .agg(
            episodes=("episode", "count"),
            settling_time_mean=("settling_time_s", "mean"),
            settling_time_std=("settling_time_s", "std"),
            settled_rate=("settled", "mean"),
            overshoot_mean=("overshoot_m", "mean"),
            overshoot_std=("overshoot_m", "std"),
            iae_mean=("iae", "mean"),
            iae_std=("iae", "std"),
            final_abs_error_mean=("final_abs_error_m", "mean"),
            final_abs_error_std=("final_abs_error_m", "std"),
            success_rate=("success", "mean"),
        )
        .sort_values(["scenario", "controller"])
    )
    summary["settled_rate"] = 100.0 * summary["settled_rate"]
    summary["success_rate"] = 100.0 * summary["success_rate"]
    return summary


def plot_stage1_summary(summary: pd.DataFrame, output_path: str) -> None:
    scenarios = list(summary["scenario"].drop_duplicates())
    controllers = list(summary["controller"].drop_duplicates())

    fig, axes = plt.subplots(3, len(scenarios), figsize=(6 * len(scenarios), 11), squeeze=False)

    metric_specs = [
        ("iae_mean", "IAE"),
        ("settling_time_mean", "Settling Time (s)"),
        ("success_rate", "Success Rate (%)"),
    ]

    colors = ["#1f77b4", "#ff7f0e"]

    for col, scenario in enumerate(scenarios):
        subset = summary[summary["scenario"] == scenario].set_index("controller")
        for row, (metric_key, ylabel) in enumerate(metric_specs):
            values = [float(subset.loc[c, metric_key]) for c in controllers]
            axes[row][col].bar(controllers, values, color=colors[: len(controllers)])
            axes[row][col].set_title(f"{scenario} - {ylabel}")
            axes[row][col].set_ylabel(ylabel)
            axes[row][col].tick_params(axis="x", rotation=18)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 1 benchmark: fixed PID baselines across thesis scenarios.")
    parser.add_argument("--episodes-per-scenario", type=int, default=30)
    parser.add_argument("--distance-target-m", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=4000)
    parser.add_argument("--tolerance-m", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="benchmark_results")
    parser.add_argument(
        "--controller-gains-csv", default=None, help="Optional CSV with controller,kp,ki,kd[,target_speed]."
    )
    parser.add_argument(
        "--render", action="store_true", help="Render first episode for each controller in Standard scenario."
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    model, data, car_body_id, floor_geom_id, base_car_mass, base_floor_friction = load_sim()
    controllers = load_controller_specs(args.controller_gains_csv)

    rng = np.random.default_rng(args.seed)
    records = []

    for scenario in SCENARIOS:
        for episode in range(args.episodes_per_scenario):
            # Same initial condition seed is reused across controllers for fair comparison.
            episode_seed = int(rng.integers(0, 1_000_000_000))

            for controller in controllers:
                apply_scenario(
                    model,
                    car_body_id,
                    floor_geom_id,
                    base_car_mass,
                    base_floor_friction,
                    scenario,
                )
                reset_episode(model, data, np.random.default_rng(episode_seed))

                render_this = bool(args.render and scenario.name == "Standard" and episode == 0)
                metrics, _ = run_episode(
                    model,
                    data,
                    controller,
                    distance_target_m=args.distance_target_m,
                    max_steps=args.max_steps,
                    tolerance_m=args.tolerance_m,
                    render=render_this,
                )

                row = {
                    "scenario": scenario.name,
                    "mass_scale": scenario.mass_scale,
                    "friction_scale": scenario.friction_scale,
                    "controller": controller.name,
                    "kp": controller.kp,
                    "ki": controller.ki,
                    "kd": controller.kd,
                    "target_speed": controller.target_speed,
                    "episode": episode,
                    "seed": episode_seed,
                }
                row.update(metrics)
                records.append(row)

    raw_df = pd.DataFrame(records)
    summary_df = aggregate_results(raw_df)

    raw_path = os.path.join(args.output_dir, "stage1_fixed_pid_raw.csv")
    summary_path = os.path.join(args.output_dir, "stage1_fixed_pid_summary.csv")
    plot_path = os.path.join(args.output_dir, "stage1_fixed_pid_summary.png")
    config_path = os.path.join(args.output_dir, "stage1_fixed_pid_config.json")

    raw_df.to_csv(raw_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    plot_stage1_summary(summary_df, plot_path)

    config = {
        "script": os.path.basename(__file__),
        "distance_target_m": args.distance_target_m,
        "max_steps": args.max_steps,
        "tolerance_m": args.tolerance_m,
        "episodes_per_scenario": args.episodes_per_scenario,
        "seed": args.seed,
        "scenarios": [asdict(s) for s in SCENARIOS],
        "controllers": [asdict(c) for c in controllers],
        "controller_gains_csv": args.controller_gains_csv,
    }
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    print("Stage 1 benchmark completed.")
    print(f"Saved raw per-episode CSV: {raw_path}")
    print(f"Saved aggregated CSV: {summary_path}")
    print(f"Saved summary plot: {plot_path}")
    print(f"Saved run config: {config_path}")


if __name__ == "__main__":
    main()

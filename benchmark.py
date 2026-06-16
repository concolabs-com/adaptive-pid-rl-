import argparse
import math
import os
import time
from contextlib import nullcontext
from dataclasses import dataclass

import matplotlib.pyplot as plt
import mujoco
import mujoco.viewer
import numpy as np
import pandas as pd

from utils.pid import PIDController

CAR_XML_PATH = os.path.join("envs", "assets", "car_model.xml")


@dataclass(frozen=True)
class ControllerSpec:
    name: str
    kp: float
    ki: float
    kd: float
    target_speed: float


CONTROLLERS = [
    ControllerSpec("PID Conservative", kp=0.9, ki=0.18, kd=0.0, target_speed=0.40),
    ControllerSpec("PID Aggressive", kp=1.8, ki=0.7, kd=0.0, target_speed=0.60),
]

DECEL_ZONE_M = 0.60
STOP_TOLERANCE_M = 0.03
STOP_SPEED_MPS = 0.05


def yaw_from_quat(quat_wxyz):
    w, x, y, z = quat_wxyz
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def build_sim():
    model = mujoco.MjModel.from_xml_path(CAR_XML_PATH)
    data = mujoco.MjData(model)
    return model, data


def reset_straight_start(model, data):
    mujoco.mj_resetData(model, data)

    # Start exactly at the origin with neutral heading.
    data.qpos[0] = 0.0
    data.qpos[1] = 0.0
    data.qpos[2] = 0.03
    data.qpos[3:7] = np.array([1.0, 0.0, 0.0, 0.0])
    mujoco.mj_forward(model, data)

    start_yaw = yaw_from_quat(data.qpos[3:7])
    heading = np.array([math.cos(start_yaw), math.sin(start_yaw)], dtype=np.float64)
    return heading


def get_forward_speed(data, heading):
    vx = float(data.qvel[0])
    vy = float(data.qvel[1])
    return float(vx * heading[0] + vy * heading[1])


def get_distance_along_heading(data, heading):
    x = float(data.qpos[0])
    y = float(data.qpos[1])
    return float(x * heading[0] + y * heading[1])


def speed_setpoint(distance_remaining, cruise_speed):
    # Linear deceleration profile close to the target.
    if distance_remaining <= 0.0:
        return 0.0
    if distance_remaining >= DECEL_ZONE_M:
        return cruise_speed
    return cruise_speed * (distance_remaining / DECEL_ZONE_M)


def detect_drive_signs(model, data, heading):
    # Probe wheel sign combinations and pick the one with best forward progress.
    start_qpos = data.qpos.copy()
    start_qvel = data.qvel.copy()

    def trial(left_sign, right_sign):
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

    # Restore initial state before real rollout.
    data.qpos[:] = start_qpos
    data.qvel[:] = start_qvel
    mujoco.mj_forward(model, data)

    return best_pair


def run_episode(model, data, controller, distance_target_m, max_steps, render=False):
    heading = reset_straight_start(model, data)
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
    speeds = []
    controls = []

    reached_time = np.nan
    reached_step = -1
    first_arrival_time = np.nan
    first_arrival_step = -1

    viewer_context = mujoco.viewer.launch_passive(model, data) if render else nullcontext(None)
    with viewer_context as viewer:
        for step in range(max_steps):
            distance_now = get_distance_along_heading(data, heading)
            distance_remaining = max(distance_target_m - distance_now, 0.0)
            pid.setpoint = speed_setpoint(distance_remaining, controller.target_speed)

            speed_forward = get_forward_speed(data, heading)
            control, _ = pid.update(speed_forward, dt)
            control = float(np.clip(control, -0.35, 1.0))

            # Straight drive: equal command to left and right motors.
            data.ctrl[0] = left_sign * control
            data.ctrl[1] = right_sign * control
            mujoco.mj_step(model, data)

            if viewer is not None:
                if not viewer.is_running():
                    break
                viewer.sync()
                time.sleep(dt)

            t = (step + 1) * dt
            dist = get_distance_along_heading(data, heading)

            times.append(t)
            distances.append(dist)
            speeds.append(speed_forward)
            controls.append(control)

            if np.isnan(first_arrival_time) and dist >= distance_target_m:
                first_arrival_time = t
                first_arrival_step = step + 1

            if dist >= distance_target_m and abs(speed_forward) <= STOP_SPEED_MPS:
                reached_time = t
                reached_step = step + 1
                break

            if abs(distance_target_m - dist) <= STOP_TOLERANCE_M and abs(speed_forward) <= STOP_SPEED_MPS:
                reached_time = t
                reached_step = step + 1
                break

    end_dist = distances[-1] if distances else 0.0
    avg_speed = np.mean(speeds) if speeds else 0.0

    metrics = {
        "controller": controller.name,
        "target_speed": controller.target_speed,
        "kp": controller.kp,
        "ki": controller.ki,
        "kd": controller.kd,
        "distance_target_m": distance_target_m,
        "reached": int(not np.isnan(reached_time)),
        "time_to_target_s": reached_time,
        "steps_to_target": reached_step,
        "time_to_first_arrival_s": first_arrival_time,
        "steps_to_first_arrival": first_arrival_step,
        "final_distance_m": float(end_dist),
        "avg_forward_speed_mps": float(avg_speed),
    }

    traces = {
        "time_s": times,
        "distance_m": distances,
        "forward_speed_mps": speeds,
        "control": controls,
    }
    return metrics, traces


def plot_traces(all_traces, distance_target_m, output_path):
    fig, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=True)

    for name, traces in all_traces.items():
        axes[0].plot(traces["time_s"], traces["distance_m"], label=name)
        axes[1].plot(traces["time_s"], traces["forward_speed_mps"], label=name)

    axes[0].axhline(distance_target_m, color="black", linestyle="--", linewidth=1.0, label="Target distance")
    axes[0].set_ylabel("Distance along heading (m)")
    axes[0].set_title("Straight-Line Progress")
    axes[0].legend()

    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Forward speed (m/s)")
    axes[1].set_title("Forward Speed")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)


def main():
    parser = argparse.ArgumentParser(description="Step 1 benchmark: straight-line drive with fixed PID controllers.")
    parser.add_argument("--distance-target-m", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=4000)
    parser.add_argument("--output-dir", default="benchmark_results")
    parser.add_argument("--render", action="store_true", help="Visualize each benchmark rollout in the MuJoCo viewer.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    model, data = build_sim()
    records = []
    all_traces = {}

    for controller in CONTROLLERS:
        if args.render:
            print(f"Rendering controller: {controller.name}")
        metrics, traces = run_episode(
            model,
            data,
            controller,
            distance_target_m=args.distance_target_m,
            max_steps=args.max_steps,
            render=args.render,
        )
        records.append(metrics)
        all_traces[controller.name] = traces

    summary_df = pd.DataFrame(records)
    summary_path = os.path.join(args.output_dir, "straight_benchmark_summary.csv")
    plot_path = os.path.join(args.output_dir, "straight_benchmark_plot.png")

    summary_df.to_csv(summary_path, index=False)
    plot_traces(all_traces, distance_target_m=args.distance_target_m, output_path=plot_path)

    print("Straight benchmark completed.")
    print(f"Saved summary to {summary_path}")
    print(f"Saved plot to {plot_path}")
    print("Initial heading with identity quaternion points along +x direction.")


if __name__ == "__main__":
    main()

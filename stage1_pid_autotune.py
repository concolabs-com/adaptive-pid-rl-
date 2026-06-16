import argparse
import os

import matplotlib.pyplot as plt
import mujoco
import numpy as np
import pandas as pd

CAR_XML_PATH = os.path.join("envs", "assets", "car_model.xml")
CONTROLLER_FIXED_STANDARD = "Fixed PID (Standard)"
CONTROLLER_FIXED_ROBUST = "Fixed PID (Robust)"


def load_sim():
    model = mujoco.MjModel.from_xml_path(CAR_XML_PATH)
    data = mujoco.MjData(model)
    return model, data


def reset_nominal(model, data):
    mujoco.mj_resetData(model, data)
    data.qpos[0] = 0.0
    data.qpos[1] = 0.0
    data.qpos[2] = 0.03
    data.qpos[3:7] = np.array([1.0, 0.0, 0.0, 0.0])
    data.qvel[:] = 0.0
    data.ctrl[:] = 0.0
    mujoco.mj_forward(model, data)


def get_forward_speed(data, heading):
    vx = float(data.qvel[0])
    vy = float(data.qvel[1])
    return float(vx * heading[0] + vy * heading[1])


def get_distance_along_heading(data, heading):
    x = float(data.qpos[0])
    y = float(data.qpos[1])
    return float(x * heading[0] + y * heading[1])


def detect_drive_signs(model, data, heading):
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

    data.qpos[:] = start_qpos
    data.qvel[:] = start_qvel
    mujoco.mj_forward(model, data)
    return best_pair


def first_crossing_time(t, y_norm, threshold):
    idx = np.flatnonzero(y_norm >= threshold)
    if len(idx) == 0:
        return float("nan")
    i = int(idx[0])
    if i == 0:
        return float(t[0])

    t0, t1 = float(t[i - 1]), float(t[i])
    y0, y1 = float(y_norm[i - 1]), float(y_norm[i])
    if abs(y1 - y0) < 1e-12:
        return t1

    alpha = (threshold - y0) / (y1 - y0)
    alpha = float(np.clip(alpha, 0.0, 1.0))
    return t0 + alpha * (t1 - t0)


def identify_fopdt_from_step(times, speeds, warmup_steps, u_step, u0=0.0):
    t = np.asarray(times, dtype=np.float64)
    y = np.asarray(speeds, dtype=np.float64)

    if len(y) <= warmup_steps + 5:
        raise RuntimeError("Not enough data points for identification.")

    y_pre = y[:warmup_steps]
    y_post = y[warmup_steps:]
    t_post = t[warmup_steps:] - t[warmup_steps]

    y0 = float(np.mean(y_pre[max(0, len(y_pre) // 2) :]))
    yss = float(np.mean(y_post[int(0.8 * len(y_post)) :]))

    du = float(u_step - u0)
    dy = float(yss - y0)
    if abs(du) < 1e-12 or abs(dy) < 1e-6:
        raise RuntimeError("Step response too small for reliable identification.")

    k = dy / du
    y_norm = (y_post - y0) / dy

    t10 = first_crossing_time(t_post, y_norm, 0.10)
    t63 = first_crossing_time(t_post, y_norm, 0.632)
    t90 = first_crossing_time(t_post, y_norm, 0.90)

    if np.isnan(t10) or np.isnan(t63) or np.isnan(t90):
        raise RuntimeError("Could not find required crossing times (10/63.2/90%).")

    tau = (t90 - t10) / np.log(9.0)
    tau = float(max(tau, 1e-3))
    theta = float(max(t63 - tau, 0.0))

    return {
        "k": float(k),
        "tau": tau,
        "theta": theta,
        "y0": y0,
        "yss": yss,
        "du": du,
        "dy": dy,
        "t10": float(t10),
        "t63": float(t63),
        "t90": float(t90),
    }


def simc_pi_gains(k, tau, theta, tau_c):
    k_eff = float(k)
    if abs(k_eff) < 1e-8:
        raise RuntimeError("Process gain is near zero; SIMC tuning is ill-conditioned.")

    kp = (1.0 / k_eff) * (tau / max(tau_c + theta, 1e-6))
    tau_i = min(tau, 4.0 * (tau_c + theta))
    ki = kp / max(tau_i, 1e-6)
    kd = 0.0
    return float(kp), float(ki), float(kd)


def plot_step_response(times, speeds, warmup_steps, id_row, output_path):
    t = np.asarray(times, dtype=np.float64)
    y = np.asarray(speeds, dtype=np.float64)

    t_rel = t - t[warmup_steps]
    y0 = id_row["y0"]
    dy = id_row["dy"]
    tau = id_row["tau"]
    theta = id_row["theta"]

    y_model = np.full_like(t_rel, y0)
    active = t_rel >= theta
    y_model[active] = y0 + dy * (1.0 - np.exp(-(t_rel[active] - theta) / tau))

    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.plot(t_rel, y, label="Measured speed", linewidth=2.0)
    ax.plot(t_rel, y_model, "--", label="FOPDT fit", linewidth=1.8)
    ax.axvline(0.0, color="black", linestyle=":", linewidth=1.0, label="Step start")
    ax.set_xlabel("Time since step (s)")
    ax.set_ylabel("Forward speed (m/s)")
    ax.set_title("Stage 1 Step Test Identification")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)


def run_step_test(model, data, step_input, warmup_steps, step_steps):
    heading = np.array([1.0, 0.0], dtype=np.float64)
    left_sign, right_sign = detect_drive_signs(model, data, heading)

    dt = float(model.opt.timestep)
    times = []
    speeds = []

    for i in range(warmup_steps + step_steps):
        u = 0.0 if i < warmup_steps else float(step_input)
        data.ctrl[0] = left_sign * u
        data.ctrl[1] = right_sign * u
        mujoco.mj_step(model, data)

        times.append((i + 1) * dt)
        speeds.append(get_forward_speed(data, heading))

    return times, speeds


def main():
    parser = argparse.ArgumentParser(description="Step-test based SIMC auto-tuning for Stage 1 fixed PID baselines.")
    parser.add_argument("--output-dir", default="benchmark_results")
    parser.add_argument("--step-input", type=float, default=0.35)
    parser.add_argument("--warmup-steps", type=int, default=200)
    parser.add_argument("--step-steps", type=int, default=1200)
    parser.add_argument("--target-speed", type=float, default=0.55)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    model, data = load_sim()
    reset_nominal(model, data)

    times, speeds = run_step_test(
        model,
        data,
        step_input=args.step_input,
        warmup_steps=args.warmup_steps,
        step_steps=args.step_steps,
    )

    identified = identify_fopdt_from_step(
        times,
        speeds,
        warmup_steps=args.warmup_steps,
        u_step=args.step_input,
        u0=0.0,
    )

    k = identified["k"]
    tau = identified["tau"]
    theta = identified["theta"]

    tau_c_standard = max(theta, 0.05)
    tau_c_robust = max(3.0 * theta, 0.15)

    kp_std, ki_std, kd_std = simc_pi_gains(k, tau, theta, tau_c_standard)
    kp_rob, ki_rob, kd_rob = simc_pi_gains(k, tau, theta, tau_c_robust)

    gains_df = pd.DataFrame(
        [
            {
                "controller": CONTROLLER_FIXED_STANDARD,
                "kp": kp_std,
                "ki": ki_std,
                "kd": kd_std,
                "target_speed": float(args.target_speed),
                "method": "SIMC-PI",
                "tau_c": tau_c_standard,
            },
            {
                "controller": CONTROLLER_FIXED_ROBUST,
                "kp": kp_rob,
                "ki": ki_rob,
                "kd": kd_rob,
                "target_speed": float(args.target_speed),
                "method": "SIMC-PI",
                "tau_c": tau_c_robust,
            },
        ]
    )

    id_df = pd.DataFrame([identified])
    trace_df = pd.DataFrame({"time_s": times, "forward_speed_mps": speeds})

    gains_path = os.path.join(args.output_dir, "stage1_simc_controller_gains.csv")
    id_path = os.path.join(args.output_dir, "stage1_step_identification.csv")
    trace_path = os.path.join(args.output_dir, "stage1_step_trace.csv")
    plot_path = os.path.join(args.output_dir, "stage1_step_identification_plot.png")

    gains_df.to_csv(gains_path, index=False)
    id_df.to_csv(id_path, index=False)
    trace_df.to_csv(trace_path, index=False)
    plot_step_response(times, speeds, args.warmup_steps, identified, plot_path)

    print("Stage 1 auto-tuning complete.")
    print(f"Identified FOPDT: k={k:.5f}, tau={tau:.5f}, theta={theta:.5f}")
    print(f"Saved controller gains CSV: {gains_path}")
    print(f"Saved identification CSV: {id_path}")
    print(f"Saved step trace CSV: {trace_path}")
    print(f"Saved identification plot: {plot_path}")


if __name__ == "__main__":
    main()

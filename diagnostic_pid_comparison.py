"""
Diagnostic script to verify PID behavior step-by-step for Stage1 vs Stage2.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from envs.adaptive_suspension import AdaptiveSuspensionEnv
from stage1_fixed_pid_benchmark import Scenario, apply_scenario, load_controller_specs, load_sim, reset_episode
from stage1_fixed_pid_benchmark import run_episode as run_stage1_episode


def diagnose_stage1_pid(target_distance: float = 5.0, max_steps: int = 100):
    """Run Stage1 PID and log each step's calculation."""
    print("\n" + "=" * 70)
    print("STAGE 1 PID DIAGNOSIS")
    print("=" * 70)

    model, data, car_body_id, floor_geom_id, base_car_mass, base_floor_friction = load_sim()
    specs = load_controller_specs("benchmark_results/stage1_simc_controller_gains.csv")
    robust = [s for s in specs if s.name == "Fixed PID (Robust)"][0]

    scenario = Scenario(name="Standard", mass_scale=1.0, friction_scale=1.0)
    apply_scenario(model, car_body_id, floor_geom_id, base_car_mass, base_floor_friction, scenario)
    reset_episode(model, data, np.random.default_rng(42))

    print(f"PID Gains (from CSV): Kp={robust.kp:.6f}, Ki={robust.ki:.6f}, Kd={robust.kd:.6f}")
    print(f"Target Speed: {robust.target_speed:.2f} m/s")
    print("Output Limits: (-0.35, 1.0)")
    print()

    metrics, traces = run_stage1_episode(
        model,
        data,
        robust,
        distance_target_m=target_distance,
        max_steps=max_steps,
        tolerance_m=0.05,
        render=False,
    )

    print("Stage1 Results:")
    print(f"  Success: {bool(metrics['success'])}")
    print(f"  Settling time: {metrics['settling_time_s']:.3f}s")
    print(f"  Final error: {metrics['final_abs_error_m']:.4f}m")
    print()

    times = np.array(traces["time_s"])
    positions = np.array(traces["distance_m"])
    velocities = np.array(traces["forward_speed_mps"])

    print("Sample Step Logs (first 10, then every 10th):")
    print(f"{'Step':<6} {'Time(s)':<8} {'Pos(m)':<8} {'Vel(m/s)':<10} {'Dist_rem(m)':<12}")
    print("-" * 54)

    idxs = list(range(min(10, len(times)))) + list(range(10, len(times), 10))
    idxs = sorted(set(idxs))

    for i in idxs:
        pos = positions[i]
        vel = velocities[i]
        dist_rem = target_distance - pos
        print(f"{i:<6} {times[i]:<8.3f} {pos:<8.3f} {vel:<10.4f} {dist_rem:<12.4f}")

    # Check velocity statistics
    print("\nVelocity Statistics:")
    print(f"  Min: {np.min(velocities):.4f} m/s")
    print(f"  Max: {np.max(velocities):.4f} m/s")
    print(f"  Mean (cruise phase): {np.mean(velocities[velocities > 0.35]):.4f} m/s")
    print(f"  Std: {np.std(velocities):.4f}")

    return times, positions, velocities


def diagnose_stage2_pid(target_distance: float = 5.0, max_steps: int = 100):
    """Run Stage2 PID and log each step's calculation."""
    print("\n" + "=" * 70)
    print("STAGE 2 PID DIAGNOSIS")
    print("=" * 70)

    # Load config from output dir
    import json

    config_path = Path("benchmark_results/stage2_warm500k_disturbance_eval_longhorizon_v2/stage2_config.json")
    with open(config_path) as f:
        cfg = json.load(f)

    print(f"Base Gains: Kp={cfg['gain_base_kp']}, Ki={cfg['gain_base_ki']}, Kd={cfg['gain_base_kd']}")
    print(f"Setpoint Type: POSITION (target_pos={target_distance}m)")
    print("Output Limits: (-1.0, 1.0)")
    print()

    # Create minimal environment for diagnostics
    env = AdaptiveSuspensionEnv(
        target_pos=target_distance,
        hold_steps=25,
        max_episode_steps=max_steps,
        stop_tolerance=0.05,
        gain_base_kp=cfg["gain_base_kp"],
        gain_base_ki=cfg["gain_base_ki"],
        gain_base_kd=cfg["gain_base_kd"],
        gain_delta_kp=cfg["gain_delta_kp"],
        gain_delta_ki=cfg["gain_delta_ki"],
        gain_delta_kd=cfg["gain_delta_kd"],
        gain_range_kp=tuple(cfg["gain_range_kp"]),
        gain_range_ki=tuple(cfg["gain_range_ki"]),
        gain_range_kd=tuple(cfg["gain_range_kd"]),
    )

    # Neutral policy (action = 0) for diagnostics
    obs, _ = env.reset(seed=42)

    times = []
    positions = []
    velocities = []
    errors = []

    print("Neutral Policy (action=[0,0,0]) Simulation:")
    print(f"{'Step':<6} {'Time(s)':<8} {'Pos(m)':<8} {'Vel(m/s)':<10} {'Error(m)':<10}")
    print("-" * 52)

    for step in range(max_steps):
        action = np.array([0.0, 0.0, 0.0], dtype=np.float32)  # Neutral action
        obs, reward, terminated, truncated, info = env.step(action)

        state = info.get("state", {})
        pos = state.get("pos", np.nan)
        vel = state.get("vel", np.nan)

        times.append((step + 1) * 0.01)  # dt = 0.01s
        positions.append(pos)
        velocities.append(vel)
        errors.append(target_distance - pos)

        if step < 10 or step % 10 == 0:
            print(f"{step:<6} {times[-1]:<8.3f} {pos:<8.3f} {vel:<10.4f} {errors[-1]:<10.4f}")

        if terminated or truncated:
            break

    env.close()

    print("\nVelocity Statistics:")
    velocities_arr = np.array(velocities)
    print(f"  Min: {np.min(velocities_arr):.4f} m/s")
    print(f"  Max: {np.max(velocities_arr):.4f} m/s")
    print(f"  Mean (positive vel only): {np.mean(velocities_arr[velocities_arr > 0.05]):.4f} m/s")
    print(f"  Std: {np.std(velocities_arr):.4f}")

    return np.array(times), np.array(positions), np.array(velocities)


def plot_comparison(stage1_times, stage1_pos, stage1_vel, stage2_times, stage2_pos, stage2_vel):
    """Plot both trajectories for comparison."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    axes[0].plot(stage1_times, stage1_pos, "o-", label="Stage1 PID (Robust)", linewidth=2, markersize=4)
    axes[0].plot(stage2_times, stage2_pos, "s-", label="Stage2 (neutral action, base gains)", linewidth=2, markersize=4)
    axes[0].axhline(5.0, color="k", linestyle="--", linewidth=1, label="Target")
    axes[0].set_ylabel("Position (m)")
    axes[0].set_title("Position vs Time: Stage1 vs Stage2 PID")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(stage1_times, stage1_vel, "o-", label="Stage1 PID (Robust)", linewidth=2, markersize=4)
    axes[1].plot(stage2_times, stage2_vel, "s-", label="Stage2 (neutral action, base gains)", linewidth=2, markersize=4)
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Velocity (m/s)")
    axes[1].set_title("Velocity vs Time: Stage1 vs Stage2 PID")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("benchmark_results/diagnostic_pid_comparison.png", dpi=150, bbox_inches="tight")
    print("\nPlot saved to: benchmark_results/diagnostic_pid_comparison.png")
    plt.close()


if __name__ == "__main__":
    s1_times, s1_pos, s1_vel = diagnose_stage1_pid(target_distance=5.0, max_steps=200)
    s2_times, s2_pos, s2_vel = diagnose_stage2_pid(target_distance=5.0, max_steps=200)

    plot_comparison(s1_times, s1_pos, s1_vel, s2_times, s2_pos, s2_vel)

    print("\n" + "=" * 70)
    print("KEY DIFFERENCES:")
    print("=" * 70)
    print("Stage1: Velocity-based PID (error = target_speed - measured_speed)")
    print("Stage2: Position-based PID (error = target_pos - measured_pos)")
    print("Stage1 Output Limits: (-0.35, 1.0) [asymmetric]")
    print("Stage2 Output Limits: (-1.0, 1.0)  [symmetric]")

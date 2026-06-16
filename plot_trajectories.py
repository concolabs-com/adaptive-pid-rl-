#!/usr/bin/env python3
"""
Trajectory comparison: Stage 4 GRU vs Stage 4b Frame Stacking.

Plots per-step Displacement-Time, Velocity-Time, and Gains-Time for all 3
eval scenarios. Both models trained with obs_keep_dims=6 (blind — no context).

Output: benchmark_results/trajectory_comparison.png
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from stage2_meta_rl_reproduction import (
    SCENARIOS,
    Agent,
    EvalScenario,
    Stage2Config,
    build_domain_randomization_config,
    make_eval_env,
    prepare_eval_episode,
)

BASE_DIR = Path(__file__).parent
GRU_MODEL = BASE_DIR / "benchmark_results/stage4_blind_meta_rl/seed_7/models/meta_rl_agent.pth"
FS_MODEL = BASE_DIR / "benchmark_results/stage4b_blind_framestacking/seed_7/models/meta_rl_agent.pth"
FS_CONFIG = BASE_DIR / "benchmark_results/stage4b_blind_framestacking/stage2_config.json"
OUT_PATH = BASE_DIR / "benchmark_results/trajectory_comparison.png"

EVAL_SEED = 70000


def load_config_from_json(path: Path) -> Stage2Config:
    with open(path) as f:
        d = json.load(f)
    tuple_fields = [
        "gain_range_kp",
        "gain_range_ki",
        "gain_range_kd",
        "disturbance_step_range",
        "disturbance_time_range_s",
        "disturbance_mass_scale_range",
        "disturbance_friction_scale_range",
        "disturbance_mass_scale_start_range",
        "disturbance_mass_scale_end_range",
        "disturbance_friction_scale_start_range",
        "disturbance_friction_scale_end_range",
        "patch_x_range",
    ]
    for field in tuple_fields:
        if field in d:
            d[field] = tuple(d[field])
    d["curriculum_phases"] = [tuple(p) for p in d["curriculum_phases"]]
    d["output_dir"] = Path(d["output_dir"])
    d["init_model_path"] = None
    # Remove keys not in Stage2Config (e.g. "scenarios" added by the eval dump)
    valid_fields = set(Stage2Config.__dataclass_fields__)
    d = {k: v for k, v in d.items() if k in valid_fields}
    return Stage2Config(**d)


def make_env_from_config(config: Stage2Config):
    return make_eval_env(
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
        use_disturbance=config.disturbance_in_eval,
        randomization_config=build_domain_randomization_config(config, for_eval=True),
    )


def run_trajectory(env, agent: Agent, scenario: EvalScenario, seed: int, recurrent: bool) -> dict:
    obs, _ = prepare_eval_episode(env, scenario, seed)
    device = next(agent.parameters()).device
    dt = float(env.unwrapped.dt)

    hidden_state = agent.get_initial_state(1, device=device) if recurrent else None
    done_tensor = torch.zeros(1, device=device) if recurrent else None

    positions, velocities, kps, kis, kds, times = [], [], [], [], [], []

    for step in range(5000):
        with torch.no_grad():
            obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
            if recurrent:
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

    return {
        "t": np.array(times),
        "pos": np.array(positions),
        "vel": np.array(velocities),
        "kp": np.array(kps),
        "ki": np.array(kis),
        "kd": np.array(kds),
    }


def main():
    fs_config = load_config_from_json(FS_CONFIG)
    gru_config = Stage2Config(
        **{
            **{k: getattr(fs_config, k) for k in Stage2Config.__dataclass_fields__},
            "recurrent_policy": True,
            "stack_size": 1,
            "recurrent_hidden_size": 128,
        }
    )

    print("Loading Frame Stack model...")
    fs_env = make_env_from_config(fs_config)
    fs_agent = Agent(fs_env, recurrent=False)
    fs_agent.load_state_dict(torch.load(FS_MODEL, map_location="cpu", weights_only=True))
    fs_agent.eval()

    print("Loading GRU model...")
    gru_env = make_env_from_config(gru_config)
    gru_agent = Agent(gru_env, recurrent=True, recurrent_hidden_size=128)
    gru_agent.load_state_dict(torch.load(GRU_MODEL, map_location="cpu", weights_only=True))
    gru_agent.eval()

    target = fs_config.target_pos  # 5.0 m

    # Collect trajectories for all scenarios
    data = {}
    for scenario in SCENARIOS:
        print(f"  Running: {scenario.name}")
        data[scenario.name] = {
            "fs": run_trajectory(fs_env, fs_agent, scenario, EVAL_SEED, recurrent=False),
            "gru": run_trajectory(gru_env, gru_agent, scenario, EVAL_SEED, recurrent=True),
        }

    fs_env.close()
    gru_env.close()

    # --- Plot ---------------------------------------------------------------
    n_scenarios = len(SCENARIOS)
    fig, axes = plt.subplots(n_scenarios, 3, figsize=(15, 4 * n_scenarios))
    fig.suptitle(
        "Trajectory Comparison: Stage 4 GRU vs Stage 4b Frame Stacking\n"
        "(obs_keep_dims=6, blind — no mass/friction context)",
        fontsize=13,
        y=1.01,
    )

    col_titles = ["Displacement (m)", "Velocity (m/s)", "PID Gains"]
    for j, ct in enumerate(col_titles):
        axes[0, j].set_title(ct, fontsize=11, fontweight="bold")

    for i, scenario in enumerate(SCENARIOS):
        traj = data[scenario.name]
        fs_d = traj["fs"]
        gru_d = traj["gru"]

        row_label = f"{scenario.name}\n(m={scenario.mass} kg, fr={scenario.friction})"

        # --- Displacement ---
        ax = axes[i, 0]
        ax.plot(fs_d["t"], fs_d["pos"], color="#2196F3", lw=1.5, label="Frame Stack")
        ax.plot(gru_d["t"], gru_d["pos"], color="#F44336", lw=1.5, linestyle="--", label="GRU")
        ax.axhline(target, color="black", lw=1, linestyle=":", label=f"Target {target} m")
        ax.set_ylabel(row_label, fontsize=9)
        ax.set_xlabel("Time (s)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # --- Velocity ---
        ax = axes[i, 1]
        ax.plot(fs_d["t"], fs_d["vel"], color="#2196F3", lw=1.5, label="Frame Stack")
        ax.plot(gru_d["t"], gru_d["vel"], color="#F44336", lw=1.5, linestyle="--", label="GRU")
        ax.axhline(0, color="black", lw=0.8, linestyle=":")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Velocity (m/s)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # --- Gains ---
        ax = axes[i, 2]
        ax.plot(fs_d["t"], fs_d["kp"], color="#2196F3", lw=1.2, label="Kp FS")
        ax.plot(gru_d["t"], gru_d["kp"], color="#2196F3", lw=1.2, linestyle="--", label="Kp GRU")
        ax.plot(fs_d["t"], fs_d["ki"], color="#4CAF50", lw=1.2, label="Ki FS")
        ax.plot(gru_d["t"], gru_d["ki"], color="#4CAF50", lw=1.2, linestyle="--", label="Ki GRU")
        ax.plot(fs_d["t"], fs_d["kd"], color="#FF9800", lw=1.2, label="Kd FS")
        ax.plot(gru_d["t"], gru_d["kd"], color="#FF9800", lw=1.2, linestyle="--", label="Kd GRU")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Gain value")
        ax.legend(fontsize=7, ncol=2)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
    print(f"\nSaved: {OUT_PATH}")


if __name__ == "__main__":
    main()

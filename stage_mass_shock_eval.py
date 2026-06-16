#!/usr/bin/env python3
"""
Mass Shock Eval: mid-approach mass doubling/tripling.

Fires a large mass increase (2-3x) at step 30-60 (mid-approach) for all three
approaches. Friction is held constant — we know friction doesn't affect dynamics
in this rolling model. Tests whether RL adaptation beats fixed PID when inertia
suddenly changes before the braking zone.

  Fixed PID  : action=[0,0,0] always — Kp=1.8, Ki=0.7, Kd=0.5, cannot react
  Stage 5a   : context-aware RL — sees updated mass_scale in obs, can raise Kd
  Stage 5b   : blind RL — infers heavier mass from trajectory, must adapt blind

Scenarios:
  Standard:   base mass=10kg  -> shocked to 20-30kg
  Heavy:      base mass=20kg  -> shocked to 40-60kg

Output: benchmark_results/mass_shock_eval/
"""

import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import gymnasium as gym
import matplotlib.pyplot as plt
import torch

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from agents.domain_randomization import DomainRandomizationWrapper
from agents.model import Agent
from envs.adaptive_suspension import AdaptiveSuspensionEnv
from stage2_meta_rl_reproduction import ObservationFeatureSelectWrapper

OUTPUT_DIR = Path("benchmark_results/mass_shock_eval")

MODEL_5A = Path("benchmark_results/stage5a_context_cliff/seed_7/models/meta_rl_agent.pth")
MODEL_5B = Path("benchmark_results/stage5b_blind_cliff/seed_7/models/meta_rl_agent.pth")

STACK_SIZE = 10

ENV_KWARGS = dict(
    target_pos=5.0,
    hold_steps=25,
    max_episode_steps=5000,
    stop_tolerance=0.05,
    gain_base_kp=1.8,
    gain_base_ki=0.7,
    gain_base_kd=0.5,
    gain_delta_kp=1.0,
    gain_delta_ki=0.6,
    gain_delta_kd=2.0,
    gain_range_kp=(0.0, 6.0),
    gain_range_ki=(0.0, 3.0),
    gain_range_kd=(0.0, 5.0),
    safety_speed_governor_enabled=False,
    safety_hard_overshoot_m=-1.0,
    safety_overshoot_penalty=0.0,
    terminal_hold_bonus=50.0,
    terminal_hold_velocity_threshold=0.08,
    approach_progress_cutoff_m=2.0,
    brake_zone_vel_sq_coef=10.0,
    near_approach_zone_m=2.0,
    near_approach_coef=0.0,
    near_target_zone_m=0.8,
    near_target_coef=0.0,
    near_target_excess_thresh=0.10,
    near_target_excess_coef=0.0,
    decel_bonus_coef=10.0,
    near_target_init_prob=0.0,
    near_target_init_range_m=0.04,
)

MASS_SHOCK_CONFIG = {
    "mass_range": (5.0, 20.0),
    "friction_range": (0.1, 2.0),
    "initial_randomization_enabled": False,  # scenario sets mass directly
    "mid_episode_disturbance_enabled": True,
    "disturbance_mode": "step",
    "disturbance_step_range": (30, 60),  # mid-approach, before braking zone
    "disturbance_mass_scale_range": (2.0, 3.0),  # double to triple base mass
    "disturbance_friction_scale_range": (1.0, 1.0),  # no friction change
    "position_patch_enabled": False,  # isolate mass effect only
}

SCENARIOS = [
    ("Standard", 10.0, 1.0),  # shocked to 20-30kg
    ("Heavy", 20.0, 0.2),  # shocked to 40-60kg
]

TOLERANCE = 0.05
HOLD_STEPS = 25
MAX_STEPS = 5000
TARGET_POS = 5.0
EVAL_SEEDS = list(range(70000, 70010))
FIXED_ACTION = np.zeros(3, dtype=np.float32)


def make_base_env(mass: float, friction: float):
    env = AdaptiveSuspensionEnv(**ENV_KWARGS)
    env.model.body_mass[1] = mass
    env.model.geom_friction[0, 0] = friction
    return env


def make_eval_env(mass: float, friction: float, obs_keep_dims: int = 0):
    base = make_base_env(mass, friction)
    env = DomainRandomizationWrapper(base, randomization_config=MASS_SHOCK_CONFIG)
    if obs_keep_dims > 0:
        env = ObservationFeatureSelectWrapper(env, keep_dims=obs_keep_dims)
    env = gym.wrappers.TimeLimit(env, max_episode_steps=MAX_STEPS)
    env = gym.wrappers.FrameStackObservation(env, stack_size=STACK_SIZE)
    return env


def load_rl_agent(model_path: Path, env) -> Agent:
    agent = Agent(env)
    state_dict = torch.load(model_path, map_location="cpu")
    agent.load_state_dict(state_dict)
    agent.eval()
    return agent


def compute_settling_time(errors: np.ndarray, dt: float) -> tuple[float, int]:
    within = np.abs(errors) <= TOLERANCE
    n = len(within)
    for i in range(n):
        if within[i]:
            end = min(i + HOLD_STEPS, n)
            if np.all(within[i:end]):
                return float(i * dt), 1
    return float(MAX_STEPS * dt), 0


def run_episode(env, agent, seed: int, scenario_name: str, mass: float, friction: float) -> dict:
    env.unwrapped.model.body_mass[1] = mass
    env.unwrapped.model.geom_friction[0, 0] = friction
    obs, _ = env.reset(seed=seed)

    dt = float(env.unwrapped.dt)
    positions, velocities, kd_hist, rewards = [], [], [], []
    disturbance_step = None

    for step_i in range(MAX_STEPS):
        if agent is None:
            action = FIXED_ACTION
        else:
            with torch.no_grad():
                obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
                action = agent.actor_mean(obs_t.reshape(1, -1))
                action = torch.clamp(action, -1.0, 1.0).squeeze(0).numpy()

        obs, reward, terminated, truncated, info = env.step(action)
        positions.append(float(info["state"]["pos"]))
        velocities.append(float(info["state"]["vel"]))
        kd_hist.append(float(info["gains"]["kd"]))
        rewards.append(float(reward))

        ext = info.get("external_factors", {})
        if ext.get("disturbance_fired") and disturbance_step is None:
            disturbance_step = step_i

        if terminated or truncated:
            break

    positions_arr = np.asarray(positions)
    errors_arr = TARGET_POS - positions_arr
    settling_s, settled = compute_settling_time(errors_arr, dt)
    overshoot = float(max(np.max(positions_arr) - TARGET_POS, 0.0))
    iae = float(np.sum(np.abs(errors_arr)) * dt)
    final_abs_error = float(abs(errors_arr[-1]))
    success = 1 if (settled and final_abs_error <= TOLERANCE) else 0

    return {
        "seed": seed,
        "scenario": scenario_name,
        "settling_time_mean": settling_s,
        "overshoot_mean": overshoot,
        "iae_mean": iae,
        "final_abs_error_mean": final_abs_error,
        "success_rate": 100.0 if success else 0.0,
        "mean_reward": float(np.mean(rewards)),
        "disturbance_step": disturbance_step if disturbance_step is not None else -1,
        "_positions": positions,
        "_velocities": velocities,
        "_kd": kd_hist,
        "_dt": dt,
    }


def plot_comparison(results_by_agent: dict, scenario_name: str, out_path: Path) -> None:
    colors = {"Fixed PID": "#E53935", "Stage 5a (Context)": "#1E88E5", "Stage 5b (Blind)": "#43A047"}
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(f"Mass Shock Eval — {scenario_name} (seed 70000)", fontsize=11)
    axes[0].set_title("Displacement (m)")
    axes[1].set_title("Velocity (m/s)")
    axes[2].set_title("Kd gain")

    for agent_name, r in results_by_agent.items():
        dt = r["_dt"]
        t = np.arange(len(r["_positions"])) * dt
        c = colors.get(agent_name, "gray")
        axes[0].plot(t, r["_positions"], label=agent_name, color=c)
        axes[1].plot(t, r["_velocities"], color=c)
        axes[2].plot(t, r["_kd"], color=c)
        if r["disturbance_step"] > 0:
            t_shock = r["disturbance_step"] * dt
            for ax in axes:
                ax.axvline(t_shock, color=c, linestyle=":", linewidth=0.8, alpha=0.7)

    axes[0].axhline(TARGET_POS, color="k", linestyle="--", linewidth=0.8, label="target")
    axes[0].axhline(TARGET_POS + TOLERANCE, color="gray", linestyle=":", linewidth=0.6)
    axes[0].axhline(TARGET_POS - TOLERANCE, color="gray", linestyle=":", linewidth=0.6)
    axes[0].legend(fontsize=7)
    axes[0].set_xlabel("Time (s)")
    axes[1].axhline(0, color="k", linestyle="--", linewidth=0.6)
    axes[1].set_xlabel("Time (s)")
    axes[2].set_xlabel("Time (s)")

    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"  Plot: {out_path}")


def run_agent(
    label: str, model_path: Path | None, obs_keep_dims: int, scenario_name: str, mass: float, friction: float
) -> list[dict]:
    env = make_eval_env(mass, friction, obs_keep_dims=obs_keep_dims)
    agent = load_rl_agent(model_path, env) if model_path is not None else None
    records = []
    for seed in EVAL_SEEDS:
        r = run_episode(env, agent, seed, scenario_name, mass, friction)
        records.append(r)
    env.close()
    return records


def main() -> int:
    for p in [MODEL_5A, MODEL_5B]:
        if not p.exists():
            print(f"ERROR: model not found: {p}", file=sys.stderr)
            return 1

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("MASS SHOCK EVAL")
    print("  disturbance: mass x2-3 at step 30-60 (mid-approach)")
    print("  friction:    unchanged (known to not affect dynamics)")
    print("  agents:      Fixed PID | Stage 5a (context) | Stage 5b (blind)")
    print(f"  output:      {OUTPUT_DIR}")
    print("=" * 65)
    print()

    agents_cfg = [
        ("Fixed PID", None, 0),
        ("Stage 5a (Context)", MODEL_5A, 0),
        ("Stage 5b (Blind)", MODEL_5B, 6),
    ]

    all_records = []

    for scenario_name, mass, friction in SCENARIOS:
        print(f"Scenario: {scenario_name}  (base mass={mass}kg, shocked to {mass*2:.0f}-{mass*3:.0f}kg)")
        seed0_by_agent = {}

        for agent_label, model_path, obs_keep_dims in agents_cfg:
            records = run_agent(agent_label, model_path, obs_keep_dims, scenario_name, mass, friction)
            for r in records:
                flat = {k: v for k, v in r.items() if not k.startswith("_")}
                flat["agent"] = agent_label
                all_records.append(flat)
            seed0_by_agent[agent_label] = records[0]

            mean_settle = np.mean([r["settling_time_mean"] for r in records])
            mean_overshoot = np.mean([r["overshoot_mean"] for r in records])
            mean_success = np.mean([r["success_rate"] for r in records])
            mean_kd_max = np.mean([max(r["_kd"]) for r in records])
            print(
                f"  {agent_label:<22}  settle={mean_settle:.3f}s  "
                f"overshoot={mean_overshoot:.4f}m  "
                f"success={mean_success:.0f}%  "
                f"max_Kd={mean_kd_max:.2f}"
            )

        plot_comparison(
            seed0_by_agent,
            scenario_name,
            OUTPUT_DIR / f"trajectories_{scenario_name.lower().replace(' ', '_')}.png",
        )
        print()

    df = pd.DataFrame(all_records)
    df.to_csv(OUTPUT_DIR / "mass_shock_summary.csv", index=False)

    print("=" * 65)
    print("FULL SUMMARY (mean across 10 seeds)")
    print("-" * 65)
    for scenario_name, _, _ in SCENARIOS:
        print(f"\n  {scenario_name}")
        sub = df[df["scenario"] == scenario_name]
        print(f"  {'Agent':<22} {'Settling':>9} {'Overshoot':>11} {'Success%':>9} {'FinalErr':>10}")
        print(f"  {'-'*61}")
        for agent_label, _, _ in agents_cfg:
            ag = sub[sub["agent"] == agent_label]
            print(
                f"  {agent_label:<22} "
                f"{ag['settling_time_mean'].mean():>9.3f}s "
                f"{ag['overshoot_mean'].mean():>11.4f}m "
                f"{ag['success_rate'].mean():>9.1f}% "
                f"{ag['final_abs_error_mean'].mean():>10.4f}m"
            )
    print("=" * 65)

    return 0


if __name__ == "__main__":
    sys.exit(main())

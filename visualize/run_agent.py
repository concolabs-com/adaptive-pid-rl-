#!/usr/bin/env python3
"""
Standalone MuJoCo visualizer for trained adaptive suspension agents.

Usage:
    python run_agent.py --model path/to/meta_rl_agent.pth
    python run_agent.py --model path/to/model.pth --target 5.0 --scenario heavy
    python run_agent.py --model path/to/model.pth --mass 15.0 --friction 0.5 --obs-dims 8

Controls:
    Terminal: Press ENTER to start the agent.
    Terminal: Press ENTER to close after episode ends.
    MuJoCo window: Close the window to quit at any point.

Preset scenarios (--scenario):
    standard   mass=10.0 kg, friction=1.0   (training nominal)
    heavy      mass=20.0 kg, friction=0.2   (heavy + slippery)
    light      mass=5.0  kg, friction=2.0   (light + grippy)

Obs-dims for each model type:
    8  context-aware (stage5a):  [pos, vel, err, kp, ki, kd, mass_scale, friction_scale]
    6  blind          (stage5b): [pos, vel, err, kp, ki, kd]
"""

import argparse
import sys
import time
from pathlib import Path

import gymnasium as gym
import mujoco
import mujoco.viewer
import numpy as np
import torch

from environment.agents.model import Agent
from environment.envs.adaptive_suspension import AdaptiveSuspensionEnv
from utils.wrappers import ObservationFeatureSelectWrapper

SCENARIOS = {
    "standard": {"mass": 10.0, "friction": 1.0},
    "heavy": {"mass": 20.0, "friction": 0.2},
    "light": {"mass": 5.0, "friction": 2.0},
}


def build_env(args: argparse.Namespace) -> gym.Env:
    env = AdaptiveSuspensionEnv(
        target_pos=args.target,
        hold_steps=args.hold_steps,
        max_episode_steps=args.max_steps,
        stop_tolerance=args.tolerance,
        gain_base_kp=args.gain_base_kp,
        gain_base_ki=args.gain_base_ki,
        gain_base_kd=args.gain_base_kd,
        gain_delta_kp=args.gain_delta_kp,
        gain_delta_ki=args.gain_delta_ki,
        gain_delta_kd=args.gain_delta_kd,
        gain_range_kp=(0.0, 6.0),
        gain_range_ki=(0.0, 3.0),
        gain_range_kd=(0.0, 5.0),
        approach_progress_cutoff_m=args.approach_progress_cutoff_m,
        near_approach_zone_m=args.near_approach_zone_m,
        brake_zone_vel_sq_coef=10.0,
        near_target_zone_m=0.8,
        near_target_coef=0.0,
        near_approach_coef=0.0,
        decel_bonus_coef=10.0,
        terminal_hold_bonus=50.0,
        terminal_hold_velocity_threshold=0.08,
        safety_speed_governor_enabled=False,
        safety_hard_overshoot_m=-1.0,
        safety_overshoot_penalty=0.0,
    )

    obs_dim = int(env.observation_space.shape[0])
    if 0 < args.obs_dims < obs_dim:
        env = ObservationFeatureSelectWrapper(env, keep_dims=args.obs_dims)

    env = gym.wrappers.TimeLimit(env, max_episode_steps=args.max_steps)
    env = gym.wrappers.FrameStackObservation(env, stack_size=args.stack_size)
    return env


def apply_scenario(
    base_env: AdaptiveSuspensionEnv, mass: float, friction: float
) -> None:
    """Set mass/friction on the MuJoCo model and update context observation."""
    nominal_mass = float(base_env.model.body_mass[1])
    nominal_friction = float(base_env.model.geom_friction[0, 0])
    base_env.model.body_mass[1] = mass
    base_env.model.geom_friction[0, 0] = friction
    base_env.set_disturbance_context(
        mass_scale=mass / max(nominal_mass, 1e-6),
        friction_scale=friction / max(nominal_friction, 1e-6),
    )


def load_agent(model_path: str, env: gym.Env) -> Agent:
    agent = Agent(env, recurrent=False)
    state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
    agent.load_state_dict(state_dict)
    agent.eval()
    return agent


def wait_for_enter(prompt: str, viewer: mujoco.viewer.Handle) -> bool:
    """Block on ENTER. Return False if viewer was closed before ENTER pressed."""
    print(prompt, end="", flush=True)
    try:
        sys.stdin.readline()
    except (EOFError, KeyboardInterrupt):
        return False
    return viewer.is_running()


def run(args: argparse.Namespace) -> None:
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"ERROR: model not found: {model_path}", file=sys.stderr)
        sys.exit(1)

    # Resolve scenario
    if args.scenario:
        sc = SCENARIOS[args.scenario]
        mass, friction = sc["mass"], sc["friction"]
    else:
        mass, friction = args.mass, args.friction

    env = build_env(args)
    base_env: AdaptiveSuspensionEnv = env.unwrapped

    # Apply mass/friction before first reset so nominal values are captured correctly
    apply_scenario(base_env, mass, friction)

    agent = load_agent(str(model_path), env)

    obs, _ = env.reset(seed=args.seed)
    mujoco.mj_forward(base_env.model, base_env.data)

    viewer = mujoco.viewer.launch_passive(base_env.model, base_env.data)
    viewer.sync()

    scenario_label = (
        args.scenario
        if args.scenario
        else f"mass={mass:.1f}kg, friction={friction:.2f}"
    )
    obs_shape = env.observation_space.shape

    print()
    print("=" * 56)
    print("  Adaptive Suspension Agent Visualizer")
    print(f"  Model    : {model_path.name}")
    print(f"  Target   : {args.target:.1f} m")
    print(f"  Scenario : {scenario_label}")
    print(
        f"  Obs shape: {obs_shape}  (stack={args.stack_size} × dims={obs_shape[0] // args.stack_size})"
    )
    print("=" * 56)
    print()
    print("  >> Press ENTER in this terminal to START  <<")
    print()

    if not wait_for_enter("", viewer):
        viewer.close()
        env.close()
        return

    for ep in range(args.repeat):

        obs, _ = env.reset(seed=args.seed)

        apply_scenario(base_env, mass, friction)

        viewer.sync()

        print(f"\nEpisode {ep+1}/{args.repeat}")
        print(">> Press ENTER to START <<")

        if not wait_for_enter("", viewer):
           break

        print("Running...")

    step = 0
    total_reward = 0.0
    done = False
    final_info = {}

    with torch.no_grad():

        while not done and viewer.is_running():

            obs_arr = np.asarray(obs,dtype=np.float32)

            obs_tensor = torch.from_numpy(
                obs_arr
            ).unsqueeze(0)

            action = agent.actor_mean(
                obs_tensor.reshape(1,-1)
            )

            action = torch.clamp(
                action,
                -1.0,
                1.0
            )

            action_np = action.squeeze(0).numpy()

            obs,reward,terminated,truncated,info = \
                env.step(action_np)

            total_reward += float(reward)

            done = terminated or truncated

            final_info = info

            step += 1

            viewer.sync()

            time.sleep(0.01)

    print(
        f"Episode {ep+1} done "
        f"({step} steps)"
    )

    if viewer.is_running():

        print(
        "\nAll episodes finished."
    )

    wait_for_enter(
        "Press ENTER to CLOSE",
        viewer
    )

    viewer.close()

    env.close()

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Visualize a trained adaptive suspension RL agent in MuJoCo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--repeat", type=int, default=1, help="Number of episodes")
    p.add_argument(
        "--model", required=True, help="Path to .pth checkpoint (agent state dict)"
    )

    # Episode configuration
    p.add_argument(
        "--target",
        type=float,
        default=5.0,
        help="Target position in metres (default: 5.0)",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for episode reset (default: 42)",
    )
    p.add_argument(
        "--max-steps",
        type=int,
        default=500,
        help="Maximum episode steps (default: 500)",
    )
    p.add_argument(
        "--hold-steps",
        type=int,
        default=25,
        help="Consecutive steps within tolerance to count as success (default: 25)",
    )
    p.add_argument(
        "--tolerance",
        type=float,
        default=0.05,
        help="Position tolerance in metres (default: 0.05)",
    )

    # Model architecture (must match training)
    p.add_argument(
        "--stack-size",
        type=int,
        default=10,
        help="Observation stack size — must match training (default: 10)",
    )
    p.add_argument(
        "--obs-dims",
        type=int,
        default=8,
        help="Obs dims per step: 8=context-aware (stage5a), 6=blind (stage5b) (default: 8)",
    )

    # Physics scenario
    sc_group = p.add_mutually_exclusive_group()
    sc_group.add_argument(
        "--scenario", choices=list(SCENARIOS.keys()), help="Preset physics scenario"
    )
    sc_group.add_argument(
        "--mass",
        type=float,
        default=10.0,
        help="Car mass in kg (default: 10.0). Ignored if --scenario is set.",
    )
    p.add_argument(
        "--friction",
        type=float,
        default=1.0,
        help="Floor friction coefficient (default: 1.0). Ignored if --scenario is set.",
    )

    # Gain schedule (thesis_v4_cliff defaults)
    p.add_argument("--gain-base-kp", type=float, default=1.8)
    p.add_argument("--gain-base-ki", type=float, default=0.7)
    p.add_argument("--gain-base-kd", type=float, default=0.5)
    p.add_argument("--gain-delta-kp", type=float, default=1.0)
    p.add_argument("--gain-delta-ki", type=float, default=0.6)
    p.add_argument("--gain-delta-kd", type=float, default=2.0)
    p.add_argument("--approach-progress-cutoff-m", type=float, default=2.0)
    p.add_argument("--near-approach-zone-m", type=float, default=2.0)

    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())

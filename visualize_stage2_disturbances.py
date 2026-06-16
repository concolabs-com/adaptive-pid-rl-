import argparse
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from agents.domain_randomization import DomainRandomizationWrapper
from agents.model import Agent
from envs.adaptive_suspension import AdaptiveSuspensionEnv

SCENARIOS = {
    "standard": {"mass": 10.0, "friction": 1.0},
    "heavy": {"mass": 20.0, "friction": 0.2},
    "light": {"mass": 5.0, "friction": 2.0},
}


class TargetRandomizationWrapper(gym.Wrapper):
    def __init__(self, env, target_min: float, target_max: float):
        super().__init__(env)
        self.target_min = float(target_min)
        self.target_max = float(target_max)

    def reset(self, **kwargs):
        rng = getattr(self.unwrapped, "np_random", np.random)
        sampled_target = float(rng.uniform(self.target_min, self.target_max))
        if hasattr(self.unwrapped, "set_target_pos"):
            self.unwrapped.set_target_pos(sampled_target)
        else:
            self.unwrapped.target_pos = sampled_target
        print(f"  target={sampled_target:.2f} m")
        return self.env.reset(**kwargs)


def make_env(args):
    env = AdaptiveSuspensionEnv(
        render_mode="human",
        target_pos=float(args.target_pos),
        hold_steps=int(args.hold_steps),
        max_episode_steps=int(args.max_steps),
    )
    if bool(args.randomize_target):
        env = TargetRandomizationWrapper(env, target_min=float(args.target_min), target_max=float(args.target_max))
    env = DomainRandomizationWrapper(
        env,
        randomization_config={
            "mass_range": tuple(args.mass_range),
            "friction_range": tuple(args.friction_range),
            "initial_randomization_enabled": False,
            "mid_episode_disturbance_enabled": bool(args.mid_episode_disturbance_enabled),
            "disturbance_mode": args.disturbance_mode,
            "disturbance_step_range": tuple(args.disturbance_step_range),
            "disturbance_time_range_s": tuple(args.disturbance_time_range_s),
            "disturbance_mass_scale_range": tuple(args.disturbance_mass_scale_range),
            "disturbance_friction_scale_range": tuple(args.disturbance_friction_scale_range),
            "position_patch_enabled": bool(args.position_patch_enabled),
            "patch_x_range": tuple(args.patch_x_range),
            "patch_friction_scale": float(args.patch_friction_scale),
            "patch_visual_enabled": True,
            "patch_visual_rgba": (1.0, 0.55, 0.12, 0.7),
            "patch_visual_y_half_width": 0.38,
            "patch_visual_half_height": 0.002,
            "mass_visual_enabled": True,
            "mass_visual_low_rgba": (0.18, 0.55, 1.0, 1.0),
            "mass_visual_high_rgba": (1.0, 0.24, 0.18, 1.0),
        },
    )
    env = gym.wrappers.FrameStackObservation(env, stack_size=int(args.stack_size))
    return env


def load_agent(env, model_path: Path):
    agent = Agent(env)
    state_dict = torch.load(model_path, map_location="cpu")
    agent.load_state_dict(state_dict)
    agent.eval()
    return agent


def run_episode(env, agent, scenario_name: str, episode_seed: int, args):
    obs, _ = env.reset(seed=episode_seed)
    done = False
    last_event_key = None
    print(f"\nScenario: {scenario_name} | seed={episode_seed}")

    while not done:
        with torch.no_grad():
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
            action = agent.actor_mean(obs_tensor.reshape(obs_tensor.shape[0], -1))
            action = torch.clamp(action, -1.0, 1.0)

        obs, reward, terminated, truncated, info = env.step(action.squeeze(0).cpu().numpy())
        done = bool(terminated or truncated)

        ext = info.get("external_factors", {})
        event = ext.get("last_event", {})
        event_key = (event.get("type", ""), event.get("step", -1), event.get("time_s", np.nan))
        if event_key != last_event_key and event.get("type"):
            print(
                "  disturbance: type={type} step={step} time={time:.3f}s "
                "mass={mass:.2f} friction={friction:.2f}".format(
                    type=event.get("type", ""),
                    step=event.get("step", -1),
                    time=float(event.get("time_s", 0.0)),
                    mass=float(event.get("mass", 0.0)),
                    friction=float(event.get("friction", 0.0)),
                )
            )
            last_event_key = event_key

        if ext:
            print(
                f"  pos={info['state']['pos']:.3f} vel={info['state']['vel']:.3f} "
                f"mass={ext.get('mass', float('nan')):.2f} friction={ext.get('friction', float('nan')):.2f}"
            )

        if args.sleep_s > 0:
            time.sleep(args.sleep_s)

    print("  episode finished")


def main():
    parser = argparse.ArgumentParser(description="Visualize mid-episode disturbances in the MuJoCo simulator.")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--scenario", choices=sorted(SCENARIOS.keys()), default="standard")
    parser.add_argument("--target-pos", type=float, default=5.0)
    parser.add_argument("--randomize-target", action="store_true", default=False)
    parser.add_argument("--target-min", type=float, default=3.0)
    parser.add_argument("--target-max", type=float, default=7.0)
    parser.add_argument("--hold-steps", type=int, default=25)
    parser.add_argument("--max-steps", type=int, default=1200)
    parser.add_argument("--stack-size", type=int, default=10)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--sleep-s", type=float, default=0.0)
    parser.add_argument("--mid-episode-disturbance-enabled", action="store_true", default=False)
    parser.add_argument("--disturbance-mode", choices=["step", "time"], default="step")
    parser.add_argument("--disturbance-step-range", type=int, nargs=2, default=(120, 220))
    parser.add_argument("--disturbance-time-range-s", type=float, nargs=2, default=(1.0, 3.0))
    parser.add_argument("--disturbance-mass-scale-range", type=float, nargs=2, default=(0.9, 1.3))
    parser.add_argument("--disturbance-friction-scale-range", type=float, nargs=2, default=(0.5, 1.4))
    parser.add_argument("--position-patch-enabled", action="store_true", default=False)
    parser.add_argument("--patch-x-range", type=float, nargs=2, default=(1.5, 2.4))
    parser.add_argument("--patch-friction-scale", type=float, default=0.35)
    parser.add_argument("--mass-range", type=float, nargs=2, default=(5.0, 20.0))
    parser.add_argument("--friction-range", type=float, nargs=2, default=(0.1, 2.0))
    args = parser.parse_args()

    env = make_env(args)
    agent = load_agent(env, args.model_path)

    scenario = SCENARIOS[args.scenario]
    env.unwrapped.model.body_mass[1] = float(scenario["mass"])
    env.unwrapped.model.geom_friction[0, 0] = float(scenario["friction"])

    run_episode(env, agent, args.scenario, args.seed, args)
    env.close()


if __name__ == "__main__":
    main()

import argparse
from pathlib import Path

import gymnasium as gym
import numpy as np
import pandas as pd
import torch

from agents.model import Agent
from envs.adaptive_suspension import AdaptiveSuspensionEnv


def make_eval_env(stack_size: int, target_pos: float, max_steps: int):
    env = AdaptiveSuspensionEnv(target_pos=target_pos)
    env = gym.wrappers.TimeLimit(env, max_episode_steps=max_steps)
    env = gym.wrappers.FrameStackObservation(env, stack_size=stack_size)
    return env


def run_episode(env, agent, seed: int, tolerance: float):
    obs, _ = env.reset(seed=seed)
    target = float(env.unwrapped.target_pos)
    max_pos = -1e9
    rewards = []
    positions = []

    while True:
        with torch.no_grad():
            obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
            action = agent.actor_mean(obs_t.reshape(obs_t.shape[0], -1))
            action = torch.clamp(action, -1.0, 1.0)

        obs, reward, terminated, truncated, info = env.step(action.squeeze(0).cpu().numpy())
        pos = float(info["state"]["pos"])
        rewards.append(float(reward))
        positions.append(pos)
        max_pos = max(max_pos, pos)

        if terminated or truncated:
            break

    final_pos = positions[-1] if positions else float("nan")
    final_err = target - final_pos
    final_abs_err = abs(final_err)
    overshoot = max(max_pos - target, 0.0)
    success = int(any(abs(target - p) <= tolerance for p in positions))

    return {
        "steps": len(positions),
        "mean_reward": float(np.mean(rewards)) if rewards else float("nan"),
        "final_pos": final_pos,
        "final_err": final_err,
        "final_abs_err": final_abs_err,
        "overshoot": overshoot,
        "success": success,
    }


def main():
    parser = argparse.ArgumentParser(description="Zero-shot distance generalization test for trained Stage-2 agent.")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--distances", type=str, default="1.0,2.0,3.0,5.0")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=1200)
    parser.add_argument("--stack-size", type=int, default=10)
    parser.add_argument("--tolerance", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=Path("benchmark_results/stage2_meta_rl_reproduction_v3/distance_generalization.csv"),
    )
    args = parser.parse_args()

    distances = [float(x.strip()) for x in args.distances.split(",") if x.strip()]

    # Build once using a reference env for network shape.
    ref_env = make_eval_env(args.stack_size, target_pos=distances[0], max_steps=args.max_steps)
    agent = Agent(ref_env)
    agent.load_state_dict(torch.load(args.model_path, map_location="cpu"))
    agent.eval()
    ref_env.close()

    rows = []
    for d in distances:
        env = make_eval_env(args.stack_size, target_pos=d, max_steps=args.max_steps)
        for ep in range(args.episodes):
            row = run_episode(env, agent, seed=args.seed * 10_000 + ep, tolerance=args.tolerance)
            row.update({"distance": d, "episode": ep})
            rows.append(row)
        env.close()

    df = pd.DataFrame(rows)
    summary = df.groupby("distance", as_index=False).agg(
        episodes=("episode", "count"),
        success_rate=("success", "mean"),
        mean_steps=("steps", "mean"),
        mean_reward=("mean_reward", "mean"),
        final_abs_error_mean=("final_abs_err", "mean"),
        overshoot_mean=("overshoot", "mean"),
        max_overshoot=("overshoot", "max"),
    )
    summary["success_rate"] *= 100.0

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_csv, index=False)
    summary_path = args.out_csv.with_name(args.out_csv.stem + "_summary.csv")
    summary.to_csv(summary_path, index=False)

    print("Raw:", args.out_csv)
    print("Summary:", summary_path)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()

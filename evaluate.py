import gymnasium as gym
import matplotlib.pyplot as plt
import mujoco
import numpy as np
import torch

from agents.domain_randomization import DomainRandomizationWrapper
from agents.model import Agent
from envs.adaptive_suspension import AdaptiveSuspensionEnv


def evaluate(model_path="meta_rl_agent.pth"):
    # Create Env
    env = AdaptiveSuspensionEnv()
    env = DomainRandomizationWrapper(env)
    env = gym.wrappers.FrameStackObservation(env, stack_size=10)

    # Load Agent
    agent = Agent(env)
    agent.load_state_dict(torch.load(model_path))
    agent.eval()

    # Eval Scenarios
    scenarios = [
        {"mass": 10.0, "friction": 1.0, "name": "Standard"},
        {"mass": 20.0, "friction": 0.2, "name": "Heavy & Slippery"},
        {"mass": 5.0, "friction": 2.0, "name": "Light & Grippy"},
    ]

    results = {}

    for sc in scenarios:
        # Start from a clean episode, then override the randomized physics.
        obs, _ = env.reset()
        model = env.unwrapped.model
        model.body_mass[1] = sc["mass"]
        model.geom_friction[0, 0] = sc["friction"]
        mujoco.mj_forward(model, env.unwrapped.data)
        print(f"Running Scenario: {sc['name']} (Mass={sc['mass']}, Friction={sc['friction']})")
        done = False

        # Logs
        positions = []
        errors = []
        gains = []

        while not done:
            with torch.no_grad():
                obs_tensor = torch.Tensor(obs).unsqueeze(0)
                action, _, _, _ = agent.get_action_and_value(obs_tensor)

            action_np = action.squeeze().numpy()
            obs, reward, done, _, info = env.step(action_np)

            positions.append(info["state"]["pos"])
            errors.append(env.unwrapped.target_pos - info["state"]["pos"])
            gains.append([info["gains"]["kp"], info["gains"]["ki"], info["gains"]["kd"]])

            if len(positions) > 500:
                break  # Safety break

        results[sc["name"]] = {"positions": positions, "errors": errors, "gains": np.array(gains)}

    # Plotting
    fig, axes = plt.subplots(3, 1, figsize=(10, 15))

    for name, data in results.items():
        axes[0].plot(data["positions"], label=name)
        axes[0].set_title("Position")
        axes[0].legend()

        axes[1].plot(data["errors"], label=name)
        axes[1].set_title("Tracking Error")

        # Plot only Kd for interest
        axes[2].plot(data["gains"][:, 2], label=f"{name} Kd")
        axes[2].set_title("Derivative Gain (Kd) Adaptation")
        axes[2].legend()

    plt.tight_layout()
    plt.savefig("evaluation_results.png")
    print("Evaluation saved to evaluation_results.png")


if __name__ == "__main__":
    evaluate()

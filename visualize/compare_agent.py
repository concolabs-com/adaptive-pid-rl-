import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt

from run_agent import build_env, apply_scenario, load_agent, SCENARIOS


def evaluate(model_path, obs_dims, scenario):

    class Args:
        pass

    args=Args()

    args.target=5.0
    args.hold_steps=25
    args.max_steps=500
    args.tolerance=0.05

    args.gain_base_kp=1.8
    args.gain_base_ki=0.7
    args.gain_base_kd=0.5

    args.gain_delta_kp=1.0
    args.gain_delta_ki=0.6
    args.gain_delta_kd=2.0

    args.approach_progress_cutoff_m=2.0
    args.near_approach_zone_m=2.0

    args.stack_size=10
    args.obs_dims=obs_dims

    env=build_env(args)

    mass=SCENARIOS[scenario]["mass"]
    friction=SCENARIOS[scenario]["friction"]

    apply_scenario(
        env.unwrapped,
        mass,
        friction
    )

    agent=load_agent(
        model_path,
        env
    )

    obs,_=env.reset()

    positions=[]
    velocities=[]

    done=False

    with torch.no_grad():

        while not done:

            obs_arr=np.asarray(
                obs,
                dtype=np.float32
            )

            obs_tensor=torch.from_numpy(
                obs_arr
            ).unsqueeze(0)

            action=agent.actor_mean(
                obs_tensor.reshape(1,-1)
            )

            action=torch.clamp(
                action,
                -1.0,
                1.0
            )

            action_np=action.squeeze(0).numpy()

            obs,reward,terminated,\
            truncated,info=env.step(
                action_np
            )

            done=terminated or truncated

            positions.append(
                info["state"]["pos"]
            )

            velocities.append(
                info["state"]["vel"]
            )

    env.close()

    target=5.0

    final_error=abs(
        target-positions[-1]
    )

    overshoot=max(positions)-target

    settled=final_error<0.05

    return {
        "pos":positions,
        "vel":velocities,
        "error":final_error,
        "overshoot":overshoot,
        "settled":settled
    }


def main():

    parser=argparse.ArgumentParser()

    parser.add_argument(
        "--scenario",
        choices=[
            "standard",
            "heavy",
            "light"
        ],
        default="standard"
    )

    args=parser.parse_args()

    stage5a_model=\
    "environment/agents/models/stage5a_context_cliff/meta_rl_agent.pth"

    stage5b_model=\
    "environment/agents/models/stage5b_blind_cliff/meta_rl_agent.pth"

    print("Running Stage5a...")

    stage5a=evaluate(
        stage5a_model,
        8,
        args.scenario
    )

    print("Running Stage5b...")

    stage5b=evaluate(
        stage5b_model,
        6,
        args.scenario
    )

    plt.figure(figsize=(10,5))

    plt.plot(
        stage5a["pos"],
        label="Stage5a"
    )

    plt.plot(
        stage5b["pos"],
        label="Stage5b"
    )

    plt.xlabel("Time Step")

    plt.ylabel("Position")

    plt.title(
        "Displacement Comparison"
    )

    plt.legend()

    plt.savefig(
        "comparison_displacement.png"
    )

    plt.close()

    plt.figure(figsize=(10,5))

    plt.plot(
        stage5a["vel"],
        label="Stage5a"
    )

    plt.plot(
        stage5b["vel"],
        label="Stage5b"
    )

    plt.xlabel("Time Step")

    plt.ylabel("Velocity")

    plt.title(
        "Velocity Comparison"
    )

    plt.legend()

    plt.savefig(
        "comparison_velocity.png"
    )

    plt.close()

    print()

    print(
        "MODEL | SCENARIO | ERROR | OVERSHOOT | SETTLED"
    )

    print(
        f"Stage5a | "
        f"{args.scenario} | "
        f"{stage5a['error']:.3f} | "
        f"{stage5a['overshoot']:.3f} | "
        f"{stage5a['settled']}"
    )

    print(
        f"Stage5b | "
        f"{args.scenario} | "
        f"{stage5b['error']:.3f} | "
        f"{stage5b['overshoot']:.3f} | "
        f"{stage5b['settled']}"
    )


if __name__=="__main__":
    main()
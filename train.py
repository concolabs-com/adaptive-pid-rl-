import random
import time

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from agents.domain_randomization import DomainRandomizationWrapper
from agents.model import Agent
from envs.adaptive_suspension import AdaptiveSuspensionEnv

# Hyperparameters
EXP_NAME = "MetaRL_GainScheduling"
SEED = 42
TOTAL_TIMESTEPS = 100000
LEARNING_RATE = 3e-4
NUM_ENVS = 4
NUM_STEPS = 2048
MINIBATCH_SIZE = 64
UPDATE_EPOCHS = 10
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_COEF = 0.2
ENT_COEF = 0.0
VF_COEF = 0.5
MAX_GRAD_NORM = 0.5


def make_env(seed, idx, capture_video=False, run_name=""):
    def thunk():
        env = AdaptiveSuspensionEnv()
        env = DomainRandomizationWrapper(env)
        # Frame Stacking for "Meta" Capability (Context)
        # Stack 10 frames to see error dynamics
        env = gym.wrappers.FrameStackObservation(env, stack_size=10)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if capture_video and idx == 0:
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    return thunk


if __name__ == "__main__":
    run_name = f"{EXP_NAME}_{SEED}_{int(time.time())}"

    # Seeding
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True

    # Vector Env
    envs = gym.vector.SyncVectorEnv(
        [make_env(SEED + i, i, capture_video=False, run_name=run_name) for i in range(NUM_ENVS)]
    )

    agent = Agent(envs)
    optimizer = optim.Adam(agent.parameters(), lr=LEARNING_RATE, eps=1e-5)

    # Storage
    obs = torch.zeros((NUM_STEPS, NUM_ENVS) + envs.single_observation_space.shape)
    actions = torch.zeros((NUM_STEPS, NUM_ENVS) + envs.single_action_space.shape)
    logprobs = torch.zeros((NUM_STEPS, NUM_ENVS))
    rewards = torch.zeros((NUM_STEPS, NUM_ENVS))
    dones = torch.zeros((NUM_STEPS, NUM_ENVS))
    values = torch.zeros((NUM_STEPS, NUM_ENVS))

    # Start
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=SEED)
    next_obs = torch.Tensor(next_obs)
    next_done = torch.zeros(NUM_ENVS)
    num_updates = TOTAL_TIMESTEPS // (NUM_STEPS * NUM_ENVS)

    for update in tqdm(range(1, num_updates + 1)):
        # Annealing
        frac = 1.0 - (update - 1.0) / num_updates
        lrnow = frac * LEARNING_RATE
        optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, NUM_STEPS):
            global_step += 1 * NUM_ENVS
            obs[step] = next_obs
            dones[step] = next_done

            # Action Logic
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # Execute
            next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            next_done = np.logical_or(terminations, truncations)
            rewards[step] = torch.tensor(reward).view(-1)
            next_obs = torch.Tensor(next_obs)
            next_done = torch.Tensor(next_done)

        # Bootstrap value
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards)
            lastgaelam = 0
            for t in reversed(range(NUM_STEPS)):
                if t == NUM_STEPS - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + GAMMA * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + GAMMA * GAE_LAMBDA * nextnonterminal * lastgaelam
            returns = advantages + values

        # Flatten
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimize
        b_inds = np.arange(NUM_STEPS * NUM_ENVS)
        for epoch in range(UPDATE_EPOCHS):
            np.random.shuffle(b_inds)
            for start in range(0, NUM_STEPS * NUM_ENVS, MINIBATCH_SIZE):
                end = start + MINIBATCH_SIZE
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # Calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs = [((ratio - 1.0).abs() > CLIP_COEF).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - CLIP_COEF, 1 + CLIP_COEF)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - ENT_COEF * entropy_loss + VF_COEF * v_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), MAX_GRAD_NORM)
                optimizer.step()

    # Save Model
    torch.save(agent.state_dict(), "meta_rl_agent.pth")
    print("Training Complete. Model saved.")

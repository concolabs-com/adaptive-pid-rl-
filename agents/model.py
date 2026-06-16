import numpy as np
import torch
import torch.nn as nn
from torch.distributions.normal import Normal


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, envs, recurrent: bool = False, recurrent_hidden_size: int = 128):
        super().__init__()
        if hasattr(envs, "single_observation_space"):
            obs_shape = np.prod(envs.single_observation_space.shape)
            action_shape = np.prod(envs.single_action_space.shape)
        else:
            obs_shape = np.prod(envs.observation_space.shape)
            action_shape = np.prod(envs.action_space.shape)

        self.recurrent = bool(recurrent)
        self.hidden_size = int(recurrent_hidden_size)

        if self.recurrent:
            self.encoder = nn.Sequential(
                layer_init(nn.Linear(obs_shape, self.hidden_size)),
                nn.Tanh(),
            )
            self.gru = nn.GRUCell(self.hidden_size, self.hidden_size)
            self.actor_mean = nn.Sequential(
                layer_init(nn.Linear(self.hidden_size, self.hidden_size)),
                nn.Tanh(),
                layer_init(nn.Linear(self.hidden_size, action_shape), std=0.01),
            )
            self.critic = nn.Sequential(
                layer_init(nn.Linear(self.hidden_size, self.hidden_size)),
                nn.Tanh(),
                layer_init(nn.Linear(self.hidden_size, 1), std=1.0),
            )
        else:
            self.critic = nn.Sequential(
                layer_init(nn.Linear(obs_shape, 64)),
                nn.Tanh(),
                layer_init(nn.Linear(64, 64)),
                nn.Tanh(),
                layer_init(nn.Linear(64, 1), std=1.0),
            )
            self.actor_mean = nn.Sequential(
                layer_init(nn.Linear(obs_shape, 64)),
                nn.Tanh(),
                layer_init(nn.Linear(64, 64)),
                nn.Tanh(),
                layer_init(nn.Linear(64, action_shape), std=0.01),
            )

        self.actor_logstd = nn.Parameter(torch.full((1, action_shape), -1.0))

    def get_initial_state(self, batch_size: int, device: torch.device):
        if not self.recurrent:
            return None
        return torch.zeros((batch_size, self.hidden_size), device=device)

    def _flatten_obs(self, x):
        return x.reshape(x.shape[0], -1)

    def _recurrent_features(self, x, hidden_state, done=None):
        x = self._flatten_obs(x)
        if done is not None:
            hidden_state = hidden_state * (1.0 - done.float().unsqueeze(-1))
        encoded = self.encoder(x)
        next_hidden = self.gru(encoded, hidden_state)
        return next_hidden

    def get_value(self, x, hidden_state=None, done=None):
        if not self.recurrent:
            x = x.reshape(x.shape[0], -1)
            return self.critic(x)

        if hidden_state is None:
            raise ValueError("hidden_state is required for recurrent value evaluation")
        next_hidden = self._recurrent_features(x, hidden_state, done)
        return self.critic(next_hidden)

    def get_action_and_value(self, x, action=None, hidden_state=None, done=None, deterministic: bool = False):
        if not self.recurrent:
            x = x.reshape(x.shape[0], -1)
            action_mean = self.actor_mean(x)
            action_logstd = self.actor_logstd.expand_as(action_mean)
            action_std = torch.exp(action_logstd)
            probs = Normal(action_mean, action_std)
            if action is None:
                action = action_mean if deterministic else probs.sample()
            return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)

        if hidden_state is None:
            raise ValueError("hidden_state is required for recurrent action evaluation")

        next_hidden = self._recurrent_features(x, hidden_state, done)
        action_mean = self.actor_mean(next_hidden)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = action_mean if deterministic else probs.sample()
        return (
            action,
            probs.log_prob(action).sum(1),
            probs.entropy().sum(1),
            self.critic(next_hidden),
            next_hidden,
        )

    def evaluate_sequence(self, obs, actions, dones, initial_hidden):
        if not self.recurrent:
            raise ValueError("evaluate_sequence is only available for recurrent agents")

        obs = obs.reshape(obs.shape[0], obs.shape[1], -1)
        hidden = initial_hidden
        logprobs = []
        entropies = []
        values = []

        for t in range(obs.shape[0]):
            hidden = self._recurrent_features(obs[t], hidden, dones[t])
            action_mean = self.actor_mean(hidden)
            action_logstd = self.actor_logstd.expand_as(action_mean)
            action_std = torch.exp(action_logstd)
            probs = Normal(action_mean, action_std)
            logprobs.append(probs.log_prob(actions[t]).sum(1))
            entropies.append(probs.entropy().sum(1))
            values.append(self.critic(hidden).squeeze(-1))

        return (
            torch.stack(logprobs, dim=0),
            torch.stack(entropies, dim=0),
            torch.stack(values, dim=0),
            hidden,
        )

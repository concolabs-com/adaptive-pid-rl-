import numpy as np
import torch
import torch.nn as nn
from torch.distributions.normal import Normal


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        # Observation space: 6 (Pos, Vel, Error, PrevAction*3)
        # Action space: 3 (Kp, Ki, Kd)

        # LSTM Support
        self.lstm_hidden_size = 64
        self.network = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, self.lstm_hidden_size)),
            nn.Tanh(),
        )
        self.lstm = nn.LSTM(self.lstm_hidden_size, self.lstm_hidden_size)

        for name, param in self.lstm.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0)
            elif "weight" in name:
                nn.init.orthogonal_(param, 1.0)

        self.actor = layer_init(nn.Linear(self.lstm_hidden_size, np.prod(envs.single_action_space.shape)), std=0.01)
        self.critic = layer_init(nn.Linear(self.lstm_hidden_size, 1), std=1)

        self.actor_logstd = nn.Parameter(torch.zeros(1, np.prod(envs.single_action_space.shape)))

    def get_states(self, x, lstm_state, done):
        hidden = self.network(x)

        # Batch LSTM logic:
        # x is (seq_len, batch_size, input_size) or (batch_size, input_size)
        # We assume flattening happens outside or here.

        # Reshape for LSTM: (1, batch, hidden) if not sequence
        batch_size = lstm_state[0].shape[1]
        hidden = hidden.reshape((-1, batch_size, self.lstm_hidden_size))

        # Reset LSTM state where done is True
        # For simplicity in this implementation, we handle done masking outside or rely on short sequences
        # But for PPO rollout, we typically carry state.

        # Standard PPO LSTM forward
        # hidden shape: (seq_len, batch, features)
        output, (h_n, c_n) = self.lstm(hidden, lstm_state)
        return output, (h_n, c_n)

    def get_value(self, x, lstm_state, done):
        hidden, _ = self.get_states(x, lstm_state, done)
        return self.critic(hidden)

    def get_action_and_value(self, x, lstm_state, done, action=None):
        hidden, next_lstm_state = self.get_states(x, lstm_state, done)

        # Flatten for heads
        hidden = hidden.reshape(-1, self.lstm_hidden_size)

        action_mean = self.actor(hidden)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)

        if action is None:
            action = probs.sample()

        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(hidden), next_lstm_state

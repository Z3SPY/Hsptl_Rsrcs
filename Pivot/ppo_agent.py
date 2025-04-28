# ppo_agent.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

class PPOActorCritic(nn.Module):
    def __init__(self, obs_shape, n_actions):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Flatten(),
            nn.Linear(obs_shape[0]*obs_shape[1]*obs_shape[2], 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )
        self.policy_head = nn.Linear(128, n_actions)
        self.value_head = nn.Linear(128, 1)

    def forward(self, x):
        x = self.shared(x)
        policy_logits = self.policy_head(x)
        state_value = self.value_head(x)
        return policy_logits, state_value

class PPOMemory:
    def __init__(self):
        self.clear()

    def store(self, obs, action, log_prob, reward, done, value):
        self.observations.append(obs)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.dones.append(done)
        self.values.append(value)

    def clear(self):
        self.observations = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.dones = []
        self.values = []

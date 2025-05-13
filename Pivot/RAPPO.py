"""
Risk-Adaptive Proximal Policy Optimization (RA-PPO)
Generalized Implementation Framework

This is a modular PyTorch-based PPO implementation with risk-sensitive extensions:
1. Risk-aware reward shaping using CVaR-style quantile penalties.
2. Optional action masking for constrained action spaces.

It can be applied across Gym environments with discrete or multi-discrete actions.
Designed for environments with episodic structure and stochasticity.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

import torch.nn.functional as F
from torch.distributions import Normal


class ContinuousPolicyNetwork(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes=[128, 128]):
        super().__init__()
        layers = []
        last_size = obs_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(last_size, h))
            layers.append(nn.ReLU())
            last_size = h
        self.shared = nn.Sequential(*layers)
        self.mean_head = nn.Linear(last_size, act_dim)
        self.value_head = nn.Linear(last_size, 1)

        # log_std as a trainable parameter (shared across dimensions)
        self.log_std = nn.Parameter(torch.zeros(act_dim))

    def forward(self, x):
        x = self.shared(x)
        mean = self.mean_head(x)
        std = torch.exp(self.log_std)
        value = self.value_head(x).squeeze(-1)
        return mean, std, value


class RA_PPO_Agent:
    def __init__(self, env, risk_alpha=0.05, beta=-1.0, clip_eps=0.2, lr=3e-4, gamma=0.99, lam=0.95):
        self.env = env
        self.obs_dim = env.observation_space.shape[0]
        self.act_dim = env.action_space.shape[0]
        self.model = ContinuousPolicyNetwork(self.obs_dim, self.act_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

        self.gamma = gamma
        self.lam = lam
        self.clip_eps = clip_eps
        self.risk_alpha = risk_alpha
        self.beta = beta

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def get_action(self, obs):
        obs = torch.FloatTensor(obs).to(self.device)
        mean, std, _ = self.model(obs)
        dist = Normal(mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        action_clipped = torch.clamp(action, 0.0, 1.0)  # clip within action bounds
        return action_clipped.cpu().numpy(), log_prob.item()

    def compute_gae(self, rewards, values, dones, last_value):
        gae = 0
        returns = []
        values = values + [last_value]
        for step in reversed(range(len(rewards))):
            delta = rewards[step] + self.gamma * values[step + 1] * (1 - dones[step]) - values[step]
            gae = delta + self.gamma * self.lam * (1 - dones[step]) * gae
            returns.insert(0, gae + values[step])
        return returns

    def risk_shape_rewards(self, episode_returns):
        threshold = np.quantile(episode_returns, self.risk_alpha)
        shaped_rewards = []
        for R in episode_returns:
            penalty = self.beta if R < threshold else 0
            shaped_rewards.append(R + penalty)
        return shaped_rewards, threshold

    def update(self, batch):
        obs = torch.FloatTensor(batch['obs']).to(self.device)
        actions = torch.FloatTensor(batch['actions']).to(self.device)
        old_log_probs = torch.FloatTensor(batch['log_probs']).to(self.device)
        returns = torch.FloatTensor(batch['returns']).to(self.device)
        advantages = torch.FloatTensor(batch['advantages']).to(self.device)

        for _ in range(4):  # PPO epochs
            mean, std, values = self.model(obs)
            dist = Normal(mean, std)
            log_probs = dist.log_prob(actions).sum(axis=-1)
            entropy = dist.entropy().sum(axis=-1).mean()

            ratio = torch.exp(log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = F.mse_loss(values, returns)
            loss = policy_loss + 0.5 * value_loss - 0.01 * entropy

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def train(self, total_episodes=1000):
        all_episode_returns = []
        for ep in range(total_episodes):
            obs = self.env.reset()
            done = False
            rewards, values, log_probs, actions, states, dones = [], [], [], [], [], []

            while not done:
                action, log_prob = self.get_action(obs)
                next_obs, reward, done, _ = self.env.step(action)

                _, _, value = self.model(torch.FloatTensor(obs).to(self.device))
                value = value.item()

                states.append(obs)
                actions.append(action)
                rewards.append(reward)
                values.append(value)
                log_probs.append(log_prob)
                dones.append(done)

                obs = next_obs

            _, _, next_value = self.model(torch.FloatTensor(obs).to(self.device))
            returns = self.compute_gae(rewards, values, dones, next_value.item())
            total_return = sum(rewards)
            all_episode_returns.append(total_return)

            shaped_returns, threshold = self.risk_shape_rewards(all_episode_returns[-100:])

            batch = {
                'obs': states,
                'actions': actions,
                'log_probs': log_probs,
                'returns': shaped_returns,
                'advantages': list(np.array(shaped_returns) - np.array(values))
            }
            self.update(batch)

            print(f"[EP {ep+1}] Return: {total_return:.2f}, Risk Threshold: {threshold:.2f}")


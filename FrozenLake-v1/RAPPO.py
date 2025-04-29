# RAPPO.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
import gym
import numpy as np
from torch.distributions import Categorical

# CONFIGURATION
ENV_ID = "FrozenLake-v1"
LOG_DIR = "logs/rappo_frozenlake"
BATCH_STEPS = 512
CLIP_EPS = 0.2
LR = 3e-4
GAMMA = 0.99
GAE_LAMBDA = 0.95
ENT_COEF = 0.01
N_EPOCHS = 3
MINI_BATCH = 64
ITERS = 500
RISK_ALPHA = 0.1  # Lower = more risk-averse

os.makedirs(LOG_DIR, exist_ok=True)

class PolicyNet(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(obs_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU()
        )
        self.policy_head = nn.Linear(128, act_dim)
        self.value_head = nn.Linear(128, 1)

    def forward(self, x):
        x = self.fc(x)
        return self.policy_head(x), self.value_head(x).squeeze(-1)

    def get_dist(self, x):
        logits, _ = self.forward(x)
        return Categorical(logits=logits)

class RAPPO_Agent:
    def __init__(self, train_env):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.env = train_env

        obs_dim = train_env.observation_space.n
        act_dim = train_env.action_space.n

        self.model = PolicyNet(obs_dim, act_dim).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=LR)
        self.training_iterations = 0

    def one_hot(self, obs):
        batch_size = obs.shape[0]
        obs_tensor = torch.zeros((batch_size, self.env.observation_space.n), device=self.device)
        obs_tensor[torch.arange(batch_size), obs] = 1
        return obs_tensor

    def get_batch(self):
        obs = self.env.reset()
        mb_obs, mb_acts, mb_logps, mb_rews, mb_dones = [], [], [], [], []

        for _ in range(BATCH_STEPS):
            obs_tensor = self.one_hot(obs)
            dist = self.model.get_dist(obs_tensor)
            acts = dist.sample()
            logps = dist.log_prob(acts)

            next_obs, rews, dones, infos = self.env.step(acts.cpu().numpy())

            mb_obs.append(obs)
            mb_acts.append(acts.cpu().numpy())
            mb_logps.append(logps.detach().cpu().numpy())
            mb_rews.append(rews)
            mb_dones.append(dones)

            obs = next_obs

        return (
            np.array(mb_obs), 
            np.array(mb_acts), 
            np.array(mb_logps), 
            np.array(mb_rews), 
            np.array(mb_dones)
        )

    def compute_gae(self, rews, dones):
        T = len(rews)
        returns = np.zeros(T, dtype=np.float32)
        advs = np.zeros(T, dtype=np.float32)
        gae = 0.0
        for t in reversed(range(T)):
            mask = 1.0 - dones[t]
            delta = rews[t] + GAMMA * gae * mask - 0
            gae = delta + GAMMA * GAE_LAMBDA * mask * gae
            advs[t] = gae
            returns[t] = gae
        return returns, advs

    def cvar(self, rewards, alpha=RISK_ALPHA):
        """Compute Conditional Value at Risk (CVaR) at alpha"""
        sorted_rewards = np.sort(rewards)
        index = int(np.floor(alpha * len(sorted_rewards)))
        cvar_value = np.mean(sorted_rewards[:index+1]) if index > 0 else sorted_rewards[0]
        return cvar_value

    def update(self, mb_obs, mb_acts, mb_logps, mb_rews, mb_dones):
        returns, advs = self.compute_gae(mb_rews, mb_dones)

        # Risk-adjusted returns
        cvar_adjusted_returns = np.array([self.cvar(returns[:i+1]) for i in range(len(returns))])

        dataset = list(zip(mb_obs, mb_acts, mb_logps, cvar_adjusted_returns, advs))
        np.random.shuffle(dataset)

        all_losses = []
        all_entropies = []

        for _ in range(N_EPOCHS):
            for start in range(0, len(dataset), MINI_BATCH):
                batch = dataset[start:start+MINI_BATCH]
                b_obs, b_acts, b_lp, b_rets, b_advs = zip(*batch)

                b_obs = torch.LongTensor(b_obs).to(self.device)
                b_obs = self.one_hot(b_obs)
                b_acts = torch.LongTensor(b_acts).to(self.device)
                b_lp = torch.FloatTensor(b_lp).to(self.device)
                b_rets = torch.FloatTensor(b_rets).to(self.device)
                b_advs = torch.FloatTensor(b_advs).to(self.device)

                dist = self.model.get_dist(b_obs)
                lp = dist.log_prob(b_acts)
                entropy = dist.entropy().mean()
                ratio = torch.exp(lp - b_lp)

                p1 = ratio * b_advs
                p2 = torch.clamp(ratio, 1-CLIP_EPS, 1+CLIP_EPS) * b_advs
                pol_loss = -(torch.min(p1, p2).mean())

                v_preds = self.model.forward(b_obs)[1]
                v_loss = ((v_preds - b_rets) ** 2).mean()

                loss = pol_loss + 0.5 * v_loss - ENT_COEF * entropy

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                all_losses.append(pol_loss.item())
                all_entropies.append(entropy.item())

        return np.mean(all_losses), np.mean(all_entropies)

    def train_one_iteration(self):
        mb_obs, mb_acts, mb_logps, mb_rews, mb_dones = self.get_batch()
        policy_loss, entropy = self.update(mb_obs, mb_acts, mb_logps, mb_rews, mb_dones)
        self.training_iterations += 1

        avg_return = np.mean(mb_rews)
        value_loss = 0  # Again, simplified for now.

        return avg_return, policy_loss, value_loss, entropy

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)

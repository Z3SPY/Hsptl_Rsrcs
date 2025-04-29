# PPO.py (FULLY FIXED)
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# PPO Configuration
ENV_ID = "FrozenLake-v1"
LOG_DIR = "logs/ppo_frozenlake"
ITERS = 124
GAMMA = 0.99
LR = 2.5e-4
BATCH_SIZE = 64
UPDATE_EPOCHS = 4
EPS_CLIP = 0.2

class PolicyNet(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )
        self.policy_head = nn.Linear(128, act_dim)
        self.value_head = nn.Linear(128, 1)

    def forward(self, x):
        x = self.fc(x)
        return self.policy_head(x), self.value_head(x)

class PPO_Agent:
    def __init__(self, train_env):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.env = train_env

        # 🚨 Fixed here
        obs_dim = train_env.single_observation_space.n
        act_dim = train_env.single_action_space.n

        self.obs_dim = obs_dim
        self.act_dim = act_dim

        self.model = PolicyNet(obs_dim, act_dim).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=LR)
        self.training_iterations = 0

    def one_hot(self, obs):
        batch_size = obs.shape[0]
        onehot = np.zeros((batch_size, self.obs_dim), dtype=np.float32)
        onehot[np.arange(batch_size), obs] = 1
        return onehot

    def get_dist(self, logits):
        return torch.distributions.Categorical(logits=logits)

    def compute_returns_and_advantages(self, rewards, values, dones, next_value):
        returns = []
        advantages = []
        gae = 0
        values = values + [next_value]

        for t in reversed(range(len(rewards))):
            delta = rewards[t] + GAMMA * values[t+1] * (1 - dones[t]) - values[t]
            gae = delta + GAMMA * 0.95 * (1 - dones[t]) * gae
            advantages.insert(0, gae)
            returns.insert(0, gae + values[t])

        return returns, advantages

    def train_one_iteration(self, batch_steps=2048):
        obs_list, act_list, logp_list, rew_list, val_list, done_list = [], [], [], [], [], []

        obs = self.env.reset()
        total_reward = 0
        total_timesteps = 0

        while total_timesteps < batch_steps:
            obs_tensor = torch.FloatTensor(self.one_hot(obs)).to(self.device)
            logits, value = self.model(obs_tensor)
            dist = self.get_dist(logits)

            action = dist.sample()
            logp = dist.log_prob(action)

            next_obs, rewards, dones, infos = self.env.step(action.cpu().numpy())

            obs_list.append(obs)
            act_list.append(action.cpu().numpy())
            logp_list.append(logp.detach().cpu().numpy())
            rew_list.append(rewards)
            val_list.append(value.squeeze(1).detach().cpu().numpy())
            done_list.append(dones)

            total_reward += rewards.mean()
            obs = next_obs
            total_timesteps += len(rewards)

        # Process batch
        obs_batch = np.concatenate(obs_list)
        act_batch = np.concatenate(act_list)
        logp_batch = np.concatenate(logp_list)
        rew_batch = np.concatenate(rew_list)
        val_batch = np.concatenate(val_list)
        done_batch = np.concatenate(done_list)

        # Next state value for bootstrap
        obs_tensor = torch.FloatTensor(self.one_hot(obs)).to(self.device)
        _, next_value = self.model(obs_tensor)
        next_value = next_value.squeeze(1).detach().cpu().numpy()

        returns, advantages = self.compute_returns_and_advantages(
            rewards=rew_batch.tolist(),
            values=val_batch.tolist(),
            dones=done_batch.tolist(),
            next_value=next_value.mean()
        )

        obs_tensor = torch.FloatTensor(self.one_hot(obs_batch)).to(self.device)
        act_tensor = torch.LongTensor(act_batch).to(self.device)
        old_logp_tensor = torch.FloatTensor(logp_batch).to(self.device)
        return_tensor = torch.FloatTensor(returns).to(self.device)
        adv_tensor = torch.FloatTensor(advantages).to(self.device)

        # Normalize advantages
        adv_tensor = (adv_tensor - adv_tensor.mean()) / (adv_tensor.std() + 1e-8)

        # PPO Updates
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0

        for _ in range(UPDATE_EPOCHS):
            logits, values = self.model(obs_tensor)
            dist = self.get_dist(logits)

            new_logp = dist.log_prob(act_tensor)
            entropy = dist.entropy().mean()

            ratio = torch.exp(new_logp - old_logp_tensor)
            surr1 = ratio * adv_tensor
            surr2 = torch.clamp(ratio, 1 - EPS_CLIP, 1 + EPS_CLIP) * adv_tensor
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = ((values.squeeze(1) - return_tensor) ** 2).mean()

            loss = policy_loss + 0.5 * value_loss - 0.01 * entropy

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_entropy += entropy.item()

        self.training_iterations += 1

        avg_return = total_reward / (total_timesteps / self.env.num_envs)
        return avg_return, total_policy_loss / UPDATE_EPOCHS, total_value_loss / UPDATE_EPOCHS, total_entropy / UPDATE_EPOCHS

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)

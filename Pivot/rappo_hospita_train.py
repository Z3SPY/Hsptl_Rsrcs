import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

class MultiPolicyNetwork(nn.Module):
    def __init__(self, obs_dim, act_dims, hidden_sizes=[128, 128]):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden_sizes[0]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.ReLU()
        )
        # One head per unit (for MultiDiscrete)
        self.policy_heads = nn.ModuleList([
            nn.Linear(hidden_sizes[-1], dim) for dim in act_dims
        ])
        self.value_head = nn.Linear(hidden_sizes[-1], 1)

    def forward(self, x):
        x = self.shared(x)
        logits = [head(x) for head in self.policy_heads]
        value = self.value_head(x).squeeze(-1)
        return logits, value

class RAPPOAgent:
    def __init__(self, env, risk_alpha=0.05, beta=-1.0, clip_eps=0.2, lr=3e-4, gamma=0.99, lam=0.95):
        self.env = env
        self.obs_dim = env.observation_space.shape[0]
        self.act_dims = env.action_space.nvec.tolist()  # for MultiDiscrete
        self.model = MultiPolicyNetwork(self.obs_dim, self.act_dims)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

        self.gamma = gamma
        self.lam = lam
        self.clip_eps = clip_eps
        self.risk_alpha = risk_alpha
        self.beta = beta

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def get_action(self, obs):
        obs = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        logits, _ = self.model(obs)
        dists = [Categorical(logits=logit) for logit in logits]
        actions = [dist.sample() for dist in dists]
        log_probs = [dist.log_prob(action) for dist, action in zip(dists, actions)]
        action = torch.stack(actions).cpu().numpy()
        log_prob = torch.stack(log_probs).sum().item()
        return action, log_prob

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
        actions = torch.LongTensor(batch['actions']).to(self.device)
        old_log_probs = torch.FloatTensor(batch['log_probs']).to(self.device)
        returns = torch.FloatTensor(batch['returns']).to(self.device)
        advantages = torch.FloatTensor(batch['advantages']).to(self.device)

        for _ in range(4):
            logits, values = self.model(obs)
            dists = [Categorical(logits=logit) for logit in logits]
            log_probs = torch.stack([
                dist.log_prob(action) for dist, action in zip(dists, actions.t())
            ], dim=1).sum(axis=1)
            entropy = torch.stack([dist.entropy() for dist in dists], dim=1).sum(axis=1).mean()

            ratio = torch.exp(log_probs - old_log_probs)

            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = ((returns - values) ** 2).mean()
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

                _, value = self.model(torch.FloatTensor(obs).unsqueeze(0).to(self.device))
                value = value.item()

                states.append(obs)
                actions.append(action)
                rewards.append(reward)
                values.append(value)
                log_probs.append(log_prob)
                dones.append(done)

                obs = next_obs

            _, next_value = self.model(torch.FloatTensor(obs).unsqueeze(0).to(self.device))
            returns = self.compute_gae(rewards, values, dones, next_value.item())
            total_return = sum(rewards)
            all_episode_returns.append(total_return)

            shaped_returns, threshold = self.risk_shape_rewards(all_episode_returns[-100:])

            batch = {
                'obs': states,
                'actions': actions,
                'log_probs': log_probs,
                'returns': returns,
                'advantages': list(np.array(returns) - np.array(values))
            }
            self.update(batch)

            print(f"Episode {ep+1}, Return: {total_return:.2f}, Risk Threshold: {threshold:.2f}")

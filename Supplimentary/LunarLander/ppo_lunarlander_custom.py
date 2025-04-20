
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gym
from torch.distributions import Categorical

if not hasattr(np, "bool8"):
    np.bool8 = bool

class PolicyNet(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU()
        )
        self.policy_head = nn.Linear(128, act_dim)
        self.value_head = nn.Linear(128, 1)

    def forward(self, x):
        x = self.net(x)
        return self.policy_head(x), self.value_head(x).squeeze(-1)

class PPOAgent:
    def __init__(self, env, clip_eps=0.2, gamma=0.99, lam=0.95, lr=3e-4):
        self.env = env
        self.gamma = gamma
        self.lam = lam
        self.clip_eps = clip_eps
        self.obs_dim = env.observation_space.shape[0]
        self.act_dim = env.action_space.n
        self.model = PolicyNet(self.obs_dim, self.act_dim).to('cpu')
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

    def get_action(self, obs):
        obs = torch.FloatTensor(obs)
        logits, _ = self.model(obs)
        dist = Categorical(logits=logits)
        a = dist.sample()
        return a.item(), dist.log_prob(a).item()

    def compute_gae(self, rewards, values, dones, last_value):
        values = values + [last_value]
        gae, returns = 0, []
        for t in reversed(range(len(rewards))):
            mask = 1 - dones[t]
            delta = rewards[t] + self.gamma * values[t+1] * mask - values[t]
            gae = delta + self.gamma * self.lam * mask * gae
            returns.insert(0, gae + values[t])
        return returns

    def update(self, batch):
        obs = torch.FloatTensor(np.vstack(batch['obs']))
        acts = torch.LongTensor(batch['actions'])
        old_logp = torch.FloatTensor(batch['log_probs'])
        rets = torch.FloatTensor(batch['returns'])
        advs = torch.FloatTensor(batch['advantages'])
        advs = (advs - advs.mean()) / (advs.std() + 1e-8)
        for _ in range(4):
            logits, vals = self.model(obs)
            dist = Categorical(logits=logits)
            logp = dist.log_prob(acts)
            ratio = torch.exp(logp - old_logp)
            s1 = ratio * advs
            s2 = torch.clamp(ratio, 1-self.clip_eps, 1+self.clip_eps) * advs
            p_loss = -torch.min(s1, s2).mean()
            v_loss = (rets - vals).pow(2).mean()
            ent = dist.entropy().mean()
            loss = p_loss + 0.5*v_loss - 0.01*ent
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def train(self, episodes=500):
        for ep in range(episodes):
            obs, _ = self.env.reset()
            done = False
            obs_buf, act_buf, logp_buf, rew_buf, val_buf, done_buf = [], [], [], [], [], []
            while not done:
                a, logp = self.get_action(obs)
                next_obs, r, done, _, _ = self.env.step(a)
                obs_buf.append(obs)
                act_buf.append(a)
                logp_buf.append(logp)
                rew_buf.append(r)
                with torch.no_grad():
                    _, v = self.model(torch.FloatTensor(obs))
                val_buf.append(v.item())
                done_buf.append(done)
                obs = next_obs
            with torch.no_grad():
                _, last_v = self.model(torch.FloatTensor(obs))
            returns = self.compute_gae(rew_buf, val_buf, done_buf, last_v.item())
            advantages = np.array(returns) - np.array(val_buf)
            batch = {'obs':obs_buf, 'actions':act_buf, 'log_probs':logp_buf,
                     'returns':returns, 'advantages':advantages.tolist()}
            self.update(batch)
            print(f"PPO Episode {ep+1}, Return: {sum(rew_buf):.2f}")

if __name__ == '__main__':
    env = gym.make("LunarLander-v2")
    agent = PPOAgent(env)
    agent.train(episodes=500)
    env.close()

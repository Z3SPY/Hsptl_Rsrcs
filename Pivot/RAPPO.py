#!/usr/bin/env python3
"""
RA-PPO training script for HospitalSimEnv
- Handles MultiDiscrete actions with independent Categorical heads
- Risk‑adaptive reward shaping via CVaR penalty
- TensorBoard and CSV logging
- Includes fix to avoid multiple backward calls
"""
import os
import csv
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter
from hospital_env import HospitalSimEnv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PolicyValueNet(nn.Module):
    def __init__(self, obs_dim, action_nvec, hidden_sizes=(128,128)):
        super().__init__()
        layers = []
        last = obs_dim
        for h in hidden_sizes:
            layers += [nn.Linear(last, h), nn.ReLU()]
            last = h
        self.shared = nn.Sequential(*layers)
        self.policy_heads = nn.ModuleList([
            nn.Linear(last, n) for n in action_nvec
        ])
        self.value_head = nn.Linear(last, 1)

    def forward(self, x):
        x = self.shared(x)
        logits = [head(x) for head in self.policy_heads]
        value = self.value_head(x).squeeze(-1)
        return logits, value

class RAPPOAgent:
    def __init__(self, env, risk_alpha=0.05, beta=-1.0,
                 clip_eps=0.2, lr=3e-4, gamma=0.99, lam=0.95):
        self.env = env
        obs_dim = env.observation_space.shape[0]
        action_nvec = env.action_space.nvec
        self.model = PolicyValueNet(obs_dim, action_nvec).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.gamma = gamma
        self.lam = lam
        self.clip_eps = clip_eps
        self.risk_alpha = risk_alpha
        self.beta = beta

    def select_action(self, obs):
        obs_t = torch.FloatTensor(obs).to(device)
        logits, value = self.model(obs_t)
        actions = []
        logp = 0
        entropy = 0
        for logit in logits:
            dist = Categorical(logits=logit)
            a = dist.sample()
            actions.append(a.item())
            logp   += dist.log_prob(a)
            entropy+= dist.entropy()
        return np.array(actions, dtype=int), logp, entropy, value

    def compute_gae(self, rewards, values, dones, last_val):
        gae = 0
        returns = []
        vals = values + [last_val]
        for i in reversed(range(len(rewards))):
            delta = rewards[i] + self.gamma * vals[i+1] * (1-dones[i]) - vals[i]
            gae = delta + self.gamma * self.lam * (1-dones[i]) * gae
            returns.insert(0, gae + vals[i])
        return returns

    def update(self, batch):
        obs = torch.FloatTensor(batch['obs']).to(device)
        acts = torch.LongTensor(batch['actions']).to(device)
        old_logp = torch.stack(batch['logps']).detach().to(device)
        returns = torch.FloatTensor(batch['returns']).to(device)
        advs    = torch.FloatTensor(batch['advs']).to(device)

        policy_loss_total = 0
        value_loss_total = 0
        entropy_loss_total = 0

        for _ in range(4):
            logits, vals = self.model(obs)
            logp_new = []
            entropy_new = 0
            for i, logit in enumerate(logits):
                dist = Categorical(logits=logit)
                logp_new.append(dist.log_prob(acts[:,i]))
                entropy_new += dist.entropy().mean()
            logp_new = torch.stack(logp_new, dim=1).sum(dim=1)
            ratio = torch.exp(logp_new - old_logp)
            s1 = ratio * advs
            s2 = torch.clamp(ratio, 1-self.clip_eps, 1+self.clip_eps) * advs
            policy_loss = -torch.min(s1,s2).mean()
            value_loss = (returns - vals).pow(2).mean()
            entropy_loss = -entropy_new
            loss = policy_loss + 0.5*value_loss + 0.01*entropy_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            policy_loss_total += policy_loss.item()
            value_loss_total  += value_loss.item()
            entropy_loss_total+= entropy_loss.item()

        return policy_loss_total/4, value_loss_total/4, entropy_loss_total/4

def train_rappo(num_episodes=1000, out_dir="./rappo_logs"):
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "episode_returns.csv")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['episode','return','length','risk_threshold'])
    writer = SummaryWriter(log_dir=os.path.join(out_dir,'tb'))

    env = HospitalSimEnv(sim_config={
        'n_triage':2,'n_reg':2,'n_exam':3,'n_trauma':2,
        'n_cubicles_1':3,'n_cubicles_2':2,'n_ward':10,'n_icu':5,
        'prob_trauma':0.12
    })
    agent = RAPPOAgent(env)

    all_returns = []
    for ep in range(1, num_episodes+1):
        obs = env.reset()
        done = False
        rewards, values, logps, entropys, actions, dones, obs_buf = [],[],[],[],[],[],[]
        total_ret = 0
        steps = 0
        while not done:
            obs_buf.append(obs)
            act, logp, ent, val = agent.select_action(obs)
            next_obs, reward, done, _ = env.step(act)
            rewards.append(reward)
            values.append(val.item())
            logps.append(logp.detach())
            entropys.append(ent.detach())
            actions.append(torch.tensor(act))
            dones.append(done)
            obs = next_obs
            total_ret += reward
            steps += 1
        _, last_val = agent.model(torch.FloatTensor(obs).to(device))
        returns = agent.compute_gae(rewards, values, dones, last_val.item())
        advantages = [ret - v for ret,v in zip(returns, values)]
        threshold = np.quantile(all_returns[-100:] + [total_ret], agent.risk_alpha)
        shaped = [r + (agent.beta if total_ret< threshold else 0) for r in rewards]
        all_returns.append(total_ret)
        batch = {
            'obs': np.vstack(obs_buf),
            'actions': np.stack(actions),
            'logps': logps,
            'returns': returns,
            'advs': advantages
        }
        pl, vl, el = agent.update(batch)
        writer.add_scalar('Return', total_ret, ep)
        writer.add_scalar('PolicyLoss', pl, ep)
        writer.add_scalar('ValueLoss', vl, ep)
        writer.add_scalar('EntropyLoss', el, ep)
        writer.add_scalar('RiskThreshold', threshold, ep)
        with open(csv_path, 'a', newline='') as f:
            csv.writer(f).writerow([ep, total_ret, steps, threshold])
        if ep % 10 == 0:
            print(f"Episode {ep:4d} | Return {total_ret:.2f} | Length {steps} | Thr {threshold:.2f}")
    writer.close()
    torch.save(agent.model.state_dict(), os.path.join(out_dir,'rappo_model.pt'))

if __name__ == '__main__':
    train_rappo(num_episodes=50000)

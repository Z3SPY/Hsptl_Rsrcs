# 🚑 Corrected RA-PPO for HospitalSim (Shift-level batching)

import os
import time
import csv
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import matplotlib.pyplot as plt
import gym

from hospital_env import HospitalSimEnv

# ─── CONFIG ─────────────────────────────────────────────────────────────────
LOG_DIR = "logs/rappo_hospital_logs"
CSV_LOG = os.path.join(LOG_DIR, "rappo_log.csv")
MODEL_PATH = os.path.join(LOG_DIR, "rappo_trained_hospital.pth")

SHIFT_BATCH = 64             # How many shifts per update
CLIP_EPS = 0.2
LR = 3e-4
GAMMA = 0.99
GAE_LAMBDA = 0.95
ENT_COEF = 0.01
N_EPOCHS = 4
MINI_BATCH = 16
ITERS = 100
ALPHA = 0.25

STEP_SIZE = 15
RC_PERIOD = 20160

os.makedirs(LOG_DIR, exist_ok=True)

# ─── ENVIRONMENT ─────────────────────────────────────────────────────────────
def make_env():
    default_sim_config = {
        "n_triage": 2,
        "n_reg": 2,
        "n_exam": 3,
        "n_trauma": 2,
        "n_cubicles_1": 3,
        "n_cubicles_2": 2,
        "random_number_set": 1,
        "prob_trauma": 0.12,
    }
    return HospitalSimEnv(sim_config=default_sim_config, step_size=STEP_SIZE, rc_period=RC_PERIOD, cost_mode='diverse')

env = make_env()
obs_shape = env.observation_space.shape
action_space = env.action_space

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─── MODEL ─────────────────────────────────────────────────────────────
class RAPPOActorCritic(nn.Module):
    def __init__(self, obs_shape, action_space):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Flatten(),
            nn.Linear(np.prod(obs_shape), 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU()
        )

        if isinstance(action_space, gym.spaces.MultiDiscrete):
            self.policy_heads = nn.ModuleList([nn.Linear(256, n) for n in action_space.nvec])
            self.multidiscrete = True
        else:
            self.policy_head = nn.Linear(256, action_space.n)
            self.multidiscrete = False

        self.value_head = nn.Linear(256, 1)

    def forward(self, x):
        x = self.shared(x)

        if self.multidiscrete:
            policy_logits = [head(x) for head in self.policy_heads]
        else:
            policy_logits = self.policy_head(x)

        state_value = self.value_head(x)
        return policy_logits, state_value

model = RAPPOActorCritic(obs_shape, action_space).to(device)
optimizer = optim.Adam(model.parameters(), lr=LR)

# ─── HELPER FUNCTIONS ─────────────────────────────────────────────────────────
def select_action(obs):
    obs_t = torch.FloatTensor(obs).unsqueeze(0).to(device)
    logits, _ = model(obs_t)

    if isinstance(env.action_space, gym.spaces.MultiDiscrete):
        action = []
        logp_sum = 0
        for logits_per_action in logits:
            dist = Categorical(logits=logits_per_action)
            a = dist.sample()
            logp = dist.log_prob(a)
            action.append(a.item())
            logp_sum += logp
        return np.array(action), logp_sum
    else:
        logits = logits.squeeze(0)
        dist = Categorical(logits=logits)
        a = dist.sample()
        logp = dist.log_prob(a)
        return a.item(), logp

def run_single_shift():
    obs = env.reset()
    if isinstance(obs, tuple):
        obs = obs[0]

    action, logp = select_action(obs)

    # Apply shift action
    shift_obs, reward, terminated, _ = env.step(action)
    done = terminated

    # Full shift done
    obs_t = torch.FloatTensor(obs).unsqueeze(0).to(device)
    _, val = model(obs_t)

    return obs, action, logp.item(), val.item(), reward, done

def compute_gae(rews, vals, dones, last_v):
    T = len(rews)
    returns, advs = np.zeros(T), np.zeros(T)
    gae = 0
    vals = np.append(vals, last_v)

    for t in reversed(range(T)):
        mask = 1.0 - dones[t]
        delta = rews[t] + GAMMA * vals[t+1] * mask - vals[t]
        gae = delta + GAMMA * GAE_LAMBDA * mask * gae
        advs[t] = gae
        returns[t] = advs[t] + vals[t]

    return returns, advs

def compute_cvar(returns, alpha=ALPHA):
    sorted_returns = np.sort(returns)
    idx = int(alpha * len(sorted_returns))
    return np.mean(sorted_returns[:max(idx, 1)])

# ─── TRAINING ───────────────────────────────────────────────────────────────
csv_file = open(CSV_LOG, "w", newline="")
csv_writer = csv.writer(csv_file)
csv_writer.writerow(["iter", "timesteps", "avg_return", "cvar_return"])

step_logs, return_logs, cvar_logs = [], [], []
total_steps = 0
start_time = time.time()

for it in range(1, ITERS + 1):
    mb_obs, mb_acts, mb_logps, mb_vals, mb_rews, mb_dones = [], [], [], [], [], []

    for _ in range(SHIFT_BATCH):
        obs, act, logp, val, rew, done = run_single_shift()
        mb_obs.append(obs)
        mb_acts.append(act)
        mb_logps.append(logp)
        mb_vals.append(val)
        mb_rews.append(rew)
        mb_dones.append(done)

    total_steps += SHIFT_BATCH

    returns, advantages = compute_gae(mb_rews, mb_vals, mb_dones, last_v=0)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    obs_flat = torch.FloatTensor(mb_obs).to(device)
    acts_flat = torch.LongTensor(mb_acts).to(device)
    old_logps_flat = torch.FloatTensor(mb_logps).to(device)
    returns_flat = torch.FloatTensor(returns).to(device)
    advs_flat = torch.FloatTensor(advantages).to(device)

    idxs = np.arange(SHIFT_BATCH)
    np.random.shuffle(idxs)

    for _ in range(N_EPOCHS):
        for start in range(0, SHIFT_BATCH, MINI_BATCH):
            batch_idx = idxs[start:start+MINI_BATCH]

            batch_obs = obs_flat[batch_idx]
            batch_acts = acts_flat[batch_idx]
            batch_old_logps = old_logps_flat[batch_idx]
            batch_returns = returns_flat[batch_idx]
            batch_advs = advs_flat[batch_idx]

            logits, vals = model(batch_obs)

            if isinstance(env.action_space, gym.spaces.MultiDiscrete):
                logps = []
                split_batch_acts = torch.split(batch_acts, 1, dim=-1)
                for logit_per_head, act_per_head in zip(logits, split_batch_acts):
                    dist = Categorical(logits=logit_per_head)
                    logp = dist.log_prob(act_per_head.squeeze(-1))
                    logps.append(logp)
                logps = torch.stack(logps, dim=-1).sum(-1)
            else:
                dist = Categorical(logits=logits)
                logps = dist.log_prob(batch_acts)

            ratio = torch.exp(logps - batch_old_logps)
            surr1 = ratio * batch_advs
            surr2 = torch.clamp(ratio, 1-CLIP_EPS, 1+CLIP_EPS) * batch_advs
            policy_loss = -torch.min(surr1, surr2).mean()

            value_loss = 0.5 * (batch_returns - vals.squeeze()).pow(2).mean()
            entropy = dist.entropy().mean()

            weights = (batch_returns <= np.percentile(batch_returns.cpu().numpy(), ALPHA * 100)).float().to(device)
            policy_loss = (policy_loss * weights).mean()

            loss = policy_loss + 0.5 * value_loss - ENT_COEF * entropy

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    avg_ret = np.mean(mb_rews)
    cvar_ret = compute_cvar(mb_rews)

    step_logs.append(total_steps)
    return_logs.append(avg_ret)
    cvar_logs.append(cvar_ret)

    csv_writer.writerow([it, total_steps, f"{avg_ret:.2f}", f"{cvar_ret:.2f}"])

    print(f"Iter {it}/{ITERS} | Steps {total_steps} | AvgRet {avg_ret:.2f} | CVaR {cvar_ret:.2f}")

csv_file.close()
torch.save(model.state_dict(), MODEL_PATH)
print(f"\n✅ RA-PPO model saved to {MODEL_PATH}")

# ─── PLOT RESULTS ────────────────────────────────────────────────
plt.figure()
plt.plot(step_logs, return_logs, label="Average Return")
plt.plot(step_logs, cvar_logs, label="CVaR Return")
plt.legend()
plt.xlabel("Timesteps")
plt.ylabel("Returns")
plt.title("RA-PPO Hospital Training")
plt.savefig(os.path.join(LOG_DIR, "rappo_hospital_training.png"))

env.close()

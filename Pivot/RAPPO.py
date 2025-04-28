# RA_PPO_gridworld.py
import os
import time
import csv
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import matplotlib.pyplot as plt
import imageio

from stochastic_gridworld import StochasticGridWorld
from ppo_agent import PPOActorCritic

# ─── CONFIG ──────────────────────────────────────────────────────────────────
LOG_DIR = "logs/ra_ppo_gridworld_logs"
CSV_LOG = os.path.join(LOG_DIR, "rapppo_log.csv")
MODEL_PATH = os.path.join(LOG_DIR, "rapppo_trained_gridworld.pth")
GIF_PATH = os.path.join(LOG_DIR, "agent_play_rappo.gif")

BATCH_STEPS = 4096
CLIP_EPS = 0.2
LR = 1e-4
GAMMA = 0.99
GAE_LAMBDA = 0.95
ENT_COEF = 0.01
N_EPOCHS = 4
MINI_BATCH = 512
ITERS = 30
ALPHA = 0.25  # CVaR alpha
# ──────────────────────────────────────────────────────────────────────────────

os.makedirs(LOG_DIR, exist_ok=True)

# ─── ENVIRONMENT ─────────────────────────────────────────────────────────────
env = StochasticGridWorld(render_mode="human")
obs_shape = env.observation_space.shape
n_actions = env.action_space.n
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─── POLICY NETWORK ──────────────────────────────────────────────────────────
class RAPPOActorCritic(nn.Module):
    def __init__(self, obs_shape, n_actions):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Flatten(),
            nn.Linear(np.prod(obs_shape), 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )
        self.policy_head = nn.Linear(128, n_actions)
        self.value_head = nn.Linear(128, 1)

    def forward(self, x):
        x = self.shared(x)
        policy_logits = self.policy_head(x)
        state_value = self.value_head(x)
        return policy_logits, state_value

model = RAPPOActorCritic(obs_shape, n_actions).to(device)
optimizer = optim.Adam(model.parameters(), lr=LR)

# ─── LOGGING ──────────────────────────────────────────────────────────────────
csv_file = open(CSV_LOG, "w", newline="")
csv_writer = csv.writer(csv_file)
csv_writer.writerow(["iter", "timesteps", "episodes", "avg_return", "cvar_return", "policy_loss", "value_loss", "entropy", "elapsed_s", "steps_per_s"])

step_logs, return_logs, cvar_logs = [], [], []
loss_logs = {"policy": [], "value": [], "entropy": []}
start_time = time.time()

# ─── HELPER FUNCTIONS ─────────────────────────────────────────────────────────
def get_batch():
    obs, _ = env.reset()
    mb_obs, mb_acts, mb_logps, mb_vals, mb_rews, mb_dones = [], [], [], [], [], []
    ep_returns = []
    episode_rews = 0

    for _ in range(BATCH_STEPS):
        obs_t = torch.FloatTensor(obs).unsqueeze(0).to(device)
        logits, vals = model(obs_t)
        dist = Categorical(logits=logits)
        act = dist.sample()
        logp = dist.log_prob(act)

        next_obs, rew, done, truncated, _ = env.step(act.item())

        mb_obs.append(obs)
        mb_acts.append(act.item())
        mb_logps.append(logp.item())
        mb_vals.append(vals.item())
        mb_rews.append(rew)
        mb_dones.append(done)

        obs = next_obs
        episode_rews += rew

        if done:
            ep_returns.append(episode_rews)
            obs, _ = env.reset()
            episode_rews = 0

    # Bootstrap last value
    obs_t = torch.FloatTensor(obs).unsqueeze(0).to(device)
    _, last_v = model(obs_t)

    return (np.array(mb_obs), np.array(mb_acts), np.array(mb_logps),
            np.array(mb_vals), np.array(mb_rews), np.array(mb_dones),
            last_v.item(), ep_returns)

def compute_gae(rews, vals, dones, last_v):
    T = len(rews)
    returns = np.zeros(T, dtype=np.float32)
    advs = np.zeros(T, dtype=np.float32)
    gae = 0
    vals = np.append(vals, last_v)

    for t in reversed(range(T)):
        mask = 1.0 - dones[t]
        delta = rews[t] + GAMMA * vals[t+1] * mask - vals[t]
        gae = delta + GAMMA * GAE_LAMBDA * mask * gae
        advs[t] = gae
        returns[t] = advs[t] + vals[t]

    return returns, advs

def compute_cvar(returns, alpha=0.25):
    sorted_returns = np.sort(returns)
    idx = int(alpha * len(sorted_returns))
    if idx == 0:
        return np.mean(sorted_returns)
    return np.mean(sorted_returns[:idx])

# ─── TRAINING LOOP ────────────────────────────────────────────────────────────
total_steps = 0
episode_count = 0

for it in range(1, ITERS + 1):
    mb_obs, mb_acts, mb_logps, mb_vals, mb_rews, mb_dones, last_v, completed_returns = get_batch()
    total_steps += BATCH_STEPS
    episode_count += len(completed_returns)

    returns, advantages = compute_gae(mb_rews, mb_vals, mb_dones, last_v)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    obs_flat = torch.FloatTensor(mb_obs).to(device)
    acts_flat = torch.LongTensor(mb_acts).to(device)
    old_logps_flat = torch.FloatTensor(mb_logps).to(device)
    returns_flat = torch.FloatTensor(returns).to(device)
    advs_flat = torch.FloatTensor(advantages).to(device)

    idxs = np.arange(BATCH_STEPS)
    np.random.shuffle(idxs)

    policy_losses, value_losses, entropies = [], [], []

    for _ in range(N_EPOCHS):
        for start in range(0, BATCH_STEPS, MINI_BATCH):
            batch_idx = idxs[start:start+MINI_BATCH]
            batch_obs = obs_flat[batch_idx]
            batch_acts = acts_flat[batch_idx]
            batch_old_logps = old_logps_flat[batch_idx]
            batch_returns = returns_flat[batch_idx]
            batch_advs = advs_flat[batch_idx]

            logits, vals = model(batch_obs)
            dist = Categorical(logits=logits)
            logps = dist.log_prob(batch_acts)

            ratio = torch.exp(logps - batch_old_logps)
            surr1 = ratio * batch_advs
            surr2 = torch.clamp(ratio, 1-CLIP_EPS, 1+CLIP_EPS) * batch_advs
            policy_loss = -torch.min(surr1, surr2).mean()

            value_loss = 0.5 * (batch_returns - vals.squeeze()).pow(2).mean()
            entropy = dist.entropy().mean()

            # RA-PPO modifies the objective by applying CVaR weighting
            weights = (batch_returns <= np.percentile(batch_returns.cpu().numpy(), ALPHA * 100)).float()
            weights = weights.to(device)
            policy_loss = (policy_loss * weights).mean()

            loss = policy_loss + 0.5 * value_loss - ENT_COEF * entropy

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            policy_losses.append(policy_loss.item())
            value_losses.append(value_loss.item())
            entropies.append(entropy.item())

    avg_ret = np.mean(completed_returns) if completed_returns else float('nan')
    cvar_ret = compute_cvar(completed_returns, alpha=ALPHA)

    elapsed = time.time() - start_time
    speed = total_steps / elapsed

    # Logging
    csv_writer.writerow([it, total_steps, episode_count, f"{avg_ret:.2f}", f"{cvar_ret:.2f}",
                          f"{np.mean(policy_losses):.4f}", f"{np.mean(value_losses):.4f}", f"{np.mean(entropies):.4f}",
                          f"{elapsed:.2f}", f"{speed:.2f}"])
    step_logs.append(total_steps)
    return_logs.append(avg_ret)
    cvar_logs.append(cvar_ret)

    print(f"Iter {it}/{ITERS} | Steps {total_steps} | Eps {episode_count} "
          f"| AvgRet {avg_ret:.2f} | CVaR {cvar_ret:.2f} "
          f"| pol {np.mean(policy_losses):.4f} | val {np.mean(value_losses):.4f} | ent {np.mean(entropies):.4f} "
          f"| Time {elapsed:.2f} sec | Speed {speed:.2f} steps/sec")

csv_file.close()

# Save model
torch.save(model.state_dict(), MODEL_PATH)
print(f"✅ RA-PPO model saved to {MODEL_PATH}")

# ─── PLOTTING ────────────────────────────────────────────────────────────────
plt.figure()
plt.plot(step_logs, return_logs, label="Average Return")
plt.plot(step_logs, cvar_logs, label="CVaR Return")
plt.legend()
plt.xlabel("Env Steps"); plt.ylabel("Returns")
plt.title("RA-PPO GridWorld Training Performance")
plt.savefig(os.path.join(LOG_DIR, "rappo_training_curves.png"))

# ─── EVALUATION AND GIF RECORDING ────────────────────────────────────────────
print("\n▶ Recording RA-PPO agent behavior...")

obs, _ = env.reset()
frames = []
done = False

while not done:
    obs_t = torch.FloatTensor(obs).unsqueeze(0).to(device)
    with torch.no_grad():
        logits, _ = model(obs_t)
        action = torch.argmax(logits, dim=1).item()

    obs, rew, done, truncated, _ = env.step(action)

    frame = env.render()  # <-- get the frame returned from render()
    frames.append(frame)

    time.sleep(0.1)

# Save frames as gif
imageio.mimsave(GIF_PATH, frames, fps=5)
print(f"🎬 RA-PPO Agent GIF saved to {GIF_PATH}")

env.close()


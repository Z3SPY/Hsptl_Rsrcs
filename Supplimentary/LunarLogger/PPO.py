# ppo_lunarlander_custom.py
import os
import time
import csv
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from gym.vector import SyncVectorEnv
from gym.wrappers import RecordVideo

if not hasattr(np, "bool8"):
    np.bool8 = np.bool

# ─── CONFIG ────────────────────────────────────────────────────────────────
ENV_ID        = "LunarLander-v2"
LOG_DIR       = "logs/ppo_logged"
CSV_LOG       = os.path.join(LOG_DIR, "ppo_log.csv")
PLOTS_DIR     = LOG_DIR
VIDEO_DIR     = os.path.join(LOG_DIR, "videos")
N_ENVS        = 4
BATCH_STEPS   = 2048      # timesteps per env per update
CLIP_EPS      = 0.2
LR            = 3e-4
GAMMA         = 0.99
GAE_LAMBDA    = 0.95
ENT_COEF      = 0.01
N_EPOCHS      = 4
MINI_BATCH    = 256
ITERS         = 125       # number of updates
# ─────────────────────────────────────────────────────────────────────────────

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(VIDEO_DIR, exist_ok=True)


class PolicyNet(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
        )
        self.policy_head = nn.Linear(128, act_dim)
        self.value_head  = nn.Linear(128, 1)

    def forward(self, x):
        x = self.shared(x)
        return self.policy_head(x), self.value_head(x).squeeze(-1)


class PPO_Agent:
    def __init__(self, train_env, record_env):
        self.env       = train_env
        self.rec_env   = record_env
        self.device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        obs_dim = train_env.single_observation_space.shape[0]
        act_dim = train_env.single_action_space.n
        self.model    = PolicyNet(obs_dim, act_dim).to(self.device)
        self.optimizer= optim.Adam(self.model.parameters(), lr=LR)

        # Logging setup
        self.csv_file = open(CSV_LOG, "w", newline="")
        self.csv     = csv.writer(self.csv_file)
        self.csv.writerow([
            "iter","timesteps","episodes",
            "elapsed_s","steps_per_s","avg_return",
            "policy_loss","value_loss","entropy"
        ])
        self.start_time = time.time()

        # Metrics buffers
        self.step_logs    = []
        self.ep_logs      = []
        self.speed_logs   = []
        self.return_logs  = []
        self.loss_logs    = {"policy":[], "value":[], "entropy":[]}
        self.episode_count = 0

    def get_batch(self):
        obs, _ = self.env.reset()
        mb_obs, mb_acts, mb_logps, mb_vals, mb_rews, mb_dones = [],[],[],[],[],[]
        ep_returns = [0.0]*N_ENVS
        completed_returns = []

        for _ in range(BATCH_STEPS):
            obs_t    = torch.FloatTensor(obs).to(self.device)
            logits, vals = self.model(obs_t)
            dist     = Categorical(logits=logits)
            acts     = dist.sample()
            logps    = dist.log_prob(acts)

            next_obs, rews, dones, _, _ = self.env.step(acts.cpu().numpy())

            mb_obs.append(obs)
            mb_acts.append(acts.cpu().numpy())
            mb_logps.append(logps.detach().cpu().numpy())
            mb_vals.append(vals.detach().cpu().numpy())
            mb_rews.append(rews)
            mb_dones.append(dones)

            obs = next_obs
            for i in range(N_ENVS):
                ep_returns[i] += rews[i]
                if dones[i]:
                    completed_returns.append(ep_returns[i])
                    ep_returns[i] = 0.0

        # bootstrap values for GAE
        obs_t      = torch.FloatTensor(obs).to(self.device)
        _, last_v  = self.model(obs_t)
        return (
            np.array(mb_obs), np.array(mb_acts), np.array(mb_logps),
            np.array(mb_vals), np.array(mb_rews), np.array(mb_dones),
            last_v.detach().cpu().numpy(), completed_returns
        )

    def compute_gae(self, rews, vals, dones, last_v):
        T, N = rews.shape
        returns = np.zeros((T, N), dtype=np.float32)
        advs    = np.zeros((T, N), dtype=np.float32)
        for n in range(N):
            gae = 0.0
            for t in reversed(range(T)):
                mask  = 1.0 - dones[t, n]
                delta = rews[t,n] + GAMMA * last_v[n] * mask - vals[t,n]
                gae   = delta + GAMMA * GAE_LAMBDA * mask * gae
                advs[t,n]    = gae
                returns[t,n] = gae + vals[t,n]
                last_v[n]    = vals[t,n]
        return returns.flatten(), advs.flatten()

    def update(self, mb_obs, mb_acts, mb_logps, mb_vals, mb_rews, mb_dones, last_v):
        rets, advs = self.compute_gae(mb_rews, mb_vals, mb_dones, last_v)
        obs = mb_obs.reshape(-1, mb_obs.shape[-1])
        acts, old_lp = mb_acts.flatten(), mb_logps.flatten()
        dataset = list(zip(obs, acts, old_lp, rets, advs))
        np.random.shuffle(dataset)

        policy_losses, value_losses, entropies = [], [], []
        for _ in range(N_EPOCHS):
            for start in range(0, len(dataset), MINI_BATCH):
                batch = dataset[start:start+MINI_BATCH]
                b_obs, b_acts, b_lp, b_rets, b_advs = zip(*batch)
                b_obs = torch.FloatTensor(b_obs).to(self.device)
                b_acts= torch.LongTensor(b_acts).to(self.device)
                b_lp  = torch.FloatTensor(b_lp).to(self.device)
                b_rets= torch.FloatTensor(b_rets).to(self.device)
                b_advs= torch.FloatTensor(b_advs).to(self.device)
                b_advs= (b_advs - b_advs.mean())/(b_advs.std()+1e-8)

                logits, vals = self.model(b_obs)
                dist   = Categorical(logits=logits)
                lp     = dist.log_prob(b_acts)
                ratio  = torch.exp(lp - b_lp)

                # clipped surrogate objective
                p1 = ratio * b_advs
                p2 = torch.clamp(ratio, 1-CLIP_EPS, 1+CLIP_EPS) * b_advs
                pol_loss = -torch.min(p1, p2).mean()

                val_loss = (b_rets - vals).pow(2).mean()
                ent      = dist.entropy().mean()

                loss = pol_loss + 0.5*val_loss - ENT_COEF*ent

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                policy_losses.append(pol_loss.item())
                value_losses.append(val_loss.item())
                entropies.append(ent.item())

        return (
            np.mean(policy_losses),
            np.mean(value_losses),
            np.mean(entropies)
        )

    def train(self):
        total_steps = 0
        for it in range(1, ITERS+1):
            mb_obs, mb_acts, mb_logps, mb_vals, mb_rews, mb_dones, last_v, comp_rets = self.get_batch()
            total_steps += N_ENVS * BATCH_STEPS
            self.episode_count += len(comp_rets)

            # do PPO update
            pl, vl, ent = self.update(mb_obs, mb_acts, mb_logps, mb_vals, mb_rews, mb_dones, last_v)
            avg_ret = np.mean(comp_rets) if comp_rets else float('nan')

            # timing & speed
            elapsed = time.time() - self.start_time
            speed   = total_steps / elapsed

            # log CSV & buffers
            self.csv.writerow([it, total_steps, self.episode_count,
                               f"{elapsed:.2f}", f"{speed:.2f}", f"{avg_ret:.2f}",
                               f"{pl:.4f}", f"{vl:.4f}", f"{ent:.4f}"])
            self.step_logs.append(total_steps)
            self.ep_logs.append(self.episode_count)
            self.speed_logs.append(speed)
            self.return_logs.append(avg_ret)
            self.loss_logs["policy"].append(pl)
            self.loss_logs["value"].append(vl)
            self.loss_logs["entropy"].append(ent)

            print(f"Iter {it}/{ITERS} | Steps {total_steps} | Eps {self.episode_count}"
                  f" | AvgRet {avg_ret:.2f} | pol {pl:.4f} | val {vl:.4f} | ent {ent:.4f}")

        # close CSV
        self.csv_file.close()

        # === Plotting ===
        import matplotlib.pyplot as plt

        # 1) Return vs Steps
        plt.figure()
        plt.plot(self.step_logs, self.return_logs, label="AvgReturn")
        plt.xlabel("Env Steps"); plt.ylabel("Return"); plt.legend()
        plt.savefig(os.path.join(PLOTS_DIR, "ppo_return_vs_steps.png"))

        # 2) Speed vs Steps
        plt.figure()
        plt.plot(self.step_logs, self.speed_logs, label="Steps/sec")
        plt.xlabel("Env Steps"); plt.ylabel("Speed"); plt.savefig(os.path.join(PLOTS_DIR, "ppo_speed_vs_steps.png"))

        # 3) Policy Loss
        plt.figure()
        plt.plot(self.step_logs, self.loss_logs["policy"])
        plt.xlabel("Env Steps"); plt.ylabel("Policy Loss")
        plt.savefig(os.path.join(PLOTS_DIR, "ppo_policy_loss.png"))

        # 4) Value Loss
        plt.figure()
        plt.plot(self.step_logs, self.loss_logs["value"])
        plt.xlabel("Env Steps"); plt.ylabel("Value Loss")
        plt.savefig(os.path.join(PLOTS_DIR, "ppo_value_loss.png"))

        # 5) Entropy
        plt.figure()
        plt.plot(self.step_logs, self.loss_logs["entropy"])
        plt.xlabel("Env Steps"); plt.ylabel("Entropy")
        plt.savefig(os.path.join(PLOTS_DIR, "ppo_entropy.png"))

        # === Deterministic Eval & Video ===
        print("▶ Deterministic evaluation over 10 episodes...")
        rets = []
        for ep in range(10):
            obs, _ = self.rec_env.reset()
            done = False; total_r = 0.0
            while not done:
                with torch.no_grad():
                    logits, _ = self.model(torch.FloatTensor(obs).unsqueeze(0).to(self.device))
                    action = torch.argmax(logits, dim=1).item()
                obs, r, done, _, _ = self.rec_env.step(action)
                total_r += r
            rets.append(total_r)
            print(f"  Ep{ep+1}: {total_r:.2f}")
        mean_r, std_r = np.mean(rets), np.std(rets)
        print(f"✅ Eval: {mean_r:.2f} ± {std_r:.2f} over 10 eps")

        # record one rollout video
        self.rec_env.close()

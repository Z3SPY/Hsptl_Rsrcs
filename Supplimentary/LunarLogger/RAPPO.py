import os
import time
import csv

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from gym.wrappers import RecordVideo
import matplotlib.pyplot as plt

# Ensure compatibility for older numpy versions
if not hasattr(np, "bool8"):
    np.bool8 = bool

class PolicyNet(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU()
        )
        self.policy_head = nn.Linear(128, act_dim)
        self.value_head  = nn.Linear(128, 1)

    def forward(self, x):
        x = self.shared(x)
        return self.policy_head(x), self.value_head(x).squeeze(-1)

class RA_PPO_Agent:
    def __init__(self, train_env, record_env, n_envs=4, batch_steps=2048,
                 clip_eps=0.2, lr=3e-4, gamma=0.99, lam=0.95,
                 risk_alpha=0.05, beta=-5.0):
        # Envs and hyperparams
        self.train_env = train_env
        self.record_env = record_env
        self.n_envs = n_envs
        self.batch_steps = batch_steps
        self.gamma, self.lam = gamma, lam
        self.clip_eps, self.risk_alpha, self.beta = clip_eps, risk_alpha, beta

        # Model & optimizer
        obs_dim = train_env.single_observation_space.shape[0]
        act_dim = train_env.single_action_space.n
        self.model = PolicyNet(obs_dim, act_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # Logging and KPI storage
        self.step_count = 0
        self.all_returns = []  # track every episode return for CVaR
        os.makedirs('logs/ra_ppo', exist_ok=True)
        self.log_path = os.path.join('logs/ra_ppo', 'rapo_log.csv')
        self.plot_dir = os.path.join('logs/ra_ppo')
        self.log_file = open(self.log_path, 'w', newline='')
        self.logger = csv.writer(self.log_file)
        self.logger.writerow([
            'iteration', 'env_steps', 'elapsed_time_s', 'speed_steps_per_s',
            'avg_return', 'cvar', 'policy_loss', 'value_loss', 'entropy'
        ])
        self.return_logs, self.cvar_logs = [], []
        self.step_logs, self.speed_logs = [], []
        self.loss_logs = {'policy': [], 'value': [], 'entropy': []}

    def get_batch(self):
        obs, _ = self.train_env.reset()
        mb_obs, mb_acts, mb_logps, mb_vals, mb_rews, mb_dones = [], [], [], [], [], []
        ep_returns = [0.0] * self.n_envs
        completed_returns = []

        for _ in range(self.batch_steps):
            obs_tensor = torch.FloatTensor(obs).to(self.device)
            logits, vals = self.model(obs_tensor)
            dist = Categorical(logits=logits)
            acts = dist.sample()
            logps = dist.log_prob(acts)

            next_obs, rews, dones, _, _ = self.train_env.step(acts.cpu().numpy())
            mb_obs.append(obs)
            mb_acts.append(acts.cpu().numpy())
            mb_logps.append(logps.detach().cpu().numpy())
            mb_vals.append(vals.detach().cpu().numpy())
            mb_rews.append(rews)
            mb_dones.append(dones)
            obs = next_obs

            for i in range(self.n_envs):
                ep_returns[i] += rews[i]
                if dones[i]:
                    completed_returns.append(ep_returns[i])
                    ep_returns[i] = 0.0

        obs_tensor = torch.FloatTensor(obs).to(self.device)
        _, last_vals = self.model(obs_tensor)
        return (
            np.array(mb_obs), np.array(mb_acts), np.array(mb_logps),
            np.array(mb_vals), np.array(mb_rews), np.array(mb_dones),
            last_vals.detach().cpu().numpy(), completed_returns
        )

    def compute_gae(self, rews, vals, dones, last_vals):
        T, N = rews.shape
        returns = np.zeros((T, N), dtype=np.float32)
        advs    = np.zeros((T, N), dtype=np.float32)
        for n in range(N):
            gae = 0
            for t in reversed(range(T)):
                mask  = 1.0 - dones[t, n]
                delta = rews[t, n] + self.gamma * last_vals[n] * mask - vals[t, n]
                gae   = delta + self.gamma * self.lam * mask * gae
                advs[t, n]    = gae
                returns[t, n] = gae + vals[t, n]
                last_vals[n]  = vals[t, n]
        return returns.flatten(), advs.flatten()

    def update(self, batch):
        mb_obs, mb_acts, mb_logps, mb_vals, mb_rews, mb_dones, last_vals = batch
        rets, advs = self.compute_gae(mb_rews, mb_vals, mb_dones, last_vals)
        obs  = mb_obs.reshape(-1, mb_obs.shape[-1])
        acts = mb_acts.flatten()
        old_lp = mb_logps.flatten()

        dataset = list(zip(obs, acts, old_lp, rets, advs))
        np.random.shuffle(dataset)

        p_sum, v_sum, e_sum = 0.0, 0.0, 0.0
        count = 0
        for _ in range(4):
            for start in range(0, len(dataset), 256):
                batch_slice = dataset[start:start+256]
                b_obs, b_acts, b_lp, b_rets, b_advs = zip(*batch_slice)
                b_obs = torch.FloatTensor(b_obs).to(self.device)
                b_acts = torch.LongTensor(b_acts).to(self.device)
                b_lp = torch.FloatTensor(b_lp).to(self.device)
                b_rets = torch.FloatTensor(b_rets).to(self.device)
                b_advs = torch.FloatTensor(b_advs).to(self.device)
                b_advs = (b_advs - b_advs.mean()) / (b_advs.std() + 1e-8)

                logits, vals = self.model(b_obs)
                dist = Categorical(logits=logits)
                lp = dist.log_prob(b_acts)
                ratio = torch.exp(lp - b_lp)

                s1 = ratio * b_advs
                s2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * b_advs
                p_loss = -torch.min(s1, s2).mean()
                v_loss = (b_rets - vals).pow(2).mean()
                ent    = dist.entropy().mean()
                loss   = p_loss + 0.5 * v_loss - 0.01 * ent

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                p_sum += p_loss.item()
                v_sum += v_loss.item()
                e_sum += ent.item()
                count += 1

        return {
            'policy_loss': p_sum / count,
            'value_loss':  v_sum / count,
            'entropy':     e_sum / count
        }

    def train(self, iters=125):
        start_time = time.time()
        for i in range(iters):
            batch = self.get_batch()
            *ppo_batch, completed_returns = batch

            # track each episode return for CVaR
            self.all_returns.extend(completed_returns)

            ep_returns = batch[4].sum(axis=0)
            self.step_count += self.batch_steps * self.n_envs
            cvar = (
                np.quantile(self.all_returns[-100:], self.risk_alpha)
                if len(self.all_returns) >= 100 else np.nan
            )
            losses = self.update(ppo_batch)

            elapsed = time.time() - start_time
            speed   = self.step_count / elapsed if elapsed > 0 else 0.0

            avg_ret = ep_returns.mean()
            self.return_logs.append(avg_ret)
            self.cvar_logs.append(cvar)
            self.step_logs.append(self.step_count)
            self.speed_logs.append(speed)
            self.loss_logs['policy'].append(losses['policy_loss'])
            self.loss_logs['value'].append(losses['value_loss'])
            self.loss_logs['entropy'].append(losses['entropy'])

            self.logger.writerow([
                i+1, self.step_count, elapsed, speed,
                avg_ret, cvar,
                losses['policy_loss'], losses['value_loss'], losses['entropy']
            ])

            print(f"Iter {i+1}/{iters} | Steps {self.step_count} | Time {elapsed:.1f}s | "
                  f"AvgRet {avg_ret:.2f} | CVaR {cvar:.2f} | Speed {speed:.1f} st/s")

        self.log_file.close()

        # 1) AvgReturn & CVaR vs steps (unchanged)
        plt.figure()
        plt.plot(self.step_logs, self.return_logs, label='AvgReturn')
        plt.plot(self.step_logs, self.cvar_logs,   label=f'CVaR_{self.risk_alpha}')
        plt.xlabel('Env Steps'); plt.ylabel('Return'); plt.legend()
        plt.savefig(os.path.join(self.plot_dir, 'return_cvar_vs_steps.png'))

        # 2) Training Speed vs steps
        plt.figure()
        plt.plot(self.step_logs, self.speed_logs, label='Steps/sec')
        plt.xlabel('Env Steps'); plt.ylabel('Speed (steps/sec)'); plt.legend()
        plt.savefig(os.path.join(self.plot_dir, 'speed_vs_steps.png'))

        # 3) Policy Loss alone
        plt.figure()
        plt.plot(self.step_logs, self.loss_logs['policy'])
        plt.xlabel('Env Steps'); plt.ylabel('Policy Loss')
        plt.title('Policy Loss vs Env Steps')
        plt.savefig(os.path.join(self.plot_dir, 'policy_loss_vs_steps.png'))

        # 4) Value Loss alone
        plt.figure()
        plt.plot(self.step_logs, self.loss_logs['value'])
        plt.xlabel('Env Steps'); plt.ylabel('Value Loss')
        plt.title('Value Loss vs Env Steps')
        plt.savefig(os.path.join(self.plot_dir, 'value_loss_vs_steps.png'))

        # 5) Entropy alone
        plt.figure()
        plt.plot(self.step_logs, self.loss_logs['entropy'])
        plt.xlabel('Env Steps'); plt.ylabel('Entropy')
        plt.title('Entropy vs Env Steps')
        plt.savefig(os.path.join(self.plot_dir, 'entropy_vs_steps.png'))


        # Final rollout video
        print("🎥 Recording final rollout...")
        obs, _ = self.record_env.reset()
        done = False; total_reward = 0; steps = 0
        while not done:
            with torch.no_grad():
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
                logits, _ = self.model(obs_tensor)
                dist  = Categorical(logits=logits)
                action = torch.argmax(logits, dim=1).item()
            obs, reward, done, _, _ = self.record_env.step(action)
            total_reward += reward; steps += 1

        self.record_env.close()
        print(f"✅ Video recorded: {steps} steps, total reward: {total_reward:.2f}")

                # 🎥 Deterministic evaluation over multiple episodes
        num_eval = 10
        eval_returns = []
        print(f"▶ Starting deterministic evaluation over {num_eval} episodes...")
        for ep in range(num_eval):
            obs, _ = self.record_env.reset()
            done = False
            total_r = 0.0
            while not done:
                # pick the highest‐probability action
                with torch.no_grad():
                    logits, _ = self.model(torch.FloatTensor(obs).unsqueeze(0).to(self.device))
                    action = torch.argmax(logits, dim=1).item()
                obs, reward, done, _, _ = self.record_env.step(action)
                total_r += reward
            eval_returns.append(total_r)
            print(f"  Ep{ep+1}: return={total_r:.2f}")
        avg_final = np.mean(eval_returns)
        std_final = np.std(eval_returns)
        print(f"✅ Deterministic eval: {avg_final:.2f} ± {std_final:.2f} over {num_eval} episodes")

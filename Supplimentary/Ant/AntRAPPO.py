import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import csv
import time
import os
from torch.distributions import Normal
from gymnasium.wrappers import RecordVideo

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class RiskAdaptiveActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes=(64,64)):
        super().__init__()
        layers = []
        last = obs_dim
        for h in hidden_sizes:
            layers += [nn.Linear(last, h), nn.Tanh()]
            last = h
        self.shared = nn.Sequential(*layers)
        self.actor_mean = nn.Linear(last, act_dim)
        self.actor_log_std = nn.Parameter(torch.zeros(act_dim))
        self.critic = nn.Linear(last, 1)

    def forward(self, obs):
        x = self.shared(obs)
        mean = self.actor_mean(x)
        std = torch.exp(self.actor_log_std)
        value = self.critic(x).squeeze(-1)
        return mean, std, value

    def act(self, obs):
        mean, std, value = self.forward(obs)
        dist = Normal(mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        return action, log_prob, value

    def evaluate(self, obs, actions):
        mean, std, value = self.forward(obs)
        dist = Normal(mean, std)
        log_probs = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return log_probs, entropy, value


def compute_gae(rewards, values, dones, gamma=0.99, lam=0.95):
    advantages = []
    gae = 0
    values_ext = values + [0.0]
    for i in reversed(range(len(rewards))):
        delta = rewards[i] + gamma * values_ext[i+1] * (1 - dones[i]) - values_ext[i]
        gae = delta + gamma * lam * (1 - dones[i]) * gae
        advantages.insert(0, gae)
    returns = [adv + val for adv, val in zip(advantages, values)]
    return advantages, returns


def rappo_update(model, optimizer, batch, clip_eps=0.2, epochs=10, beta=1.0, shaped_returns=None):
    obs, actions, old_log_probs, returns, advantages = batch
    total_policy_loss = 0.0
    total_value_loss = 0.0
    total_entropy = 0.0

    for _ in range(epochs):
        log_probs, entropy, values = model.evaluate(obs, actions)
        ratio = torch.exp(log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        value_loss = (returns - values).pow(2).mean()
        entropy_bonus = entropy.mean()

        loss = policy_loss + 0.5 * value_loss - 0.01 * entropy_bonus

        if shaped_returns is not None:
            cvar_threshold = np.quantile(shaped_returns, 0.1)
            tail_returns = [r for r in shaped_returns if r <= cvar_threshold]
            if len(tail_returns) > 0:
                cvar_penalty = -beta * torch.tensor(tail_returns, dtype=torch.float32).mean()
                loss += cvar_penalty

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_policy_loss += policy_loss.item()
        total_value_loss += value_loss.item()
        total_entropy += entropy_bonus.item()

    n = float(epochs)
    return total_policy_loss / n, total_value_loss / n, total_entropy / n, loss.item()


class RAPPOAntTrainer:
    def __init__(self,
                 env_id="Ant-v4",
                 total_timesteps=5_000_000,
                 rollout_steps=2048,
                 beta=1.0,
                 video_folder="./videos",
                 csv_path="rappo_ant_benchmark.csv",
                 record_points=(0.0, 0.25, 0.5, 0.75, 1.0)):
        self.env_id = env_id
        self.total_timesteps = total_timesteps
        self.rollout_steps = rollout_steps
        self.beta = beta
        self.csv_path = csv_path
        self.record_points = record_points
        self.video_folder = video_folder
        os.makedirs(self.video_folder, exist_ok=True)
        self.episode_counter = 0

        rec_eps = [int(frac * (self.total_timesteps / 1000)) for frac in self.record_points]
        self.rec_eps_set = set(rec_eps)

        temp_env = gym.make(env_id, render_mode="rgb_array")
        temp_env = RecordVideo(temp_env, video_folder=self.video_folder,
                               episode_trigger=lambda ep: ep in self.rec_eps_set)
        obs_dim = temp_env.observation_space.shape[0]
        act_dim = temp_env.action_space.shape[0]
        temp_env.close()
        self.model = RiskAdaptiveActorCritic(obs_dim, act_dim).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=3e-4)

    def make_env(self):
        env = gym.make(self.env_id, render_mode="rgb_array")
        env = RecordVideo(env, video_folder=self.video_folder,
                          episode_trigger=lambda ep: self.episode_counter in self.rec_eps_set)
        return env

    def init_csv(self):
        with open(self.csv_path, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["iter", "env_steps", "elapsed_time_s", "steps_per_s",
                             "avg_return", "cvar", "policy_loss", "value_loss",
                             "entropy", "avg_loss"])

    def train(self):
        self.init_csv()
        env = self.make_env()
        obs, _ = env.reset()

        total_steps = 0
        iteration = 0
        start_time = time.time()
        returns_history = []

        while total_steps < self.total_timesteps:
            obs_buf, act_buf, logp_buf, rew_buf, val_buf, done_buf = [], [], [], [], [], []
            episode_returns = []
            ep_ret = 0
            for _ in range(self.rollout_steps):
                obs_t = torch.tensor(obs, dtype=torch.float32).to(device)
                action, logp, value = self.model.act(obs_t)
                next_obs, reward, terminated, truncated, _ = env.step(action.cpu().numpy())
                done = terminated or truncated

                obs_buf.append(obs_t)
                act_buf.append(action)
                logp_buf.append(logp)
                rew_buf.append(reward)
                val_buf.append(value.item())
                done_buf.append(done)

                ep_ret += reward
                if done:
                    if not np.isnan(ep_ret):
                        episode_returns.append(ep_ret)
                        returns_history.append(ep_ret)
                    ep_ret = 0
                    obs, _ = env.reset()
                    self.episode_counter += 1
                else:
                    obs = next_obs
                total_steps += 1

            advs, returns = compute_gae(rew_buf, val_buf, done_buf)
            advs = torch.tensor(advs, dtype=torch.float32).to(device)
            returns_t = torch.tensor(returns, dtype=torch.float32).to(device)
            advs = (advs - advs.mean()) / (advs.std() + 1e-8)

            obs_batch = torch.stack(obs_buf)
            act_batch = torch.stack(act_buf)
            logp_batch = torch.stack(logp_buf).detach()

            pl, vl, ent, al = rappo_update(
                self.model, self.optimizer,
                (obs_batch, act_batch, logp_batch, returns_t, advs),
                beta=self.beta,
                shaped_returns=episode_returns
            )

            iteration += 1
            elapsed = time.time() - start_time
            sps = total_steps / elapsed if elapsed > 0 else 0.0
            avg_ret = np.mean(episode_returns) if episode_returns else 0.0

            if returns_history:
                cvar_threshold = np.quantile(returns_history, 0.1)
                tail = [r for r in episode_returns if r <= cvar_threshold]
                cvar = np.mean(tail) if tail else 0.0
            else:
                cvar = 0.0

            with open(self.csv_path, mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([iteration, total_steps, f"{elapsed:.2f}", f"{sps:.2f}",
                                  f"{avg_ret:.2f}", f"{cvar:.2f}", f"{pl:.4f}",
                                  f"{vl:.4f}", f"{ent:.4f}", f"{al:.4f}"])

            print(f"[RAPPO] Iter {iteration} | Steps {total_steps} | Ret {avg_ret:.2f} | CVar {cvar:.2f} | SPS {sps:.1f}")

        env.close()
        torch.save(self.model.state_dict(), "rappo_ant_model.pt")

if __name__ == "__main__":
    trainer = RAPPOAntTrainer(
        env_id="Ant-v4",
        total_timesteps=5_000_000,
        rollout_steps=2048,
        beta=1.0,
        video_folder="./videos",
        csv_path="rappo_ant_benchmark.csv",
        record_points=(0.0, 0.25, 0.5, 0.75, 1.0)
    )
    trainer.train()

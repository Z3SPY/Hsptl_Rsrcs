import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import csv
import time
from torch.distributions import Normal
from gymnasium.wrappers import RecordVideo

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ActorCritic(nn.Module):
    """
    Shared actor-critic network for continuous actions.
    """
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
        """Sample action and return log-prob and value."""
        mean, std, value = self.forward(obs)
        dist = Normal(mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        return action, log_prob, value

    def evaluate(self, obs, actions):
        """Compute log-prob, entropy, and value for given states and actions."""
        mean, std, value = self.forward(obs)
        dist = Normal(mean, std)
        log_probs = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return log_probs, entropy, value


def compute_gae(rewards, values, dones, gamma=0.99, lam=0.95):
    """
    Compute GAE advantages and returns.
    """
    advantages = []
    gae = 0
    values_ext = values + [0.0]
    for i in reversed(range(len(rewards))):
        delta = rewards[i] + gamma * values_ext[i+1] * (1 - dones[i]) - values_ext[i]
        gae = delta + gamma * lam * (1 - dones[i]) * gae
        advantages.insert(0, gae)
    returns = [adv + val for adv, val in zip(advantages, values)]
    return advantages, returns


def ppo_update(model, optimizer, batch, clip_eps=0.2, epochs=10):
    """
    Perform PPO update and return metrics.
    """
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
        entropy_mean = entropy.mean()
        loss = policy_loss + 0.5 * value_loss - 0.01 * entropy_mean

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_policy_loss += policy_loss.item()
        total_value_loss += value_loss.item()
        total_entropy += entropy_mean.item()

    n = float(epochs)
    avg_policy_loss = total_policy_loss / n
    avg_value_loss = total_value_loss / n
    avg_entropy = total_entropy / n
    avg_loss = avg_policy_loss + 0.5 * avg_value_loss - 0.01 * avg_entropy
    return avg_policy_loss, avg_value_loss, avg_entropy, avg_loss


class PPOAntTrainer:
    def __init__(self,
                 env_id="Ant-v4",
                 total_timesteps=5_000_000,
                 rollout_steps=2048,
                 video_folder="./videos",
                 csv_path="ppo_ant_benchmark.csv",
                 record_points=(0.0, 0.25, 0.5, 0.75, 1.0)):
        self.env_id = env_id
        self.total_timesteps = total_timesteps
        self.rollout_steps = rollout_steps
        self.csv_path = csv_path
        self.record_points = record_points
        self.video_folder = video_folder
        temp_env = gym.make(env_id)
        obs_dim = temp_env.observation_space.shape[0]
        act_dim = temp_env.action_space.shape[0]
        temp_env.close()
        self.model = ActorCritic(obs_dim, act_dim).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=3e-4)

    def make_env(self):
        rec_iters = [int(frac * (self.total_timesteps / self.rollout_steps)) for frac in self.record_points]
        env = gym.make(self.env_id, render_mode="rgb_array")
        env = RecordVideo(env, video_folder=self.video_folder,
                          episode_trigger=lambda epi: epi in rec_iters)
        return env

    def init_csv(self):
        with open(self.csv_path, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["iter", "env_steps", "elapsed_time_s", "steps_per_s", \
                              "avg_return", "cvar", "policy_loss", "value_loss", \
                              "entropy", "avg_loss"]);

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
                    episode_returns.append(ep_ret)
                    returns_history.append(ep_ret)
                    ep_ret = 0
                    obs, _ = env.reset()
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

            pl, vl, ent, al = ppo_update(
                self.model, self.optimizer,
                (obs_batch, act_batch, logp_batch, returns_t, advs)
            )

            iteration += 1
            elapsed = time.time() - start_time
            sps = total_steps / elapsed if elapsed > 0 else 0.0
            avg_ret = np.mean(episode_returns) if episode_returns else 0.0
            cvar_thr = np.quantile(returns_history, 0.1) if returns_history else 0.0
            cvar = np.mean([r for r in episode_returns if r <= cvar_thr]) if episode_returns else 0.0

            with open(self.csv_path, mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([iteration, total_steps, f"{elapsed:.2f}", f"{sps:.2f}", \
                                  f"{avg_ret:.2f}", f"{cvar:.2f}", f"{pl:.4f}", \
                                  f"{vl:.4f}", f"{ent:.4f}", f"{al:.4f}"])

            print(f"Iter {iteration} | Steps {total_steps} | Ret {avg_ret:.2f} | CVar {cvar:.2f} | SPS {sps:.1f}")

        env.close()
        torch.save(self.model.state_dict(), "ppo_ant_video_model.pt")

if __name__ == "__main__":
    trainer = PPOAntTrainer(
        env_id="Ant-v4",
        total_timesteps=5_000_000,
        rollout_steps=2048,
        video_folder="./videos",
        csv_path="ppo_ant_benchmark.csv",
        record_points=(0.0, 0.25, 0.5, 0.75, 1.0)
    )
    trainer.train()

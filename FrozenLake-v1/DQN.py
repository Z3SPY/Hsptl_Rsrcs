# DQN.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gym
from collections import deque

# CONFIGURATION
ENV_ID = "FrozenLake-v1"
LOG_DIR = "logs/dqn_frozenlake"
ITERS = 1000
BATCH_SIZE = 64
BUFFER_SIZE = 50000
GAMMA = 0.99
LR = 1e-3
EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY = 500

os.makedirs(LOG_DIR, exist_ok=True)

class QNetwork(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, act_dim)
        )

    def forward(self, x):
        return self.fc(x)

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = np.random.choice(len(self.buffer), batch_size, replace=False)
        state, action, reward, next_state, done = zip(*[self.buffer[i] for i in batch])
        return (np.array(state), np.array(action), np.array(reward),
                np.array(next_state), np.array(done))

    def __len__(self):
        return len(self.buffer)

class DQN_Agent:
    def __init__(self, train_env):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.env = train_env

        obs_dim = train_env.observation_space.n
        act_dim = train_env.action_space.n

        self.obs_dim = obs_dim
        self.act_dim = act_dim

        self.policy_net = QNetwork(obs_dim, act_dim).to(self.device)
        self.target_net = QNetwork(obs_dim, act_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LR)

        self.buffer = ReplayBuffer(BUFFER_SIZE)
        self.epsilon = EPS_START
        self.total_steps = 0
        self.training_iterations = 0

    def one_hot(self, obs):
        batch_size = obs.shape[0]
        onehot = np.zeros((batch_size, self.obs_dim), dtype=np.float32)
        onehot[np.arange(batch_size), obs] = 1
        return onehot

    def select_action(self, state):
        if np.random.rand() < self.epsilon:
            return self.env.action_space.sample()
        else:
            state = np.array([state])  # Wrap in batch dimension
            state = torch.FloatTensor(self.one_hot(state)).to(self.device)
            q_values = self.policy_net(state)
            return q_values.argmax().item()

    def train_one_iteration(self):
        obs, _ = self.env.reset()
        done = False
        total_reward = 0
        losses = []

        while not done:
            action = self.select_action(obs)
            next_obs, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated

            self.buffer.push(obs, action, reward, next_obs, done)

            obs = next_obs
            total_reward += reward
            self.total_steps += 1

            self.epsilon = max(EPS_END, EPS_START - (self.total_steps / EPS_DECAY))

            if len(self.buffer) >= BATCH_SIZE:
                loss = self.update()
                losses.append(loss)

        self.training_iterations += 1

        avg_loss = np.mean(losses) if losses else 0
        avg_return = total_reward

        return avg_return, avg_loss, 0, 0  # avg_return, policy_loss, value_loss, entropy (0s)

    def update(self):
        states, actions, rewards, next_states, dones = self.buffer.sample(BATCH_SIZE)

        states = torch.FloatTensor(self.one_hot(states)).to(self.device)
        next_states = torch.FloatTensor(self.one_hot(next_states)).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        curr_q = self.policy_net(states).gather(1, actions)
        next_q = self.target_net(next_states).max(1)[0].unsqueeze(1)
        expected_q = rewards + (1 - dones) * GAMMA * next_q

        loss = nn.MSELoss()(curr_q, expected_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Soft update
        self.target_net.load_state_dict(self.policy_net.state_dict())

        return loss.item()

    def save_model(self, path):
        torch.save(self.policy_net.state_dict(), path)

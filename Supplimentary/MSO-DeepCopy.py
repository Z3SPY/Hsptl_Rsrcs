# Refactored SafeDQN with controlled epsilon decay and improved MSO usage
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
from copy import deepcopy
import matplotlib.pyplot as plt

# Q-network definition
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(QNetwork, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )

    def forward(self, x):
        return self.model(x)

# Replay buffer
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.array, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)

# MSO Planner with Q-value bootstrap
class MSOPlanner:
    def __init__(self, env_maker, q_network, depth=3, branching=3, discount=0.99):
        self.env_maker = env_maker
        self.q_network = q_network
        self.depth = depth
        self.branching = branching
        self.discount = discount

    def compute_return(self, env, initial_action):
        env_copy = self.env_maker()
        env_copy.reset()
        env_copy.unwrapped.state = deepcopy(env.unwrapped.state)

        obs, reward, done, truncated, _ = env_copy.step(initial_action)
        total_reward = reward
        gamma = self.discount

        for _ in range(1, self.depth):
            if done or truncated:
                break
            action = env_copy.action_space.sample()
            obs, reward, done, truncated, _ = env_copy.step(action)
            total_reward += gamma * reward
            gamma *= self.discount

        if not done and not truncated:
            with torch.no_grad():
                obs_t = torch.FloatTensor(obs).unsqueeze(0)
                q_val = self.q_network(obs_t).max().item()
            total_reward += gamma * q_val

        return total_reward

    def best_action(self, env):
        best_action, best_value = None, -float('inf')
        for action in range(env.action_space.n):
            avg_return = np.mean([self.compute_return(env, action) for _ in range(self.branching)])
            if avg_return > best_value:
                best_value = avg_return
                best_action = action
        return best_action, best_value

# SafeDQN Agent
class SafeDQNAgent:
    def __init__(self, state_dim, action_dim):
        self.q_net = QNetwork(state_dim, action_dim)
        self.target_net = QNetwork(state_dim, action_dim)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=1e-3)
        self.replay_buffer = ReplayBuffer(10000)
        self.gamma = 0.99

        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995  # NOTE: Called manually once per episode

    def select_action(self, state, env, mso_planner, use_mso=False):
        if np.random.rand() < self.epsilon:
            return env.action_space.sample()
        state_t = torch.FloatTensor(state).unsqueeze(0)
        q_values = self.q_net(state_t).detach().numpy()[0]
        a_dqn = np.argmax(q_values)
        if use_mso:
            a_mso, mso_val = mso_planner.best_action(env)
            if mso_val > q_values[a_dqn]:
                return a_mso
        return a_dqn

    def train(self, batch_size):
        if len(self.replay_buffer) < batch_size:
            return
        s, a, r, s_, d = self.replay_buffer.sample(batch_size)

        s = torch.FloatTensor(s)
        a = torch.LongTensor(a).unsqueeze(1)
        r = torch.FloatTensor(r).unsqueeze(1)
        s_ = torch.FloatTensor(s_)
        d = torch.FloatTensor(d).unsqueeze(1)

        q_sa = self.q_net(s).gather(1, a)
        q_next = self.target_net(s_).max(1, keepdim=True)[0].detach()
        q_target = r + self.gamma * (1 - d) * q_next

        loss = nn.MSELoss()(q_sa, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target(self):
        self.target_net.load_state_dict(self.q_net.state_dict())

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

# Main training loop

def train(env_name="Acrobot-v1", episodes=800):
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = SafeDQNAgent(state_dim, action_dim)
    mso_planner = MSOPlanner(lambda: gym.make(env_name), agent.q_net, depth=2, branching=3)

    rewards = []

    for ep in range(episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0

        for t in range(500):
            use_mso = (ep > 50 and t % 5 == 0)
            action = agent.select_action(obs, env, mso_planner, use_mso=use_mso)
            next_obs, reward, done, truncated, _ = env.step(action)
            agent.replay_buffer.push(obs, action, reward, next_obs, done or truncated)
            agent.train(batch_size=64)
            obs = next_obs
            total_reward += reward
            if done or truncated:
                break

        agent.decay_epsilon()
        if ep % 10 == 0:
            agent.update_target()
        rewards.append(total_reward)

        if ep % 20 == 0:
            avg = np.mean(rewards[-20:])
            print(f"Episode {ep}, Avg Reward: {avg:.2f}, Epsilon: {agent.epsilon:.3f}")

    env.close()



    plt.figure()
    plt.plot(rewards)
    plt.title(f"MSO Curriculum (first {ep} episodes) for Acrobot")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.show() 

    return rewards


if __name__ == "__main__":
    train()
# MSO Pretraining and MSO Curriculum for Acrobot-v1
# -------------------------------------------------
# This single script demonstrates two advanced ways to incorporate
# MSO into a DQN training process:
#
# 1) OFFLINE MSO PRETRAINING:
#    - Generate expert transitions offline using MSO.
#    - Fill a replay buffer with those transitions.
#    - Pretrain the DQN on this offline data.
#    - Then do normal DQN training.
#
# 2) CURRICULUM MSO:
#    - In early episodes, do MSO-based actions.
#    - After some episodes, turn off MSO and rely on pure DQN.
#
# Both approaches can be tested by calling their respective functions.
# We also include a helper function to plot and compare training curves.
#
# NOTE:
#  - 'Online training' typically means the agent is interacting with the environment in real-time,
#    but you can interpret it as simply the normal step-by-step RL loop.
#  - This code uses Acrobot-v1 for demonstration, but you can adapt it to other Gym environments.

import gymnasium as gym
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from collections import deque
from copy import deepcopy

# --------------------------
#   1) Q-Network
# --------------------------
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(QNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, x):
        return self.net(x)

# --------------------------
#   2) Replay Buffer
# --------------------------
class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size=64):
        batch = random.sample(self.buffer, batch_size)
        s, a, r, s2, d = zip(*batch)
        return (np.array(s),
                np.array(a),
                np.array(r),
                np.array(s2),
                np.array(d))

    def __len__(self):
        return len(self.buffer)

# --------------------------
#   3) MSO Planner
# --------------------------
class MSOPlanner:
    def __init__(self, env_maker, q_net, depth=3, branching=3, discount=0.99):
        """
        Offline MSO: We'll randomly sample actions up to 'depth' steps,
        then bootstrap at the end with Q(s). This helps generate better transitions.
        """
        self.env_maker = env_maker
        self.q_net = q_net  # for terminal-value bootstrap
        self.depth = depth
        self.branching = branching
        self.discount = discount

    def rollout(self, env, initial_obs):
        """Simulate an MSO trajectory from initial_obs. Returns transitions list."""
        transitions = []
        obs = deepcopy(initial_obs)
        done = False
        truncated = False
        discount_factor = 1.0
        total_reward = 0.0

        # We'll pick a single action at t=0, then random for next steps.
        # You could also do a tree search, etc.
        best_action = None
        best_return = -float('inf')
        possible_actions = range(env.action_space.n)
        # Evaluate each action's short-horizon
        for a in possible_actions:
            sum_returns = 0.0
            for _ in range(self.branching):
                sum_returns += self.compute_short_horizon_return(env, obs, a)
            avg_val = sum_returns / self.branching
            if avg_val > best_return:
                best_return = avg_val
                best_action = a

        # Now, let's do 1 step with best_action
        next_obs, reward, done, truncated, _ = env.step(best_action)
        transitions.append((obs, best_action, reward, next_obs, done or truncated))
        obs = next_obs
        return transitions

    def compute_short_horizon_return(self, env, obs, initial_action):
        # Clone env
        env_copy = self.env_maker()
        env_copy.reset()
        env_copy.unwrapped.state = deepcopy(env.unwrapped.state)
        # Step initial action
        next_obs, reward, done, truncated, _ = env_copy.step(initial_action)
        total_reward = reward
        gamma = self.discount

        for _ in range(1, self.depth):
            if done or truncated:
                break
            # random action
            a = env_copy.action_space.sample()
            next_obs, r, done, truncated, _ = env_copy.step(a)
            total_reward += gamma * r
            gamma *= self.discount

        # terminal bootstrap
        if not done and not truncated:
            with torch.no_grad():
                obs_t = torch.FloatTensor(next_obs).unsqueeze(0)
                q_val = self.q_net(obs_t).max().item()
            total_reward += gamma * q_val
        return total_reward
    
    def best_action(self, env):
        obs = deepcopy(env.unwrapped.state)
        best_action = None
        best_value = -float('inf')
        for a in range(env.action_space.n):
            avg_value = np.mean([
                self.compute_short_horizon_return(env, obs, a)
                for _ in range(self.branching)
            ])
            if avg_value > best_value:
                best_value = avg_value
                best_action = a
        return best_action, best_value

# --------------------------
#   4) DQN Agent
# --------------------------
class DQNAgent:
    def __init__(self, state_dim, action_dim, lr=1e-3, gamma=0.99,
                 epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995,
                 buffer_capacity=10000):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_end
        self.epsilon_decay = epsilon_decay

        self.q_net = QNetwork(state_dim, action_dim)
        self.target_net = QNetwork(state_dim, action_dim)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)

        self.replay_buffer = ReplayBuffer(capacity=buffer_capacity)

    def select_action(self, state, env):
        if np.random.rand() < self.epsilon:
            return env.action_space.sample()
        state_t = torch.FloatTensor(state).unsqueeze(0)
        q_values = self.q_net(state_t).detach().numpy()[0]
        return np.argmax(q_values)

    def train_step(self, batch_size=64):
        if len(self.replay_buffer) < batch_size:
            return
        s, a, r, s2, d = self.replay_buffer.sample(batch_size)

        s_t = torch.FloatTensor(s)
        a_t = torch.LongTensor(a).unsqueeze(1)
        r_t = torch.FloatTensor(r).unsqueeze(1)
        s2_t = torch.FloatTensor(s2)
        d_t = torch.FloatTensor(d).unsqueeze(1)

        q_vals = self.q_net(s_t).gather(1, a_t)
        with torch.no_grad():
            q_next = self.target_net(s2_t).max(dim=1, keepdim=True)[0]
            q_target = r_t + self.gamma * (1 - d_t) * q_next

        loss = nn.MSELoss()(q_vals, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target(self):
        self.target_net.load_state_dict(self.q_net.state_dict())

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


# -----------------------------------------------------------------------------
#   5) OFFLINE MSO PRETRAINING
# -----------------------------------------------------------------------------
def run_mso_pretrain_acrobot(
    episodes=500,
    pretrain_steps=2000,
    depth=5, branching=5,
    batch_size=64,  
    plot_results=True
):
    """
    Steps:
      1) Use MSO to generate transitions offline from random states.
      2) Fill replay buffer with these transitions.
      3) Pretrain the DQN (no env stepping, just learning from this data).
      4) Then train DQN as usual.

    returns: list of average rewards per episode.
    """
    env = gym.make("Acrobot-v1")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = DQNAgent(state_dim, action_dim)
    # We'll build an MSO planner using the agent's Q-net for terminal bootstrap
    mso = MSOPlanner(lambda: gym.make("Acrobot-v1"), agent.q_net,
                     depth=depth, branching=branching)

    # --------------------------------
    # 1) Generate Expert Data Offline
    # --------------------------------
    # We'll randomly sample states, run MSO to get some transitions.
    offline_transitions = 0
    while offline_transitions < pretrain_steps:
        # We can just reset the real env and clone it.
        # or sample random states if you like.
        obs, _ = env.reset()
        # single MSO rollout
        trans = mso.rollout(env, obs)
        for (s0, a0, rew, s1, dn) in trans:
            agent.replay_buffer.push(s0, a0, rew, s1, dn)
        offline_transitions += len(trans)

    # ----------------------
    # 2) Pretrain the DQN
    # ----------------------
    pretrain_epochs = 1000  # arbitrary, can tweak
    for _ in range(pretrain_epochs):
        agent.train_step(batch_size)

    agent.update_target()

    # ----------------------
    # 3) Normal Online Training
    # ----------------------
    rewards_history = []
    max_steps = 500

    for ep in range(episodes):
        obs, _ = env.reset()
        total_reward = 0
        for t in range(max_steps):
            action = agent.select_action(obs, env)
            next_obs, reward, done, truncated, _ = env.step(action)
            agent.replay_buffer.push(obs, action, reward, next_obs, done or truncated)

            agent.train_step(batch_size)
            obs = next_obs
            total_reward += reward
            if done or truncated:
                break
        agent.update_target()
        agent.decay_epsilon()
        rewards_history.append(total_reward)

    env.close()

    if plot_results:
        plt.figure()
        plt.plot(rewards_history)
        plt.title("Offline MSO Pretrain for Acrobot")
        plt.xlabel("Episode")
        plt.ylabel("Total Reward")
        plt.show()

    return rewards_history


# -----------------------------------------------------------------------------
#   6) CURRICULUM MSO TRAINING
# -----------------------------------------------------------------------------
def run_mso_curriculum_acrobot(
    episodes=500,
    mso_episodes=100,
    depth=2, branching=3,
    batch_size=64,
    plot_results=True
):
    """
    Steps:
      1) For the first 'mso_episodes' episodes, agent uses MSO-based actions.
      2) After that, switch to standard DQN actions.

    returns: list of total rewards per episode.
    """
    env = gym.make("Acrobot-v1")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = DQNAgent(state_dim, action_dim)
    # We'll build MSO for action selection in early episodes only
    mso = MSOPlanner(lambda: gym.make("Acrobot-v1"), agent.q_net, depth=depth, branching=branching)

    rewards_history = []
    max_steps = 500

    for ep in range(episodes):
        obs, _ = env.reset()
        total_reward = 0.0
        for t in range(max_steps):
            if ep < mso_episodes:
                # use MSO to find best action
                # we do a short-horizon approach each step to find best action
                # that might be slow, but it's a demonstration.
                best_a, _ = mso.best_action(env)
                action = best_a
            else:
                # standard DQN action
                action = agent.select_action(obs, env)

            next_obs, reward, done, truncated, _ = env.step(action)
            agent.replay_buffer.push(obs, action, reward, next_obs, done or truncated)
            agent.train_step(batch_size)
            obs = next_obs
            total_reward += reward
            if done or truncated:
                break

        agent.update_target()
        agent.decay_epsilon()
        rewards_history.append(total_reward)

    env.close()

    if plot_results:
        plt.figure()
        plt.plot(rewards_history)
        plt.title(f"MSO Curriculum (first {mso_episodes} episodes) for Acrobot")
        plt.xlabel("Episode")
        plt.ylabel("Total Reward")
        plt.show()

    return rewards_history


# -----------------------------------------------------------------------------
#   7) Example Runner & Plot Comparison
# -----------------------------------------------------------------------------
def compare_two_runs():
    """
    Example of how you might run both methods & compare in a single figure.
    """
    # 1) Offline MSO Pretraining
    rewards_pretrain = run_mso_pretrain_acrobot(
        episodes=500,
        pretrain_steps=1000,
        depth=7,
        branching=5,
        batch_size=64,
        plot_results=False
    )

    # 2) Curriculum MSO
    rewards_curriculum = run_mso_curriculum_acrobot(
        episodes=500,
        mso_episodes=50,
        depth=7,
        branching=5,
        batch_size=64,
        plot_results=False
    )

    # Plot both on one chart
    plt.figure()
    plt.plot(rewards_pretrain, label="Offline Pretrain")
    plt.plot(rewards_curriculum, label="Curriculum MSO")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Comparison: Pretrain vs Curriculum")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    # Example usage:
    # 1) Run offline MSO pretraining approach
    # run_mso_pretrain_acrobot(plot_results=True)

    # 2) Run MSO curriculum approach
    # run_mso_curriculum_acrobot(plot_results=True)

    # 3) Compare them in a single plot
    compare_two_runs()

    # You can comment/uncomment whichever approach you want to test.

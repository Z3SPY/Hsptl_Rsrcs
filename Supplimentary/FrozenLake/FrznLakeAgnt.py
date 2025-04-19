import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class PolicyNetwork(nn.Module):
    def __init__(self, n_states, n_actions):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_states, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        return self.net(x)

class Agent:
    def __init__(self, n_states, n_actions, lr=1e-3, gamma=0.99):
        self.n_states = n_states
        self.n_actions = n_actions
        self.policy = PolicyNetwork(n_states, n_actions)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.gamma = gamma

    def select_action(self, state):
        x = np.zeros(self.n_states, dtype=np.float32)
        x[state] = 1.0
        x_tensor = torch.from_numpy(x).unsqueeze(0)
        probs = self.policy(x_tensor)
        probs = probs.clamp(min=1e-8, max=1.0)
        probs = probs / probs.sum(dim=-1, keepdim=True)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action)

    def update(self, rewards, log_probs):
        R = 0
        returns = []
        for r in reversed(rewards):
            R = r + self.gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns, dtype=torch.float32)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        loss = torch.stack([-log_prob * R for log_prob, R in zip(log_probs, returns)]).sum()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

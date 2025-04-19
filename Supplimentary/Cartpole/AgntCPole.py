import torch
import torch.nn as nn
import torch.optim as optim

class PolicyNetwork(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, act_dim),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        return self.net(x)

class Agent:
    def __init__(self, obs_dim, act_dim, lr=1e-3, gamma=0.99):
        self.policy = PolicyNetwork(obs_dim, act_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.gamma = gamma

    def select_action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        probs = self.policy(state)

        # ✅ Clamp to prevent NaNs
        probs = probs.clamp(min=1e-8, max=1.0)
        probs = probs / probs.sum(dim=-1, keepdim=True)

        # ⚠️ Check for NaNs before proceeding
        if torch.isnan(probs).any():
            print("❌ NaN detected in action probs. Resetting.")
            print("State:", state)
            print("Raw probs:", probs)
            raise ValueError("Policy returned NaN probabilities.")

        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action)


    def update(self, rewards, log_probs):
        discounted = []
        R = 0
        for r in reversed(rewards):
            R = r + self.gamma * R
            discounted.insert(0, R)
        discounted = torch.tensor(discounted)
        if len(discounted) > 1:
            discounted = (discounted - discounted.mean()) / (discounted.std() + 1e-8)
        else:
            discounted = discounted
        
        loss = torch.stack([-log_prob * R for log_prob, R in zip(log_probs, discounted)]).sum()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

import gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import copy
from collections import deque
import matplotlib.pyplot as plt

# ---------------------
# Hyperparameters
# ---------------------
BUFFER_SIZE = 100000
BATCH_SIZE = 128
GAMMA = 0.99
EPSILON_START = 1.0
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.995  # faster decay for acrobot
LEARNING_RATE = 0.0005
GRAD_CLIP = 5.0
MSO_SCENARIOS = 5        # number of scenario rollouts to generate
MSO_HORIZON = 5          # how many steps to simulate in each scenario
SCENARIO_PROB = 0.3      # probability to perform scenario generation

np.float_ = np.float64  # Monkey-patch to maintain backward compatibility
np.bool8 = np.bool_

# ---------------------
# Prioritized Replay Buffer
# ---------------------
class PrioritizedReplayBuffer:
    def __init__(self, capacity=BUFFER_SIZE, alpha=0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.pos = 0
        self.max_prio = 1.0

    def add(self, experience):
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
        else:
            self.buffer[self.pos] = experience
        self.priorities[self.pos] = self.max_prio
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size, beta=0.4):
        if len(self.buffer) == 0:
            return [], [], []
        probs = self.priorities[:len(self.buffer)] ** self.alpha
        probs /= probs.sum()
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        weights = (len(self.buffer) * probs[indices]) ** (-beta)
        weights /= weights.max()
        samples = [self.buffer[i] for i in indices]
        return samples, indices, weights

    def update_priorities(self, indices, priorities):
        for idx, prio in zip(indices, priorities):
            self.priorities[idx] = prio
        self.max_prio = max(self.max_prio, np.max(priorities))


# ---------------------
# Neural Network for Acrobot
# ---------------------
class AcrobotDQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(AcrobotDQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_size)
        )
        
    def forward(self, x):
        return self.net(x)


# ---------------------
# DRL + MSO Agent for Acrobot
# ---------------------
class AcrobotMSODQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = PrioritizedReplayBuffer()
        self.model = AcrobotDQN(state_size, action_size)
        self.target_model = AcrobotDQN(state_size, action_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=LEARNING_RATE)
        self.loss_fn = nn.MSELoss()
        self.epsilon = EPSILON_START
        self.update_target()

    def update_target(self):
        self.target_model.load_state_dict(self.model.state_dict())
        
    def act(self, state):
        if random.random() < self.epsilon:
            return random.randrange(self.action_size)
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = self.model(state_tensor)
        return torch.argmax(q_values).item()
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.add((state, action, reward, next_state, done))
    
    def replay(self):
        if len(self.memory.buffer) < BATCH_SIZE:
            return

        samples, indices, weights = self.memory.sample(BATCH_SIZE)
        states, actions, rewards, next_states, dones = zip(*samples)
        
        states = torch.FloatTensor(np.array(states))
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(np.array(next_states))
        dones = torch.FloatTensor(dones)
        weights = torch.FloatTensor(weights)
        
        current_q = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_q = self.target_model(next_states).max(1)[0]
        target_q = rewards + (1 - dones) * GAMMA * next_q
        
        loss = (weights * self.loss_fn(current_q, target_q)).mean()
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), GRAD_CLIP)
        self.optimizer.step()
        
        # Update priorities using TD error
        with torch.no_grad():
            td_errors = (target_q - current_q).abs().detach().cpu().numpy()
        self.memory.update_priorities(indices, td_errors + 1e-5)
        
        # Decay epsilon
        if self.epsilon > EPSILON_MIN:
            self.epsilon *= EPSILON_DECAY

    def generate_scenarios(self, env, state):
        """
        Generate MSO scenarios by branching out from the current state.
        This function assumes that the Acrobot environment allows setting its state
        via env.env.state. Not all Gym environments support this directly.
        """
        scenarios = []
        # Save the current environment state
        try:
            saved_state = env.env.state.copy()
        except Exception as e:
            print("Warning: Could not access env.env.state, using provided state.")
            saved_state = state.copy()
        
        for _ in range(MSO_SCENARIOS):
            # Restore original state
            try:
                env.env.state = saved_state.copy()
            except:
                pass
            # Perturb state slightly
            perturbed_state = saved_state + np.random.normal(0, 0.05, size=saved_state.shape)
            try:
                env.env.state = perturbed_state.copy()
            except:
                pass
            scenario = []
            current_state = perturbed_state.copy()
            for _ in range(MSO_HORIZON):
                action = self.act(current_state)
                next_state, reward, done, truncated, info = env.step(action)
                scenario.append((current_state, action, reward, next_state, done))
                if done:
                    break
                current_state = next_state.copy()
            scenarios.append(scenario)
        # Restore the original environment state
        try:
            env.env.state = saved_state.copy()
        except:
            pass
        return scenarios


# ---------------------
# Training Loop
# ---------------------
def train():
    env = gym.make("Acrobot-v1")
    state_size = env.observation_space.shape[0]  # typically 6 for Acrobot
    action_size = env.action_space.n             # typically 3 for Acrobot-v1
    agent = AcrobotMSODQNAgent(state_size, action_size)
    episodes = 1000
    scores = []
    avg_scores = []

    for episode in range(episodes):
        state = env.reset()[0]  # Gym v0.26+ returns (obs, info)
        total_reward = 0
        done = False
        
        while not done:
            action = agent.act(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            agent.remember(state, action, reward, next_state, done)
            
            # Occasionally generate extra scenario rollouts (MSO style)
            if random.random() < SCENARIO_PROB:
                scenarios = agent.generate_scenarios(env, state)
                for scenario in scenarios:
                    for exp in scenario:
                        # Add each experience from the scenario to the replay buffer
                        agent.remember(*exp)
            
            agent.replay()
            state = next_state
            total_reward += reward
        
        # Update target network every 20 episodes
        if episode % 20 == 0:
            agent.update_target()
        
        scores.append(total_reward)
        avg_score = np.mean(scores[-50:])
        avg_scores.append(avg_score)
        print(f"Episode {episode:4d} | Score: {total_reward:4.2f} | Avg: {avg_score:4.2f} | ε: {agent.epsilon:.3f}")
    
    # Plot training progress
    plt.plot(avg_scores)
    plt.xlabel("Episodes")
    plt.ylabel("Average Score (over 50 episodes)")
    plt.title("Acrobot Training with DRL + MSO")
    plt.show()

if __name__ == "__main__":
    train()

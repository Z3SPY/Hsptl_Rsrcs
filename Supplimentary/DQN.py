import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import matplotlib.pyplot as plt

class StandardDQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=100000)
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.998
        self.batch_size = 128
        self.tau = 0.005  # For soft target updates
        
        # Q-Network
        self.q_network = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_size)
        )
        # Target Network
        self.target_network = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_size))
        
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=0.0005)
        self.loss_fn = nn.MSELoss()
        self.update_target_network()
        
    def update_target_network(self):
        # Soft update (Polyak averaging)
        for target_param, q_param in zip(self.target_network.parameters(), 
                                        self.q_network.parameters()):
            target_param.data.copy_(
                self.tau * q_param.data + (1.0 - self.tau) * target_param.data)
        
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        
    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state = torch.FloatTensor(state)
        with torch.no_grad():
            act_values = self.q_network(state)
        return torch.argmax(act_values).item()
    
    def replay(self):
        if len(self.memory) < self.batch_size:
            return
            
        minibatch = random.sample(self.memory, self.batch_size)
        states = torch.FloatTensor([x[0] for x in minibatch])
        actions = torch.LongTensor([x[1] for x in minibatch])
        rewards = torch.FloatTensor([x[2] for x in minibatch])
        next_states = torch.FloatTensor([x[3] for x in minibatch])
        dones = torch.FloatTensor([x[4] for x in minibatch])
        
        # Get current Q values
        current_q = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Compute target Q values
        with torch.no_grad():
            next_q = self.target_network(next_states).max(1)[0]
            target_q = rewards + (1 - dones) * self.gamma * next_q
        
        # Compute loss
        loss = self.loss_fn(current_q.squeeze(), target_q)
        
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 10)
        self.optimizer.step()
        
        # Epsilon decay
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

def train_baseline():
    env = gym.make("LunarLander-v3")
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = StandardDQNAgent(state_size, action_size)
    
    scores = []
    avg_scores = []
    
    for episode in range(2000):
        state, _ = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            action = agent.act(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            
            agent.replay()
            agent.update_target_network()  # Continuous soft update
        
        scores.append(total_reward)
        avg_score = np.mean(scores[-100:])
        avg_scores.append(avg_score)
        
        print(f"Ep {episode+1:03d} | Score: {total_reward:7.2f} | Avg: {avg_score:7.2f} | ε: {agent.epsilon:.3f}")
        
        #if avg_score >= 200:
        #    print(f"Solved in {episode+1} episodes!")
        #    break
    
    # Plot results
    plt.plot(avg_scores)
    plt.xlabel("Episodes")
    plt.ylabel("Average Score (100-episode window)")
    plt.title("Standard DQN Performance")
    plt.savefig("baseline_training_curve.png")
    plt.show()
    
    env.close()

if __name__ == "__main__":
    train_baseline()
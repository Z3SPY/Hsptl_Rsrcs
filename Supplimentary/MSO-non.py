import gymnasium as gym
import numpy as np
import random
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# Set device to CUDA if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------------
# Q-Network definition using PyTorch
# -------------------------------
class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )
        
    def forward(self, x):
        return self.net(x)

# -------------------------------
# Replay Buffer
# -------------------------------
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        # Expand dims for state and next_state to maintain consistency for batch sampling
        state = np.expand_dims(state, 0)
        next_state = np.expand_dims(next_state, 0)
        self.buffer.append((state, action, reward, next_state, done))
        
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        # For states and next_states, we concatenate as they are already arrays with extra dims
        states = np.concatenate([item[0] for item in batch])
        # For actions, rewards, and dones, use np.array to get proper dimensions
        actions = np.array([item[1] for item in batch])
        rewards = np.array([item[2] for item in batch])
        next_states = np.concatenate([item[3] for item in batch])
        dones = np.array([item[4] for item in batch])
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        return len(self.buffer)

# -------------------------------
# Epsilon-Greedy Action Selection
# -------------------------------
def epsilon_greedy_policy(model, state, epsilon, action_dim):
    if random.random() > epsilon:
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            q_values = model(state_tensor)
        action = q_values.max(1)[1].item()
    else:
        action = random.randrange(action_dim)
    return action

# -------------------------------
# Compute TD Loss for one training step
# -------------------------------
def compute_td_loss(model, target_model, optimizer, replay_buffer, batch_size, gamma):
    state, action, reward, next_state, done = replay_buffer.sample(batch_size)
    
    # Convert to torch tensors and move to the selected device
    state      = torch.FloatTensor(state).to(device)
    next_state = torch.FloatTensor(next_state).to(device)
    action     = torch.LongTensor(actions := action).to(device)
    reward     = torch.FloatTensor(rewards := reward).to(device)
    done       = torch.FloatTensor(dones := done).to(device)
    
    # Get current Q values from the model for the taken actions
    q_values = model(state)
    q_value = q_values.gather(1, action.unsqueeze(1)).squeeze(1)
    
    # Compute the next Q values from the target network
    next_q_values = target_model(next_state)
    next_q_value = next_q_values.max(1)[0]
    
    # Compute expected Q value using the Bellman equation
    expected_q_value = reward + gamma * next_q_value * (1 - done)
    
    # Calculate the loss (mean squared error)
    loss = nn.MSELoss()(q_value, expected_q_value.detach())
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return loss

# -------------------------------
# Main training loop
# -------------------------------
def train():
    # Create the Acrobot environment
    env = gym.make("Acrobot-v1")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    # Hyperparameters
    num_episodes = 500
    batch_size = 64
    gamma = 0.99
    replay_buffer_capacity = 10000
    learning_rate = 1e-3
    target_update_interval = 100  # steps after which to update the target network
    epsilon_start = 1.0
    epsilon_final = 0.01
    epsilon_decay = 500  # controls the rate of exponential decay
    
    # Initialize the main and target networks
    model = DQN(state_dim, action_dim).to(device)
    target_model = DQN(state_dim, action_dim).to(device)
    target_model.load_state_dict(model.state_dict())
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    replay_buffer = ReplayBuffer(replay_buffer_capacity)
    
    episode_rewards = []
    steps_done = 0
    
    for episode in range(num_episodes):
        state, _ = env.reset()  # Gymnasium returns a tuple (observation, info)
        episode_reward = 0
        done = False
        
        while not done:
            # Calculate epsilon for current step (exponential decay)
            epsilon = epsilon_final + (epsilon_start - epsilon_final) * np.exp(-1. * steps_done / epsilon_decay)
            
            # Select action using the epsilon-greedy policy
            action = epsilon_greedy_policy(model, state, epsilon, action_dim)
            
            # Take action in the environment
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Store the transition in the replay buffer
            replay_buffer.push(state, action, reward, next_state, done)
            state = next_state
            episode_reward += reward
            steps_done += 1
            
            # Perform a training step if there are enough samples in the buffer
            if len(replay_buffer) > batch_size:
                loss = compute_td_loss(model, target_model, optimizer, replay_buffer, batch_size, gamma)
                
            # Update the target network periodically
            if steps_done % target_update_interval == 0:
                target_model.load_state_dict(model.state_dict())
        
        episode_rewards.append(episode_reward)
        print(f"Episode {episode:03d} | Reward: {episode_reward:3.1f} | Epsilon: {epsilon:.3f}")



    plt.figure()
    plt.plot(episode_rewards)
    plt.title(f"MSO Curriculum (first {episode} episodes) for Acrobot")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.show() 
    
    # Save the trained model
    torch.save(model.state_dict(), "dqn_acrobot.pth")
    print("Training completed and model saved as dqn_acrobot.pth")
    env.close()

if __name__ == "__main__":
    train()

# File: Agent.py
import stable_baselines3 as sb3
import torch

class HybridAgent:
    """
    Wrapper for the DRL agent using DQN from Stable Baselines3.
    This agent handles discrete actions and can be extended with custom policies.
    """
    def __init__(self, env, use_mso=True):

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print("Using device:", device)

        self.use_mso = use_mso
        # Initialize a DQN agent on the given environment
        self.model = sb3.DQN("MlpPolicy", env, verbose=1, device=device)
    
    def train(self, timesteps=10000):
        """
        Train the DQN agent for a given number of timesteps.
        """
        self.model.learn(total_timesteps=timesteps)
    
    def act(self, obs):
        """
        Get action from the trained agent (for deployment or evaluation).
        """
        action, _ = self.model.predict(obs, deterministic=True)
        return action

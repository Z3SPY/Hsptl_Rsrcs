# File: Agent.py (PPO implementation with custom Torch-based hyperparameters)

from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.vec_env import DummyVecEnv
import torch

class HybridAgent:
    def __init__(self, env, device='cuda'):
        self.env = env
        self.device = device

        # Optional: Wrap env if needed
        self.vec_env = DummyVecEnv([lambda: env])

        # Custom PPO parameters
        custom_params = {
            'policy': 'MlpPolicy',
            'env': self.vec_env,
            'learning_rate': 3e-4,
            'n_steps': 2048,
            'batch_size': 64,
            'n_epochs': 10,
            'gamma': 0.99,
            'gae_lambda': 0.95,
            'clip_range': 0.2,
            'ent_coef': 0.01,
            'vf_coef': 0.5,
            'max_grad_norm': 0.5,
            'verbose': 1,
            'device': self.device
        }

        # Initialize PPO
        self.model = PPO(**custom_params)

    def train(self, timesteps=10000):
        self.model.learn(total_timesteps=timesteps)

    def act(self, obs):
        action, _ = self.model.predict(obs, deterministic=True)
        return action

    def get_value_function(self):
        return self.model.policy

    def save(self, path="ppo_model"):
        self.model.save(path)

    def load(self, path="ppo_model"):
        self.model = PPO.load(path, env=self.vec_env, device=self.device)


    # I am not using this Idk where to put it
    def debug_distribution(self, obs):
        """
        (Manual 5) Print the underlying distribution 
        for each sub-action in MultiDiscrete.
        """
        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device)

        # SB3 internal method to get distribution
        dist = self.model.policy.get_distribution(obs_tensor)

        print("[DEBUG] Dist details:", dist.distribution)
        # This will typically show a MultiCategorical distribution with sub-categoricals
# ADD THIS:
from stable_baselines3.common.callbacks import BaseCallback

class PlannerControlCallback(BaseCallback):
    def __init__(self, env, disable_after=5000, verbose=1):
        super().__init__(verbose)
        self.env = env
        self.disable_after = disable_after

    def _on_step(self) -> bool:
        if self.num_timesteps >= self.disable_after:
            if self.env.use_mso:
                self.env.use_mso = False
                if self.verbose:
                    print(f"[PlannerControlCallback] MSO planner disabled at step {self.num_timesteps}")
        return True
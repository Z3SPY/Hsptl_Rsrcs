# ppo_hospital_train.py

import os
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from hospital_env import HospitalSimEnv  # Your environment

# === Simulation Configuration ===
default_sim_config = {
    "n_triage": 2,
    "n_reg": 2,
    "n_exam": 3,
    "n_trauma": 2,
    "n_cubicles_1": 3,
    "n_cubicles_2": 2,
    "random_number_set": 1,
    "prob_trauma": 0.12,
}

# === Environment Setup ===
def create_env():
    env = HospitalSimEnv(sim_config=default_sim_config, step_size=60, rc_period=20160)
    return env

# Use DummyVecEnv wrapper for Stable-Baselines3
env = DummyVecEnv([create_env])

# === PPO Model Setup ===
model = PPO(
    policy="MlpPolicy",
    env=env,
    learning_rate=3e-4,
    n_steps=2048,           # How many steps to run per rollout
    batch_size=64,          # Mini-batch size
    n_epochs=10,            # Number of training epochs per update
    gamma=0.995,            # Discount factor (make agent care about long term)
    gae_lambda=0.95,        # GAE advantage smoothing
    clip_range=0.2,         # PPO clipping range
    ent_coef=0.01,          # Encourage exploration slightly
    verbose=1,
    tensorboard_log="./ppo_hospital_tensorboard/"
)

# === Training ===
TIMESTEPS = 100_000
model.learn(total_timesteps=TIMESTEPS)

# Save model
model_save_path = "./ppo_hospital_agent"
os.makedirs(model_save_path, exist_ok=True)
model.save(os.path.join(model_save_path, "ppo_model"))

print("Training complete and model saved.")

# === Evaluation ===
# Create fresh evaluation environment
eval_env = create_env()

obs = eval_env.reset()
done = False
total_reward = 0
step = 0

while not done:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = eval_env.step(action)
    total_reward += reward
    step += 1

print(f"\nEvaluation Result:")
print(f"Total reward: {total_reward:.2f}")
print(f"Total steps: {step}")

eval_env.close()

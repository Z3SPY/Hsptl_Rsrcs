# ppo_hospital_train.py (Final, Plug-and-Play Version)

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from hospital_env import HospitalSimEnv
import numpy as np
import csv
import os

# Base simulation config
default_sim_config = {
    "n_triage": 2,
    "n_reg": 2,
    "n_exam": 3,
    "n_trauma": 2,
    "n_cubicles_1": 3,
    "n_cubicles_2": 2,
    "random_number_set": 1,
    'n_icu': 5,
    'n_ward': 10,
    "prob_trauma": 0.12,
}

# Environment factory
def make_env():
    return HospitalSimEnv(sim_config=default_sim_config, step_size=15, rc_period=40320, cost_mode='diverse')

# Wrap environment
env = DummyVecEnv([make_env])

# PPO Model
model = PPO(
    policy="MlpPolicy",
    env=env,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.995,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01,
    verbose=1,
    tensorboard_log="./ppo_hospital_tensorboard/"
)

# Training control 
csv_file = "ppo_hospital_training_log.csv"
log_interval = 5000
total_timesteps = 6000
sim_env = env.envs[0]

# Setup CSV
csv_header = [
    "timesteps",
    "sim_time_minutes",
    "avg_reward",
    "avg_triage_wait",
    "avg_registration_wait",
    "avg_exam_wait",
    "avg_trauma_wait",
    "avg_non_trauma_treat_wait",
    "avg_trauma_treat_wait",
    "avg_fatigue",
    "avg_throughput"
]
if not os.path.exists(csv_file):
    with open(csv_file, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(csv_header)

# Train Loop
steps_so_far = 0
while steps_so_far < total_timesteps:
    model.learn(total_timesteps=log_interval, reset_num_timesteps=False)
    steps_so_far += log_interval

    # Calculate averages
    avg_reward = np.mean(sim_env.cumulative_rewards) if sim_env.cumulative_rewards else 0.0
    avg_triage_wait = np.mean(sim_env.cumulative_triage_waits) if sim_env.cumulative_triage_waits else 0.0
    avg_registration_wait = np.mean(sim_env.cumulative_registration_waits) if sim_env.cumulative_registration_waits else 0.0
    avg_exam_wait = np.mean(sim_env.cumulative_exam_waits) if sim_env.cumulative_exam_waits else 0.0
    avg_trauma_wait = np.mean(sim_env.cumulative_trauma_waits) if sim_env.cumulative_trauma_waits else 0.0
    avg_cub1_wait = np.mean(sim_env.cumulative_cub1_waits) if sim_env.cumulative_cub1_waits else 0.0
    avg_cub2_wait = np.mean(sim_env.cumulative_cub2_waits) if sim_env.cumulative_cub2_waits else 0.0
    avg_fatigue = np.mean(sim_env.cumulative_fatigues) if sim_env.cumulative_fatigues else 0.0
    avg_throughput = np.mean(sim_env.cumulative_throughput) if sim_env.cumulative_throughput else 0.0

    # Log to CSV
    with open(csv_file, mode='a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            steps_so_far,
            sim_env.current_time,
            avg_reward,
            avg_triage_wait,
            avg_registration_wait,
            avg_exam_wait,
            avg_trauma_wait,
            avg_cub1_wait,
            avg_cub2_wait,
            avg_fatigue,
            avg_throughput
        ])

    # Reset internal trackers
    sim_env.cumulative_rewards.clear()
    sim_env.cumulative_triage_waits.clear()
    sim_env.cumulative_registration_waits.clear()
    sim_env.cumulative_exam_waits.clear()
    sim_env.cumulative_trauma_waits.clear()
    sim_env.cumulative_cub1_waits.clear()
    sim_env.cumulative_cub2_waits.clear()
    sim_env.cumulative_fatigues.clear()
    sim_env.cumulative_throughput.clear()

    print(f"[INFO] Logged at {steps_so_far} timesteps!")

# Final Save
model.save("ppo_hospital_final")
print("[DONE] Model saved.")

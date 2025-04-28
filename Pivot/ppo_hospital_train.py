# ppo_hospital_train.py
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from hospital_env import HospitalSimEnv
import numpy as np
import torch

# Base simulation config
default_sim_config = {
    "n_triage": 2,
    "n_reg": 2,
    "n_exam": 3,
    "n_trauma": 2,
    "n_cubicles_1": 3,
    "n_cubicles_2": 2,
    "random_number_set": 1,
    "prob_trauma": 0.12,
    "override_arrival_rate": True,
    "manual_arrival_rate": 500
}

def make_env():
    return HospitalSimEnv(sim_config=default_sim_config, step_size=15, rc_period=40320, cost_mode='diverse')

# Environment setup
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

# Training Loop
total_timesteps = 200_000
log_interval = 5000
timesteps_so_far = 0
episode_rewards = []

# === NEW DEBUG TRACKING ===
shift_rewards = []
shift_discharge_rates = []
shift_avg_fatigues = []
shift_queue_sizes = []
shift_efficiencies = []

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device")

while timesteps_so_far < total_timesteps:
    model.learn(total_timesteps=log_interval, reset_num_timesteps=False)
    timesteps_so_far += log_interval

    # Manual evaluation
    sim_env = env.envs[0]
    obs = sim_env.reset()
    if isinstance(obs, tuple):
        obs = obs[0]

    done = False
    total_reward = 0.0
    total_discharges = 0
    total_queue = 0
    fatigue_sum = 0.0
    total_steps = 0
    efficiency_sum = 0.0

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        action = np.array(action)
        obs, reward, terminated, truncated, _ = sim_env.step(action)
        done = terminated or truncated
        total_reward += reward
        total_steps += 1

        # Pull performance info
        total_discharges += sim_env._count_shift_discharges()
        fatigue_by_unit = sim_env._average_fatigue_by_unit()
        fatigue_sum += np.mean(list(fatigue_by_unit.values()))
        current_queue = sum(obs[i] for i in [1,3,5,7,9,11])
        total_queue += current_queue

        active_staff_total = sum(sim_env.active_staff.values())
        if active_staff_total > 0:
            efficiency_sum += total_discharges / active_staff_total

    # Save tracking
    shift_rewards.append(total_reward)
    shift_discharge_rates.append(total_discharges / (total_steps/8))  # normalized to shift periods
    shift_avg_fatigues.append(fatigue_sum / total_steps)
    shift_queue_sizes.append(total_queue / total_steps)
    shift_efficiencies.append(efficiency_sum / total_steps)

    print("\n=== Training Progress Snapshot ===")
    print(f"Timesteps So Far: {timesteps_so_far}")
    print(f"Avg Reward this Month: {total_reward/total_steps:.2f}")
    print(f"Avg Discharge Rate: {shift_discharge_rates[-1]:.2f} patients per shift")
    print(f"Avg Fatigue per Staff: {shift_avg_fatigues[-1]:.2f}")
    print(f"Avg Total Queue Size: {shift_queue_sizes[-1]:.2f}")
    print(f"Avg Efficiency (Discharges / Staff): {shift_efficiencies[-1]:.4f}")
    print("="*80)

model.save("ppo_hospital_final")

# Final Summary
print("\n=== Final Training Summary ===")
print(f"Overall Average Reward: {np.mean(shift_rewards):.2f}")
print(f"Overall Average Discharge Rate: {np.mean(shift_discharge_rates):.2f}")
print(f"Overall Average Fatigue: {np.mean(shift_avg_fatigues):.2f}")
print(f"Overall Average Queue Size: {np.mean(shift_queue_sizes):.2f}")
print(f"Overall Average Efficiency: {np.mean(shift_efficiencies):.4f}")

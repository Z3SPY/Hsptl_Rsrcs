# train_dqn_agent.py
import os
import gym
import pandas as pd
import glob
import torch
import numpy as np
from tqdm import tqdm
from DQN import DQN_Agent, ENV_ID, LOG_DIR, ITERS
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder

def make_env():
    return gym.make(ENV_ID, render_mode="rgb_array")

def get_next_run_number(directory, algo_name):
    files = glob.glob(os.path.join(directory, f"{algo_name}_run*.csv"))
    if not files:
        return 1
    run_numbers = [int(os.path.basename(f).split('run')[1].split('.')[0]) for f in files]
    return max(run_numbers) + 1

if __name__ == "__main__":
    os.makedirs(os.path.join(LOG_DIR, "videos"), exist_ok=True)
    os.makedirs(os.path.join(LOG_DIR, "runs"), exist_ok=True)

    algo_name = "DQN"
    run_number = get_next_run_number(os.path.join(LOG_DIR, "runs"), algo_name)
    run_path = os.path.join(LOG_DIR, "runs", f"{algo_name}_run{run_number}.csv")

    # === Create raw training environment
    train_env = make_env()
    agent = DQN_Agent(train_env)

    # === Record BEFORE training ===
    print("\n🎥 Recording 'before training' video...")
    record_env = DummyVecEnv([make_env])
    record_env = VecVideoRecorder(
        record_env,
        os.path.join(LOG_DIR, "videos"),
        record_video_trigger=lambda x: x == 0,
        video_length=500,
        name_prefix=f"DQN_before_training_run{run_number}"
    )

    obs = record_env.reset()
    for _ in range(500):
        action = record_env.action_space.sample()
        obs, rewards, dones, infos = record_env.step([action])  # ✅ DummyVecEnv returns (obs, rewards, dones, infos)
        done = dones[0]  # because DummyVecEnv returns list
        if done:
            obs = record_env.reset()
    record_env.close()

    # === Training Loop ===
    all_logs = []

    print(f"\n🚀 Starting {algo_name} training (Run {run_number})...")
    with tqdm(total=ITERS) as pbar:
        for _ in range(ITERS):
            avg_return, policy_loss, value_loss, entropy = agent.train_one_iteration()
            log_entry = {
                'iteration': agent.training_iterations,
                'avg_return': avg_return,
                'policy_loss': policy_loss,
                'value_loss': value_loss,
                'entropy': entropy
            }
            all_logs.append(log_entry)
            pbar.update(1)

    df = pd.DataFrame(all_logs)
    df.to_csv(run_path, index=False)
    print(f"\n✅ DQN training complete! Run saved to {run_path}")

    # === Save model
    agent.save_model(os.path.join(LOG_DIR, f"dqn_frozenlake_model_run{run_number}.pth"))

    # === Record AFTER training ===
    print("\n🎥 Recording 'after training' video...")
    record_env = DummyVecEnv([make_env])
    record_env = VecVideoRecorder(
        record_env,
        os.path.join(LOG_DIR, "videos"),
        record_video_trigger=lambda x: x == 0,
        video_length=500,
        name_prefix=f"DQN_after_training_run{run_number}"
    )

    obs = record_env.reset()
    for _ in range(500):
        # Use the trained policy
        state = np.array([obs])
        state_tensor = torch.FloatTensor(agent.one_hot(state)).to(agent.device)
        q_values = agent.policy_net(state_tensor)
        action = q_values.argmax().item()

        obs, rewards, dones, infos = record_env.step([action])
        done = dones[0]
        if done:
            obs = record_env.reset()
    record_env.close()

    print("\n✅ DQN Training, Logging, and Video Recording Complete!")

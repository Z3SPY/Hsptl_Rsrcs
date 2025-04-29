# train_agent_PPO.py (FINAL CORRECTED VERSION)
import os
import gym
import pandas as pd
import glob
import torch
import numpy as np
from tqdm import tqdm
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder
from gym.vector import SyncVectorEnv
from PPO import PPO_Agent, ENV_ID, LOG_DIR, ITERS

def make_single_env():
    def _init():
        return gym.make(ENV_ID, render_mode="rgb_array")
    return _init

def get_next_run_number(directory, algo_name):
    files = glob.glob(os.path.join(directory, f"{algo_name}_run*.csv"))
    if not files:
        return 1
    run_numbers = [int(os.path.basename(f).split('run')[1].split('.')[0]) for f in files]
    return max(run_numbers) + 1

if __name__ == "__main__":
    os.makedirs(os.path.join(LOG_DIR, "videos"), exist_ok=True)
    os.makedirs(os.path.join(LOG_DIR, "runs"), exist_ok=True)

    algo_name = "PPO"
    run_number = get_next_run_number(os.path.join(LOG_DIR, "runs"), algo_name)
    run_path = os.path.join(LOG_DIR, "runs", f"{algo_name}_run{run_number}.csv")

    n_envs = 4
    batch_steps = 2048

    # Create parallel training environment
    train_env = SyncVectorEnv([make_single_env() for _ in range(n_envs)])
    agent = PPO_Agent(train_env)

    # === Recording before training ===
    print("\n🎥 Recording 'before training' video...")
    record_env = DummyVecEnv([make_single_env()])
    record_env = VecVideoRecorder(
        record_env,
        os.path.join(LOG_DIR, "videos"),
        record_video_trigger=lambda x: x == 0,
        video_length=500,
        name_prefix=f"PPO_before_training_run{run_number}"
    )

    obs = record_env.reset()
    for _ in range(500):
        obs_tensor = torch.LongTensor(obs)
        obs_onehot = torch.zeros((obs_tensor.shape[0], agent.env.single_observation_space.n))
        obs_onehot.scatter_(1, obs_tensor.view(-1,1), 1)

        logits, _ = agent.model(obs_onehot.to(agent.device))  # ✅ FIX
        dist = agent.get_dist(logits)                        # ✅ FIX
        action = dist.sample()

        obs, rewards, dones, infos = record_env.step(action.cpu().numpy())
        done = dones[0]
        if done:
            obs, _ = record_env.reset()
    record_env.close()

    # === Training Loop ===
    all_logs = []

    print(f"\n🚀 Starting PPO training (Run {run_number})...")
    with tqdm(total=ITERS) as pbar:
        for _ in range(ITERS):
            avg_return, policy_loss, value_loss, entropy = agent.train_one_iteration(batch_steps=batch_steps)
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
    print(f"\n✅ PPO training complete! Run saved to {run_path}")

    agent.save_model(os.path.join(LOG_DIR, f"ppo_frozenlake_model_run{run_number}.pth"))

    # === Recording after training ===
    print("\n🎥 Recording 'after training' video...")
    record_env = DummyVecEnv([make_single_env()])
    record_env = VecVideoRecorder(
        record_env,
        os.path.join(LOG_DIR, "videos"),
        record_video_trigger=lambda x: x == 0,
        video_length=500,
        name_prefix=f"PPO_after_training_run{run_number}"
    )

    obs = record_env.reset()
    for _ in range(500):
        obs_tensor = torch.LongTensor(obs)
        obs_onehot = torch.zeros((obs_tensor.shape[0], agent.env.single_observation_space.n))
        obs_onehot.scatter_(1, obs_tensor.view(-1,1), 1)

        logits, _ = agent.model(obs_onehot.to(agent.device))  # ✅ FIX
        dist = agent.get_dist(logits)                        # ✅ FIX
        action = dist.sample()

        obs, rewards, dones, infos = record_env.step(action.cpu().numpy())
        done = dones[0]
        if done:
            obs = record_env.reset()
    record_env.close()

    print("\n✅ PPO Training, Logging, and Video Recording Complete!")

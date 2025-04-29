# record_video.py
import os
import torch
import numpy as np
import gym
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder
from env_wrapper import CarRacingWrapper

def record_video(agent, prefix, video_folder):
    os.makedirs(video_folder, exist_ok=True)

    # Create environment
    env = gym.make(agent.env_id, render_mode="rgb_array")
    env = CarRacingWrapper(env)
    env = DummyVecEnv([lambda: env])
    env = VecVideoRecorder(env, video_folder,
                           record_video_trigger=lambda x: x == 0,
                           video_length=1000,
                           name_prefix=prefix)

    obs = env.reset()
    for _ in range(1000):
        obs_tensor = torch.FloatTensor(obs).to(agent.device) / 255.0

        with torch.no_grad():
            mean_action, _ = agent.model(obs_tensor)
            action = mean_action.cpu().numpy()[0]  # Remove batch dim

        # DummyVecEnv expects actions inside a list
        obs, _, done, _ = env.step([action])
        if done:
            obs = env.reset()

    env.close()

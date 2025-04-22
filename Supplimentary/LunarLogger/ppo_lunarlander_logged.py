# run_ppo_custom.py
import os
import gym
from gym.vector import SyncVectorEnv
from gym.wrappers import RecordVideo

from PPO import PPO_Agent, ENV_ID, LOG_DIR


def make_env():
    return gym.make(ENV_ID)

if __name__ == "__main__":
    # 1) Training env
    train_env = SyncVectorEnv([make_env] * 4)

    # 2) Recording env for final video
    video_folder = os.path.join(LOG_DIR, "videos")
    rec_env = RecordVideo(
        gym.make(ENV_ID, render_mode="rgb_array"),
        video_folder=video_folder,
        name_prefix="ppo_custom"
    )

    # 3) Instantiate & train
    agent = PPO_Agent(train_env, rec_env)
    agent.train()

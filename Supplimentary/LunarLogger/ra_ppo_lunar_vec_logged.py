import gym
from gym.vector import SyncVectorEnv
from gym.wrappers import RecordVideo

from RAPPO import RA_PPO_Agent

def make_env():
    return gym.make("LunarLander-v2")

if __name__ == "__main__":
    # vectorized training env (4 parallel envs)
    train_env = SyncVectorEnv([make_env] * 4)

    # single env for video recordings
    record_env = RecordVideo(
        gym.make("LunarLander-v2", render_mode="rgb_array"),
        video_folder="logs/ra_ppo",
        name_prefix="ra_ppo_logged"
    )

    # instantiate and train
    agent = RA_PPO_Agent(train_env, record_env)
    agent.train(iters=125)

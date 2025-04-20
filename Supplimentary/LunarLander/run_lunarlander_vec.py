
import gym
from gym.vector import SyncVectorEnv
from gym.wrappers import RecordVideo
import matplotlib.pyplot as plt
from ra_ppo_lunar_vec import RA_PPO_Agent

def make_env():
    return gym.make("LunarLander-v2")

train_env = SyncVectorEnv([make_env]*4)
record_env = RecordVideo(gym.make("LunarLander-v2"), "videos", name_prefix="ra_ppo_vec")

agent = RA_PPO_Agent(train_env, record_env)
agent.train(iters=125)

plt.plot(agent.returns_hist, label="Returns")
plt.plot(agent.cvar_hist, label="CVaR")
plt.legend()
plt.savefig("training_plot_vec.png")

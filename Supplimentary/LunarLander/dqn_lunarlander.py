
import gym
from stable_baselines3 import DQN

env = gym.make("LunarLander-v2")
model = DQN("MlpPolicy", env, learning_rate=1e-3, buffer_size=50000, verbose=1)
model.learn(total_timesteps=200_000)
model.save("dqn_lunarlander")
obs, _ = env.reset()
done = False
total_reward = 0
while not done:
    action, _ = model.predict(obs)
    obs, reward, done, _, _ = env.step(action)
    total_reward += reward
print(f"DQN Evaluation Reward: {total_reward:.2f}")
env.close()

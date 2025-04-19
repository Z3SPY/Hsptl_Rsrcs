import gym, copy
import numpy as np

class FrozenLakeWrapper(gym.Env):
    def __init__(self, horizon=3, use_mpc=False, hybrid=False, sims=20):
        super().__init__()
        self.env = gym.make('FrozenLake-v1', is_slippery=True, map_name='8x8')
        self.horizon = horizon
        self.use_mpc = use_mpc
        self.hybrid = hybrid
        self.sims = sims
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space

    def reset(self, **kwargs):
        result = self.env.reset(**kwargs)
        obs = result[0] if isinstance(result, tuple) else result
        self.state = obs
        return obs

    def step(self, action):
        if self.use_mpc and not self.hybrid:
            action = self.get_mpc_action(self.state)

        result = self.env.step(action)
        if len(result) == 5:
            obs, reward, terminated, truncated, info = result
            done = terminated or truncated
        else:
            obs, reward, done, info = result
        self.state = obs
        return obs, reward, done, info

    def render(self):
        return self.env.render()

    def get_mpc_action(self, state):
        best_action = None
        best_avg = -float('inf')

        for a in range(self.action_space.n):
            returns = []
            for _ in range(self.sims):
                env_copy = copy.deepcopy(self.env)
                env_copy.reset()
                env_copy.unwrapped.s = state  # set state manually

                obs2, reward2, done2, _ = env_copy.step(a)[:4]
                total = reward2
                if done2:
                    returns.append(total)
                    continue

                for _ in range(self.horizon - 1):
                    a2 = env_copy.action_space.sample()
                    obs2, reward2, done2, _ = env_copy.step(a2)[:4]
                    total += reward2
                    if done2:
                        break
                returns.append(total)

            avg = np.mean(returns)
            if avg > best_avg:
                best_avg = avg
                best_action = a

        return best_action if best_action is not None else self.action_space.sample()

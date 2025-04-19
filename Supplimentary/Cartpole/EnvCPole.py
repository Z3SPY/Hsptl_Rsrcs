import gym, copy
from queue import PriorityQueue
import numpy as np

class CartPoleEnvWrapper(gym.Env):
    def __init__(self, horizon=3, use_mpc=False, hybrid=False):
        super().__init__()
        self.env = gym.make('CartPole-v1')
        self.horizon = horizon
        self.use_mpc = use_mpc
        self.hybrid = hybrid
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

        obs, reward, terminated, truncated, info = self.env.step(action)
        self.state = obs
        done = terminated or truncated

        # Reward shaping
        x, x_dot, theta, theta_dot = obs
        shaped_reward = reward
        return obs, shaped_reward, terminated, truncated, info

    def render(self, mode='human'):
        return self.env.render(mode)

    def f(self, state, action):
        env_copy = copy.deepcopy(self.env)
        env_copy.state = np.array(state, dtype=np.float32)
        obs, _, _, _, _ = env_copy.step(action)
        return obs

    def J(self, state, action):
        x, x_dot, theta, theta_dot = state
        return x**2 + 10 * theta**2 + 0.1 * action**2

    def get_mpc_action(self, state):
        return mpc_tree(self.f, self.J, self.horizon, range(self.action_space.n), state)

def mpc_tree(f, J, H, U, x0):
    class Node:
        def __init__(self, state, cost, actions):
            self.state = state
            self.cost = cost
            self.actions = actions
        def __lt__(self, other):
            return self.cost < other.cost

    pq = PriorityQueue()
    pq.put(Node(x0, 0, []))
    best_seq = []
    min_cost = float('inf')

    while not pq.empty():
        node = pq.get()
        if len(node.actions) == H:
            if node.cost < min_cost:
                min_cost = node.cost
                best_seq = node.actions
            continue

        for u in U:
            x_next = f(node.state, u)
            cost = J(node.state, u)
            total_cost = node.cost + cost
            pq.put(Node(x_next, total_cost, node.actions + [u]))

    return best_seq[0] if best_seq else np.random.choice(list(U))

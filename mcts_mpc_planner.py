# mcts_mpc_planner.py

import numpy as np
import torch

class MCTSMPCPlanner:
    def __init__(self, env_maker, value_model, depth=3, branching=3, discount=0.99):
        self.env_maker = env_maker
        self.value_model = value_model  # PPO's actor-critic policy
        self.depth = depth
        self.branching = branching
        self.discount = discount

    def best_action(self, env):
        # We'll search root actions
        root_snap = env.snapshot_state()
        best_act = None
        best_val = -float("inf")

        possible_actions = self._sample_root_actions(env)
        for action in possible_actions:
            roll_vals = []
            for _ in range(self.branching):
                env.restore_state(root_snap)
                val = self.simulate_branch(env, action, self.depth)
                roll_vals.append(val)
            avg_val = np.mean(roll_vals)
            if avg_val > best_val:
                best_val = avg_val
                best_act = action

        return best_act, best_val

    def simulate_branch(self, env, action, depth):
        env.planning_mode = True
        try:
            obs, reward, done, _ = env.step(action)
        finally:
            env.planning_mode = False
        if done or depth <= 1:
            leaf_val = 0.0
            if not done:
                leaf_val = self.compute_leaf_value(obs)
            return reward + self.discount * leaf_val

        # else pick a random next action
        snapshot = env.snapshot_state()
        next_action = env.action_space.sample()
        env.restore_state(snapshot)

        return reward + self.discount * self.simulate_branch(env, next_action, depth - 1)

    def compute_leaf_value(self, obs):
        with torch.no_grad():
            obs_t = torch.FloatTensor(obs).unsqueeze(0)
            # obs_t = obs_t.to(self.value_model.device)  # if needed
            # PPO: uses .predict_values()
            val = self.value_model.predict_values(obs_t)
            return val.item()

    def _sample_root_actions(self, env):
        # For MultiDiscrete, let's just sample 10 at root, or do a smarter approach
        # Possibly you can do a full enumeration if the size is small
        acts = [env.action_space.sample() for _ in range(10)]
        return acts

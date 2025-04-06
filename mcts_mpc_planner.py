# mcts_mpc_planner.py

import numpy as np
import torch

class MCTSMPCPlanner:
    def __init__(self, env_maker, value_model, depth=3, branching=3, discount=0.99, 
                 use_value_bootstrap=True, exploration_weight=1.0):
        self.env_maker = env_maker
        self.value_model = value_model  # PPO's actor-critic policy
        self.depth = depth
        self.branching = branching
        self.discount = discount
        self.use_value_bootstrap = use_value_bootstrap
        self.exploration_weight = exploration_weight
    def best_action(self, env):
        root_snap = env.snapshot_state()
        best_act = None
        best_val = -float("inf")
        visit_counts = {}
        possible_actions = self._sample_root_actions(env)
        for action in possible_actions:
            roll_vals = []
            visit_counts[action] = 0
            
            for _ in range(self.branching):
                env.restore_state(root_snap)
                val = self.simulate_branch(env, action, self.depth)
                roll_vals.append(val)
                visit_counts[action] += 1
            
            # UCB1 exploration bonus
            avg_val = np.mean(roll_vals)
            exploration_bonus = self.exploration_weight * np.sqrt(
                np.log(sum(visit_counts.values())) / visit_counts[action]
            )
            
            total_value = avg_val + exploration_bonus
            if total_value > best_val:
                best_val = total_value
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
            if not done and self.use_value_bootstrap:
                leaf_val = self.compute_leaf_value(obs)
            return reward + self.discount * leaf_val
        snapshot = env.snapshot_state()
        next_action = self._select_simulation_action(env, obs)
        env.restore_state(snapshot)
        return reward + self.discount * self.simulate_branch(env, next_action, depth - 1)
    def _select_simulation_action(self, env, obs):
        """Strategy for selecting actions during simulation"""
        if np.random.random() < 0.5:  # Mix of random and value-guided
            return env.action_space.sample()
        else:
            with torch.no_grad():
                obs_t = torch.FloatTensor(obs).unsqueeze(0)
                action_probs = self.value_model.get_distribution(obs_t).probs
                return torch.multinomial(action_probs, 1).item()

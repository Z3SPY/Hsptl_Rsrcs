# NEW SCRIPT: planner_hybrid_mcts_mpc.py
# -------------------------------------------------------------
# This file implements the hybrid MCTS + MPC planning logic.
# MCTS handles the long-horizon stochastic search tree,
# while MPC serves as the short-horizon deterministic rollout policy.

import numpy as np
import random
import copy


class HybridPlanner:
    def __init__(self, env_builder, action_sampler, max_depth=5, num_simulations=30, mpc_horizon=3):
        """
        env_builder: function to build a new env from snapshot
        action_sampler: function to produce a set of plausible actions
        max_depth: max depth of MCTS tree
        num_simulations: number of MCTS simulations
        mpc_horizon: number of steps for the inner MPC rollout
        """
        self.env_builder = env_builder
        self.action_sampler = action_sampler
        self.max_depth = max_depth
        self.num_simulations = num_simulations
        self.mpc_horizon = mpc_horizon

    def best_action(self, state_snapshot):
        """
        Performs MCTS using MPC rollout at the leaves.
        """
        root = Node(state_snapshot)

        for _ in range(self.num_simulations):
            node = root
            depth = 0

            # 1. Selection
            while node.is_fully_expanded() and not node.is_terminal():
                node = node.select_child()
                depth += 1

            # 2. Expansion
            if not node.is_terminal():
                action = self.action_sampler()
                env = self.env_builder(node.state_snapshot)
                next_state, reward, done, _ = env.step(action)
                next_snapshot = env.snapshot_state()
                child = Node(next_snapshot, parent=node, action=action)
                node.children.append(child)
                node = child

            # 3. Simulation using MPC
            reward = self.mpc_rollout(node.state_snapshot)

            # 4. Backpropagation
            while node:
                node.visits += 1
                node.value += reward
                node = node.parent

        best = root.best_child()
        return best.action, best.value / best.visits

    def mpc_rollout(self, snapshot):
        env = self.env_builder(snapshot)
        total_reward = 0.0

        for _ in range(self.mpc_horizon):
            best_action = self.greedy_action(env)
            _, reward, done, _ = env.step(best_action)
            total_reward += reward
            if done:
                break

        return total_reward

    def greedy_action(self, env):
        """
        Simplified greedy policy: pick the action that reduces LOS or discharge delay.
        Can be made smarter.
        """
        best_action = None
        best_value = float('-inf')

        for _ in range(10):  # Try 10 random actions
            action = env.action_space.sample()
            env_copy = copy.deepcopy(env)
            _, reward, _, _ = env_copy.step(action)
            if reward > best_value:
                best_value = reward
                best_action = action

        return best_action


class Node:
    def __init__(self, state_snapshot, parent=None, action=None):
        self.state_snapshot = state_snapshot
        self.parent = parent
        self.action = action
        self.children = []
        self.visits = 0
        self.value = 0.0

    def is_fully_expanded(self):
        return len(self.children) > 0

    def is_terminal(self):
        # We assume horizon depth terminal condition
        return False

    def select_child(self):
        return max(self.children, key=lambda c: c.value / (c.visits + 1e-5))

    def best_child(self):
        return max(self.children, key=lambda c: c.visits)


# HOW TO USE THIS PLANNER (IN YOUR ENV):
# --------------------------------------
# from planner_hybrid_mcts_mpc import HybridPlanner
# planner = HybridPlanner(env_builder=lambda s: HospitalEnv.from_snapshot(s),
#                         action_sampler=lambda: action_space.sample())
# planner.best_action(snapshot)

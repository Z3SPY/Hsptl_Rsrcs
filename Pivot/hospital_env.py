# hospital_env.py

import gym
from gym import spaces
import numpy as np
import random
from model_classes import Scenario, TreatmentCentreModel, CustomResource
import simpy

class HospitalSimEnv(gym.Env):
    """
    Hospital Emergency Department Simulation Environment
    for Deep Reinforcement Learning Resource Allocation
    """

    def __init__(self, sim_config, step_size=60, alpha=1.0, beta=0.1, gamma=0.01, rc_period=20160):
    
        super(HospitalSimEnv, self).__init__()

        self.resource_targets = [
            'triage', 'reg', 'exam', 'trauma', 'cubicles_1', 'cubicles_2'
        ]

        # Actions:
        # 0 = No-op
        # 1 = Add triage
        # 2 = Remove triage
        # 3 = Add reg
        # 4 = Remove reg
        # ...
        self.action_mapping = {}
        i = 1
        for res in self.resource_targets:
            self.action_mapping[i] = ('add', res)
            i += 1
            self.action_mapping[i] = ('remove', res)
            i += 1

        # Gym Action Space
        self.action_space = gym.spaces.Discrete(len(self.action_mapping) + 1)  # +1 for no-op


        self.rc_period = rc_period  # default to 2 weeks
        # === Environment Configurations ===
        self.sim_config = sim_config
        self.step_size = step_size  # how many minutes per step
        self.alpha = alpha  # reward per patient served
        self.beta = beta    # penalty per waiting time
        self.gamma = gamma  # penalty per resource use

        # === Action Space ===
        self.action_space = spaces.Discrete(13)  # 0: no-op, 1-12: +/- resource
        self.wait_penalty_coeff = 0.1  # tuneable
        self.resource_penalty_coeff = 0.05  # tuneable


        # === Observation Space ===
        obs_high = np.array([20160] + [500]*12, dtype=np.float32)  # 2 weeks max time
        self.observation_space = spaces.Box(low=0, high=obs_high, dtype=np.float32)

        # === Internal Variables ===
        self.current_time = 0
        self.previous_patients_served = 0
        self.failed_resource_removals = 0

        self.reset()

    def reset(self):
        self.scenario = Scenario(**self.sim_config)
        self.model = TreatmentCentreModel(self.scenario)

        self.current_time = 0
        self.previous_patients_served = 0
        self.failed_resource_removals = 0

        self.model.env.run(until=0.1)  # initialize any background processes

        return self._get_observation()

    def step(self, action):
        self._apply_action(action)

        next_time = self.current_time + self.step_size
        self.model.env.run(until=next_time)
        self.current_time = next_time

        obs = self._get_observation()
        reward = self._compute_reward()
        done = self.current_time >= self.rc_period
        info = {}

        return obs, reward, done, info

    def _apply_action(self, action):
        try:
            if action == 0:
                print(f"[ACTION] No-op (no change).")
            elif action in self.action_mapping.keys():
                act_type, target_resource = self.action_mapping[action]
                print(f"[ACTION] Attempting to {act_type} resource: {target_resource}")
                if act_type == 'add':
                    self._add_resource(target_resource)
                elif act_type == 'remove':
                    self._remove_resource(target_resource)
            else:
                print(f"[WARN] Invalid action {action} taken.")
        except Exception as e:
            print(f"[ERROR] Exception during action {action}: {str(e)}")



    def _add_resource(self, target):
        if hasattr(self.model.args, target):
            res_obj = getattr(self.model.args, target)
            try:
                if isinstance(res_obj, simpy.Store):
                    # Add a dummy item back into Store
                    res_obj.items.append(CustomResource(target))  # Assuming you have CustomResource class
                    setattr(self.scenario, f'n_{target}', getattr(self.scenario, f'n_{target}') + 1)
                    print(f"[SUCCESS] Added 1 item to Store '{target}'. New count: {getattr(self.scenario, f'n_{target}')}")
                elif isinstance(res_obj, simpy.Resource):
                    res_obj.capacity += 1
                    setattr(self.scenario, f'n_{target}', getattr(self.scenario, f'n_{target}') + 1)
                    print(f"[SUCCESS] Increased Resource '{target}' capacity. New count: {getattr(self.scenario, f'n_{target}')}")
                else:
                    print(f"[FAIL] Unknown resource type for {target}.")
            except Exception as e:
                print(f"[ERROR] Failed to add resource '{target}': {str(e)}")
        else:
            print(f"[ERROR] Target resource '{target}' not found.")


    def _remove_resource(self, target):
        """
        Tries to remove 1 resource unit from the specified resource pool.
        Logs success/failure.
        """
        if hasattr(self.model.args, target):
            res_obj = getattr(self.model.args, target)
            try:
                if isinstance(res_obj, simpy.Store):
                    if len(res_obj.items) > 0:
                        item = res_obj.items.pop(0)
                        new_count = getattr(self.scenario, f'n_{target}') - 1
                        setattr(self.scenario, f'n_{target}', new_count)
                        print(f"[SUCCESS] Removed 1 item from Store '{target}'. New count: {new_count}")
                    else:
                        print(f"[FAIL] No idle items to remove from Store '{target}'.")
                        self.failed_resource_removals += 1
                elif isinstance(res_obj, simpy.Resource):
                    if res_obj.count == 0:
                        res_obj.capacity -= 1
                        new_count = getattr(self.scenario, f'n_{target}') - 1
                        setattr(self.scenario, f'n_{target}', new_count)
                        print(f"[SUCCESS] Decreased Resource '{target}' capacity. New count: {new_count}")
                    else:
                        print(f"[FAIL] Resource '{target}' in use. Cannot remove.")
                        self.failed_resource_removals += 1
                else:
                    print(f"[FAIL] Unknown resource type for {target}.")
                    self.failed_resource_removals += 1
            except Exception as e:
                print(f"[ERROR] Failed to remove resource '{target}': {str(e)}")
                self.failed_resource_removals += 1
        else:
            print(f"[ERROR] Target resource '{target}' not found.")
            self.failed_resource_removals += 1


    def _get_observation(self):
        """
        Generate current observation vector.
        """
        obs = [self.current_time]

        queues = ['triage', 'reg', 'exam', 'trauma', 'cubicles_1', 'cubicles_2']
        for res in queues:
            try:
                obj = getattr(self.model.args, res)
                queue_length = len(obj.get_queue()) if hasattr(obj, 'get_queue') else 0
                busy_count = obj.count if hasattr(obj, 'count') else 0
                obs.append(queue_length)
                obs.append(busy_count)
            except Exception:
                obs.append(0)
                obs.append(0)

        return np.array(obs, dtype=np.float32)

    def _compute_reward(self):
        """
        Reward is composed of:
        - Negative of average patient waiting time
        - Negative of excess unused resources
        """
        # Mean wait time in current system
        avg_wait_time = self.model.get_average_wait_time()  # Make sure this method exists in your model
        
        # Number of idle resources
        idle_resources = 0
        for res_name in self.resource_targets:
            res_obj = getattr(self.model.args, res_name)
            if isinstance(res_obj, simpy.Resource):
                idle = res_obj.capacity - res_obj.count
                idle_resources += max(idle, 0)
            elif isinstance(res_obj, simpy.Store):
                idle_resources += len(res_obj.items)

        # Penalty Components
        wait_penalty = self.wait_penalty_coeff * avg_wait_time
        resource_penalty = self.resource_penalty_coeff * idle_resources

        # Total reward (negative, want to minimize penalties)
        reward = -(wait_penalty + resource_penalty)

        # Debug
        print(f"[REWARD] Wait Penalty: {-wait_penalty:.3f}, Resource Penalty: {-resource_penalty:.3f}, Total Reward: {reward:.3f}")

        return reward


    def seed(self, seed=None):
        random.seed(seed)
        np.random.seed(seed)

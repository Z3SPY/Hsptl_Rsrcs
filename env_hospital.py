# File: env_hospital.py
import simpy
import numpy as np
import gym
import datetime
import time
from gym import spaces
from modelclass import Scenario, WardFlowModel
from planner_mso import HospitalMSOPlanner

class HospitalEnv(gym.Env):
    """
    OpenAI Gym-compatible environment that wraps the SimPy hospital simulation.
    Integrates an MSO planner for periodic long-term planning and uses a DRL agent for short-term control.
    This updated version includes episode-level debugging and logging enhancements.
    """
    def __init__(self, scenario: Scenario, mso_planner: HospitalMSOPlanner = None, mso_frequency_hours=8):
        super().__init__()
        self.scenario = scenario
        self.model = None  # Will hold the WardFlowModel (SimPy simulation instance)
        self.mso_planner = mso_planner
        self.mso_interval = mso_frequency_hours * 60.0  # convert hours to simulation minutes
        self.next_plan_time = 0.0

        # Expanded action space (10 discrete actions)
        self.action_space = spaces.Discrete(10)
        # Observation space: [ED_in_use, ICU_free, MedSurg_free, Nurses_free, Recommended_ICU]
        high = np.array([
            scenario.n_ed_beds,                           # Maximum ED in use
            scenario.n_icu_beds + scenario.n_medsurg_beds,  # Maximum free beds (ICU + MedSurg)
            scenario.n_icu_beds + scenario.n_medsurg_beds,  # Same as above
            scenario.day_shift_nurses,                      # Maximum available nurses
            scenario.n_icu_beds + scenario.n_medsurg_beds   # Recommended ICU beds (0 to total beds)
        ], dtype=np.float32)
        self.observation_space = spaces.Box(low=0.0, high=high, shape=(5,), dtype=np.float32)

        # Episode-level logging variables
        self.action_counts = np.zeros(self.action_space.n)
        self.cumulative_reward = 0.0
        self.mso_time = 0.0
        self.episode_start_time = None

        # Additional state variables for nurse pool and bed capacities
        self.total_nurses = scenario.day_shift_nurses + scenario.night_shift_nurses
        self.current_icu_beds = scenario.n_icu_beds
        self.current_medsurg_beds = scenario.n_medsurg_beds

        # Initialize nurse counts for each shift
        self.current_day_nurses = scenario.day_shift_nurses
        self.current_night_nurses = scenario.night_shift_nurses

        # Set current shift (default to 'day')
        self.current_shift = 'day'


    def reset(self):
        # Initialize a new simulation episode
        self.model = WardFlowModel(self.scenario, start_datetime=datetime.datetime.now())
        # Start background processes in the SimPy model
        self.model.env.process(self.model.nurse_shift_scheduler())
        self.model.env.process(self.model.feedback_controller())
        self.model.env.process(self.model.audit_utilisation(interval=1))
        self.model.env.process(self.model.arrivals_generator())
        self.next_plan_time = 0.0

        # Initial MSO plan at time 0
        if self.mso_planner:
            self.recommended_plan = self.mso_planner.plan_allocation()
        else:
            self.recommended_plan = {
                "icu_beds": self.scenario.n_icu_beds,
                "medsurg_beds": self.scenario.n_medsurg_beds
            }

        # Initialize episode-level logging variables
        self.action_counts = np.zeros(self.action_space.n)
        self.cumulative_reward = 0.0
        self.mso_time = 0.0
        self.episode_start_time = time.time()

        # Reset nurse and bed capacities as per the scenario
        self.current_icu_beds = self.scenario.n_icu_beds
        self.current_medsurg_beds = self.scenario.n_medsurg_beds

        return self._get_obs()

    def step(self, action):
        # Count the action for logging purposes
        self.action_counts[action] += 1

        # Apply the RL agent's action (short-term decision)
        # Actions 0-9 correspond to:
        # 0: No-op
        # 1: Shift one MedSurg bed to ICU
        # 2: Shift one ICU bed to MedSurg
        # 3: Shift two MedSurg beds to ICU
        # 4: Shift two ICU beds to MedSurg
        # 5: Add one nurse to current shift
        # 6: Remove one nurse from current shift
        # 7: Add two nurses to current shift
        # 8: Remove two nurses from current shift
        # 9: Rebalance beds according to MSO plan
        if action == 0:
            pass  # Do nothing
        elif action == 1:
            if len(self.model.medsurg.items) > 0:
                self.model.env.process(self._reallocate_bed(self.model.medsurg, self.model.icu))
                self.current_icu_beds += 1
                self.current_medsurg_beds -= 1
        elif action == 2:
            if len(self.model.icu.items) > 0:
                self.model.env.process(self._reallocate_bed(self.model.icu, self.model.medsurg))
                self.current_icu_beds -= 1
                self.current_medsurg_beds += 1
        elif action == 3:
            beds_to_move = min(2, len(self.model.medsurg.items))
            for _ in range(beds_to_move):
                self.model.env.process(self._reallocate_bed(self.model.medsurg, self.model.icu))
            self.current_icu_beds += beds_to_move
            self.current_medsurg_beds -= beds_to_move
        elif action == 4:
            beds_to_move = min(2, len(self.model.icu.items))
            for _ in range(beds_to_move):
                self.model.env.process(self._reallocate_bed(self.model.icu, self.model.medsurg))
            self.current_icu_beds -= beds_to_move
            self.current_medsurg_beds += beds_to_move
        elif action == 5:
            # Add one nurse to current shift
            if self.current_shift == 'day' and self.current_day_nurses < self.total_nurses:
                self.current_day_nurses += 1
            elif self.current_shift == 'night' and self.current_night_nurses < self.total_nurses:
                self.current_night_nurses += 1
            # Instead of modifying capacity directly, compute new capacity and recreate the resource
            new_capacity = self.model.nurses.capacity + 1
            self.model.nurses = simpy.PreemptiveResource(self.model.env, capacity=new_capacity)

        elif action == 6:
            # Remove one nurse from current shift (ensure at least one remains)
            if self.current_shift == 'day' and self.current_day_nurses > 1:
                self.current_day_nurses -= 1
            elif self.current_shift == 'night' and self.current_night_nurses > 1:
                self.current_night_nurses -= 1
            # Use max() to ensure new capacity is at least 1
            new_capacity = max(1, self.model.nurses.capacity - 1)
            self.model.nurses = simpy.PreemptiveResource(self.model.env, capacity=new_capacity)
        

        elif action == 7:
            # Add two nurses to current shift
            for _ in range(2):
                if self.current_shift == 'day' and self.current_day_nurses < self.total_nurses:
                    self.current_day_nurses += 1
                    new_capacity = self.model.nurses.capacity + 1
                    self.model.nurses = simpy.PreemptiveResource(self.model.env, capacity=new_capacity)
                elif self.current_shift == 'night' and self.current_night_nurses < self.total_nurses:
                    self.current_night_nurses += 1
                    new_capacity = self.model.nurses.capacity + 1
                    self.model.nurses = simpy.PreemptiveResource(self.model.env, capacity=new_capacity)

        elif action == 8:
            # Remove two nurses from current shift
            for _ in range(2):
                if self.current_shift == 'day' and self.current_day_nurses > 1:
                    self.current_day_nurses -= 1
                    new_capacity = max(1, self.model.nurses.capacity - 1)
                    self.model.nurses = simpy.PreemptiveResource(self.model.env, capacity=new_capacity)
                elif self.current_shift == 'night' and self.current_night_nurses > 1:
                    self.current_night_nurses -= 1
                    new_capacity = max(1, self.model.nurses.capacity - 1)
                    self.model.nurses = simpy.PreemptiveResource(self.model.env, capacity=new_capacity)
        
        elif action == 9:
            # Rebalance beds per the MSO planner's recommendation
            icu_target = self.recommended_plan.get("icu_beds", self.scenario.n_icu_beds)
            delta = icu_target - self.current_icu_beds
            if delta > 0:
                beds_to_move = min(delta, len(self.model.medsurg.items))
                for _ in range(beds_to_move):
                    self.model.env.process(self._reallocate_bed(self.model.medsurg, self.model.icu))
                self.current_icu_beds += beds_to_move
                self.current_medsurg_beds -= beds_to_move
            elif delta < 0:
                beds_to_move = min(-delta, len(self.model.icu.items))
                for _ in range(beds_to_move):
                    self.model.env.process(self._reallocate_bed(self.model.icu, self.model.medsurg))
                self.current_icu_beds -= beds_to_move
                self.current_medsurg_beds += beds_to_move

        # Update MSO plan if the next planning time has been reached
        current_time = self.model.env.now
        if self.mso_planner and current_time >= self.next_plan_time:
            t0 = time.time()
            self.recommended_plan = self.mso_planner.plan_allocation(current_state=self._snapshot_state())
            self.mso_time += time.time() - t0
            self.next_plan_time += self.mso_interval

        # Advance the simulation by one decision interval (e.g., 60 minutes)
        step_duration = 60.0  # 1 hour per RL step
        next_stop = current_time + step_duration
        done = False
        if next_stop >= self.scenario.simulation_time:
            next_stop = self.scenario.simulation_time
            done = True
            #Call final cancelation maybe

        self.model.env.run(until=next_stop + 0.001)


        obs = self._get_obs()
        reward = self._compute_reward()
        self.cumulative_reward += reward

        # At episode end, compile detailed logging information
        info = {}
        if done:
            total_steps = int(np.sum(self.action_counts))
            action_usage = {f"action_{i}": int(n) for i, n in enumerate(self.action_counts)}
            episode_time = time.time() - self.episode_start_time
            # Here we provide placeholder reward component values.
            # In practice, you may wish to track each component separately.
            reward_components = {
                "base": 10.0 * total_steps,
                "congestion_penalty": 0.0,
                "deviation_penalty": 0.0,
                "utilization_bonus": 0.0,
                "overwork_penalty": 0.0
            }
            info['episode_summary'] = {
                "steps": total_steps,
                "action_counts": action_usage,
                "total_reward": self.cumulative_reward,
                "reward_components": reward_components,
                "mso_planning_time": round(self.mso_time, 4),
                "episode_time": round(episode_time, 4)
            }
        return obs, reward, done, info

    def _reallocate_bed(self, from_store, to_store):
        """SimPy process: move one resource unit from from_store to to_store."""
        bed = yield from_store.get()
        yield to_store.put(bed)

    def _get_obs(self):
        # Gather current system metrics for the observation vector
        ed_in_use = self.model.ed.count
        icu_free = len(self.model.icu.items)
        med_free = len(self.model.medsurg.items)
        nurses_free = self.model.nurses.capacity - self.model.nurses.count
        rec_icu = self.recommended_plan["icu_beds"] if self.recommended_plan else self.scenario.n_icu_beds

        obs = np.array([ed_in_use, icu_free, med_free, nurses_free, rec_icu], dtype=np.float32)
        return obs

    def _compute_reward(self):
        """
        Computes a reward that gives a positive signal when the system operates near target levels,
        and subtracts penalties when resources are overutilized or deviate from MSO recommendations.
        """
        # Baseline reward per time step
        R_base = 10.0
        
        # Example congestion penalty calculations:
        ed_util_ratio = self.model.ed.count / self.scenario.n_ed_beds
        icu_free = len(self.model.icu.items)
        icu_util_ratio = (self.current_icu_beds - icu_free) / self.current_icu_beds if self.current_icu_beds > 0 else 0
        medsurg_free = len(self.model.medsurg.items)
        medsurg_util_ratio = (self.current_medsurg_beds - medsurg_free) / self.current_medsurg_beds if self.current_medsurg_beds > 0 else 0

        target_util = 0.6
        congestion_penalty = 0.0
        if ed_util_ratio > target_util:
            congestion_penalty += 5.0 * (ed_util_ratio - target_util)
        if icu_util_ratio > target_util:
            congestion_penalty += 5.0 * (icu_util_ratio - target_util)
        if medsurg_util_ratio > target_util:
            congestion_penalty += 5.0 * (medsurg_util_ratio - target_util)

        # Deviation penalty (if using MSO)
        deviation_penalty = 0.0
        if self.mso_planner:
            rec_icu = self.recommended_plan.get("icu_beds", self.current_icu_beds)
            deviation = abs(self.current_icu_beds - rec_icu)
            deviation_penalty = 3.0 * (deviation / (self.current_icu_beds + self.current_medsurg_beds))**2

        # Nurse utilization bonus (example)
        nurses_available = self.model.nurses.capacity - self.model.nurses.count
        nurse_bonus = max(0, (nurses_available / self.model.nurses.capacity) - 0.3)

        # Placeholder overwork penalty (to be refined with proper nurse shift logic)
        overwork_penalty = 0.0

        total_reward = R_base - (congestion_penalty + deviation_penalty) + nurse_bonus - overwork_penalty
        return total_reward

    def _snapshot_state(self):
        # Create a snapshot of the current state for use by the MSO planner
        state = {
            "icu_occupancy": self.current_icu_beds - len(self.model.icu.items),
            "medsurg_occupancy": self.current_medsurg_beds - len(self.model.medsurg.items),
            "ed_in_use": self.model.ed.count,
            "nurses_available": self.model.nurses.capacity - self.model.nurses.count
        }
        return state

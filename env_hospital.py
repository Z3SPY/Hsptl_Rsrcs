#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Refactored Hospital Environment for DRL & MSO Integration (env_hospital.py)
------------------------------------------------------------------------------
This Gym environment wraps the SimPy-based hospital simulation (from modelclass.py)
and provides:
    - A snapshot-based observation space.
    - A MultiDiscrete action space to control hospital resource allocation policy.
    - Extensive debug logging for training integration and frontend snapshots.
    
Action Space (MultiDiscrete with 2 dimensions):
    - Dimension 0: ICU Priority Policy 
         0: Default ICU admission (only critical patients admitted)
         1: Reserved ICU – become more conservative with moderate admissions
         2: Overbook ICU – allow moderate patients to be admitted more aggressively.
         
    - Dimension 1: Nurse Shift Policy
         0: Balanced Nurse Schedule (baseline)
         1: Increase nurse capacity (nurse-heavy)
         2: Decrease nurse capacity (nurse-light)
         
Observation: A vector from the current snapshot, including:
    - ED available capacity (ED beds free)
    - ICU available tokens (beds not in use)
    - Ward available tokens (beds in Ward/MedSurg)
    - Nurses available (current nurse capacity minus in-use count)
    - Current simulation time (in minutes)
    
The environment advances in steps (each step has a fixed simulation run time in minutes).
"""

import simpy
import numpy as np
import gym
from gym import spaces
import datetime
import time

# Import your simulation model from modelclass
from modelclass import WardFlowModel, Scenario

# For debugging in this file, toggle this flag.
DEBUG_LOGS = True

class HospitalEnv(gym.Env):
    """
    OpenAI Gym Environment wrapping the WardFlowModel simulation.
    """
    metadata = {'render.modes': ['human']}

    def __init__(self,
                 scenario: Scenario,
                 step_minutes: int = 60,   # Time advanced per step (in minutes)
                 max_icu_policy: int = 2,    # 0,1,2 for ICU policy adjustments
                 max_nurse_policy: int = 2): # 0,1,2 for nurse shift policy
        super(HospitalEnv, self).__init__()
        
        # Store scenario and simulation step
        self.scenario = scenario
        self.step_minutes = step_minutes

        # Initialize the underlying simulation model.
        self.model = WardFlowModel(scenario, start_datetime=datetime.datetime.now())
        # Run initialization (simulate 0 minutes) so background processes start.
        self.model.run(0)
        
        # Set up observation space: we'll create a state vector of 5 elements.
        # For clarity, we assume:
        #   obs[0]: ED available = (ED capacity - ed in use)
        #   obs[1]: ICU available tokens (# tokens in ICU store)
        #   obs[2]: Ward available tokens (# tokens in Ward store)
        #   obs[3]: Nurses available = (nurse capacity - nurses in use)
        #   obs[4]: Current simulation time (minutes)
        obs_high = np.array([
            self.scenario.n_ed_beds,                      # max ED available
            self.scenario.n_icu_beds,                     # max ICU tokens
            self.scenario.n_ward_beds,                    # max Ward tokens
            self.scenario.day_shift_nurses + self.scenario.night_shift_nurses,  # max nurses
            self.scenario.simulation_time               # simulation total time in minutes
        ], dtype=np.float32)
        self.observation_space = spaces.Box(low=0.0, high=obs_high, shape=(5,), dtype=np.float32)
        
        # Define MultiDiscrete action space for policy adjustments:
        # Two components:
        #   - ICU policy: 0 (default), 1 (conservative), 2 (aggressive)
        #   - Nurse shift policy: 0 (baseline), 1 (increase), 2 (decrease)
        self.action_space = spaces.MultiDiscrete([3, 3])
        
        # Internal state variables for tracking policy parameters.
        # Set initial nurse capacity policy from scenario.
        self.current_nurse_policy = 0  # baseline
        self.current_icu_policy = 0    # default
        self.current_time = self.model.env.now  # in minutes
        
        # For reward tracking:
        self.last_reward_time = 0.0
        self.cumulative_reward = 0.0
        
    def reset(self):
        """
        Resets the simulation environment and returns the initial observation.
        Also resets the simulation's internal clock and resource allocations.
        """
        if DEBUG_LOGS:
            print("[DEBUG] Resetting Hospital Environment.")
        self.model = WardFlowModel(self.scenario, start_datetime=datetime.datetime.now())
        self.model.run(0)  # Reinitialize simulation processes
        
        # Reset any policy-related variables if needed.
        self.current_nurse_policy = 0
        self.current_icu_policy = 0
        self.last_reward_time = 0.0
        self.cumulative_reward = 0.0
        
        obs = self._get_obs()
        if DEBUG_LOGS:
            print(f"[DEBUG] Reset observation: {obs}")
        return obs

    def _get_obs(self):
        """
        Returns the current observation vector using the model's snapshot.
        Converts the snapshot dictionary into a vector.
        """
        snapshot = self.model.snapshot_state()
        # We assume snapshot contains: ed_in_use, icu_available, ward_available, nurses_available, time, total_patients
        ed_avail = self.scenario.n_ed_beds - snapshot.get('ed_in_use', 0)
        icu_avail = snapshot.get('icu_available', 0)
        ward_avail = snapshot.get('ward_available', 0)
        nurses_avail = snapshot.get('nurses_available', 0)
        current_time = snapshot.get('time', 0)
        obs_vector = np.array([ed_avail, icu_avail, ward_avail, nurses_avail, current_time], dtype=np.float32)
        if DEBUG_LOGS:
            print(f"[DEBUG] Observation vector: {obs_vector}")
        return obs_vector

    def step(self, action):
        """
        Applies the agent's action, advances the simulation by a fixed time step (step_minutes),
        and returns the new observation, reward, done flag, and additional info.
        """
        if DEBUG_LOGS:
            print(f"[DEBUG] Step called with action: {action}")
        # Decode action using MultiDiscrete structure.
        # Action: [icu_policy, nurse_shift_policy]
        icu_policy = int(action[0])
        nurse_policy = int(action[1])
        if DEBUG_LOGS:
            print(f"[DEBUG] Decoded action - ICU Policy: {icu_policy}, Nurse Shift Policy: {nurse_policy}")
        
        # Update internal policy parameters accordingly:
        # For ICU policy:
        #   0: default, 1: be conservative (e.g., only admit critical patients to ICU)
        #   2: aggressive (allow more moderate patients into ICU, even if it means some delay)
        self.current_icu_policy = icu_policy
        # Here, you could adjust a parameter in the scenario or WardFlowModel accordingly.
        # For example, modify a threshold (this is a placeholder for your logic):
        if icu_policy == 1:
            # More conservative: perhaps increase the chance that moderate patients remain in Ward.
            self.scenario.p_icu = 0.2  # lower probability for ICU admission
        elif icu_policy == 2:
            # More aggressive: increase ICU admissions.
            self.scenario.p_icu = 0.4
        else:
            self.scenario.p_icu = 0.3  # baseline
        
        # For Nurse shift policy:
        #   0: baseline, 1: increase nurse capacity, 2: decrease nurse capacity.
        self.current_nurse_policy = nurse_policy
        base_nurse = self.scenario.day_shift_nurses  # assume day shift for simplicity here
        if nurse_policy == 1:
            new_nurse_capacity = base_nurse + 2
        elif nurse_policy == 2:
            new_nurse_capacity = max(1, base_nurse - 2)
        else:
            new_nurse_capacity = base_nurse
        # Update nurse resource capacity
        self.model.nurses.capacity = new_nurse_capacity
        if DEBUG_LOGS:
            print(f"[DEBUG] Nurse capacity updated to: {new_nurse_capacity}")
        
        # Advance simulation by fixed step
        self.model.run(self.step_minutes)
        self.current_time = self.model.env.now
        
        # Compute reward based on events that occurred since last reward time.
        recent_events = [e for e in self.model.event_log if self.last_reward_time < e["time"] <= self.current_time]
        step_reward = self.compute_reward(recent_events)
        self.last_reward_time = self.current_time
        self.cumulative_reward += step_reward
        
        # Get new observation.
        obs = self._get_obs()
        
        # Determine if simulation is done.
        done = self.current_time >= self.scenario.simulation_time
        
        # Build info dictionary (for debugging & logging)
        info = {
            'current_time': self.current_time,
            'step_reward': step_reward,
            'cumulative_reward': self.cumulative_reward,
            'recent_events': recent_events  # optionally, for debugging
        }
        if DEBUG_LOGS:
            print(f"[DEBUG] Step result - Time: {self.current_time}, Reward: {step_reward}, Done: {done}")
        return obs, step_reward, done, info

    def compute_reward(self, events):
        """
        Compute reward based on recent events.
        For example, reward successful discharges and penalize long waits or complications.
        Here, a simple reward function is implemented.
        """
        discharge_bonus = 0.0
        delay_penalty = 0.0
        complication_penalty = 0.0

        for event in events:
            if event.get('event') == 'discharge':
                # Reward discharge; if LOS (total_time) is less than a target, give bonus.
                los = event.get('total_time', 0)
                target_los = 480  # e.g., 8 hours target LOS
                if los <= target_los:
                    discharge_bonus += 1.0
                else:
                    delay_penalty += (los - target_los) * 0.01
            if event.get('event_type') == 'complication':
                complication_penalty += 5.0
        
        # Additional penalty: higher ED congestion results in lower reward.
        ed_queue_penalty = self._get_obs()[0] * 0.1

        total_reward = discharge_bonus - delay_penalty - complication_penalty - ed_queue_penalty
        if DEBUG_LOGS:
            print(f"[DEBUG] Computed Reward: {total_reward} (Discharge bonus: {discharge_bonus}, Delay penalty: {delay_penalty}, Complication penalty: {complication_penalty}, ED queue penalty: {ed_queue_penalty})")
        return total_reward

    def render(self, mode='human'):
        # For basic rendering, print out the current snapshot.
        state = self.model.snapshot_state()
        print(f"Render at time {state['time']} minutes: ED usage: {self.model.ed.count}, ICU available: {len(self.model.icu.items)}, Ward available: {len(self.model.ward.items)}, Nurses available: {self.model.nurses.capacity - self.model.nurses.count}")
        return state

    def close(self):
        # Cleanup if necessary.
        print("[DEBUG] Closing Hospital Environment")


if __name__ == "__main__":
    
    scenario = Scenario(
        simulation_time=4 * 60,  # 4 hours in minutes
        random_number_set=42,
        n_triage=4,
        n_ed_beds=4,
        n_icu_beds=4,
        n_ward_beds=4,          # Ward is the rebranded MedSurg
        triage_mean=10.0,
        ed_eval_mean=60.0,
        ed_imaging_mean=30.0,
        icu_stay_mean=360.0,
        ward_stay_mean=240.0,
        discharge_delay_mean=15.0,
        icu_proc_mean=30.0,
        p_icu=0.3,
        p_ward=0.5,
        p_deteriorate=0.3,
        p_icu_procedure=0.5,
        day_shift_nurses=10,
        night_shift_nurses=5,
        shift_length=12,
        model="resource_allocation"
    )

    # Instantiate the Gym environment.
    env = HospitalEnv(scenario=scenario, step_minutes=60)
    
    # Reset environment
    obs = env.reset()
    print("Initial Observation:", obs)
    
    done = False
    total_reward = 0.0
    step_count = 0

    # Run simulation with random actions for testing.
    while not done:
        action = env.action_space.sample()  # Replace with your agent's action during training.
        print(f"[DEBUG] Step {step_count}, Action: {action}")
        obs, reward, done, info = env.step(action)
        total_reward += reward
        step_count += 1
        env.render()  # Print out current state snapshot.

    print(f"Simulation finished after {step_count} steps. Total Reward: {total_reward}")

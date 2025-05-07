# hospital_env.py

import gym
from gym import spaces
import numpy as np
import simpy
from model_classes import Scenario, TreatmentCentreModel, CustomResource, SimulationSummary
import pandas as pd
import csv
import os

class HospitalSimEnv(gym.Env):
    """
    Hospital ED simulation environment for RL (resource allocation & patient flow).
    Continuous-time SimPy model controlled in discrete time steps.
    """
    metadata = {'render.modes': []}

    def __init__(self, sim_config, step_size=15, alpha=5.0, beta=0.2, gamma=0.1, delta=0.01, epsilon=0.05, rc_period=20160, cost_mode='diverse'):

        super().__init__()
        
        # Simulation setup
        self.base_sim_config = dict(sim_config)
        self.step_size = step_size
        self.rc_period = rc_period
        self.alpha = alpha
        self.beta = beta
        self.epsilon = epsilon
        self.gamma = gamma
        self.delta = delta
        self.cost_mode = cost_mode
        self.cur_reward = 0

        # Resource definitions
        self.resource_targets = ['triage', 'registration', 'exam', 'trauma', 'cubicle_1', 'cubicle_2', 'icu_beds', 'ward_beds']
        self.cost_per_unit = {
            'triage': 1.0,
            'registration': 0.7,
            'exam': 1.5,
            'trauma': 3.0,
            'cubicle_1': 1.2,
            'cubicle_2': 2.5,
        }

        self.resource_capacity = {res: sim_config.get(f'n_{res}', 1) for res in self.resource_targets}

        # Pending resource changes (for delayed add/remove)
        self.pending_resource_changes = []
        self.resource_change_delay = 480  # 8 hours = 480 minutes


        # Wait TImes
        # Initialize wait time tracking
        self.wait_time_totals = {unit: 0.0 for unit in ["triage", "registration", "exam", "trauma", "cubicle_1", "cubicle_2"]}
        self.wait_time_counts = {unit: 0 for unit in ["triage", "registration", "exam", "trauma", "cubicle_1", "cubicle_2"]}


        # Shift management
        self.shift_duration = 480  # 8 hours
        self.time_in_shift = 0
        self.is_shift_planning = True
        self.total_staff_pool = {
            'triage': 10,
            'registration': 10,
            'exam': 10,
            'trauma': 10,
            'cubicle_1': 17,
            'cubicle_2': 17,
            'ward_beds': 30,
            'icu_beds': 15
        }
        self.active_staff = {k: 0 for k in self.total_staff_pool.keys()}
        self.resting_staff = {k: self.total_staff_pool[k] for k in self.total_staff_pool.keys()}
        self.latest_mso_plan = np.zeros(len(self.resource_targets), dtype=np.float32)


        # Action space: MultiDiscrete for shift staffing
        self.action_space = spaces.MultiDiscrete(
            [self.total_staff_pool[k] + 1 for k in self.total_staff_pool.keys()]
        )

        # Legacy action mapping (for delayed add/remove if needed)
        self.action_mapping = {0: ('noop', None)}
        idx = 1
        for res in self.resource_targets:
            self.action_mapping[idx] = ('add', res); idx += 1
            self.action_mapping[idx] = ('remove', res); idx += 1

        # Observation space
        obs_dim = 3 + 2 * len(self.resource_targets) + len(self.resource_targets)
        high = np.array([rc_period] + [500]*(obs_dim-1), dtype=np.float32)
        self.observation_space = spaces.Box(low=0, high=high, dtype=np.float32)

        # Initialize environment
        self.current_time = 0.0
        self.reset()


    # -----------------------------
    # MSO PLANNER
    # -----------------------------


    def reset(self):
        """Randomize day type: Normal, Surge, Light (Minimal MSO)."""
        cfg = dict(self.base_sim_config)

        # Random Day Type
        day_type = np.random.choice(['normal', 'surge', 'light'], p=[0.6, 0.3, 0.1])
        if day_type == 'normal':
            cfg['override_arrival_rate'] = True
            cfg['manual_arrival_rate'] = 60.0
        elif day_type == 'surge':
            cfg['override_arrival_rate'] = True
            cfg['manual_arrival_rate'] = 30.0
        elif day_type == 'light':
            cfg['override_arrival_rate'] = True
            cfg['manual_arrival_rate'] = 90.0

        # Randomize small trauma probability variations
        base_p = cfg.get('prob_trauma', 0.12)
        cfg['prob_trauma'] = float(np.clip(base_p + np.random.normal(0, 0.02), 0.0, 1.0))
        cfg['model'] = "full"

        # Create new scenario
        self.scenario = Scenario(**cfg)
        self.model = TreatmentCentreModel(self.scenario)
        self.model.sim_summary = SimulationSummary(self.model)

        self.model.hospitalenv = self  
        self.model.rc_period = self.rc_period

        
        

        
        self.model.env.process(self.model.arrivals_generator())


        self.current_time = 0.0

        raw_plan = self.mso_shift_planner()  # integer plan from MSO
        pool = np.array([4, 4, 5, 4, 5, 4, 30, 30])      # Max capacity per unit (match your staff pool definition)
        self.latest_mso_plan = raw_plan / pool  # Normalize plan to [0,1]

        print("[DEBUG] latest_mso_plan:", self.latest_mso_plan)
        print("[DEBUG] Plan length:", len(self.latest_mso_plan))

        return self._get_observation()
    

    def _apply_shift_action(self, action):
        """Apply agent's shift staffing plan."""
        keys = list(self.total_staff_pool.keys())
        for idx, k in enumerate(keys):
            requested = action[idx]
            available = self.total_staff_pool[k]
            # Limit requested active staff to available staff
            actual = min(requested, available)
            self.active_staff[k] = actual
            self.resting_staff[k] = available - actual

            # Update SimPy resource stores (rebuild them)
            store = getattr(self.model.args, k)
            if hasattr(store, "items"):
                # Clear all resources
                store.items.clear()
                # Add active resources
                for i in range(actual):
                    store.put(CustomResource(self.model.env, capacity=1, id_attribute=i))

    def forecast_arrivals(self):
        """Simple forecast: Assume average arrivals based on base config."""
        # Assume base trauma probability
        trauma_prob = self.base_sim_config.get('prob_trauma', 0.12)
        
        # Assume arrival rate of 1 patient per hour (very rough for now)
        patients_per_hour = 1.0

        # Forecast for the whole shift (8 hours)
        total_patients = patients_per_hour * (self.shift_duration / 60)

        expected_trauma = total_patients * trauma_prob
        expected_minor = total_patients * (1 - trauma_prob)

        return expected_minor, expected_trauma

    def estimate_service_demand(self, expected_minor, expected_trauma):
        """Estimate minimum service staff required based on expected arrivals."""
        # Simple assumption: 1 minor = 1 triage + 1 registration + 1 exam + 1 cubicle_1
        # Simple assumption: 1 trauma = 1 triage + 1 trauma stabilization + 1 cubicle_2
        
        staff_needed = {
            'triage': int(expected_minor + expected_trauma),
            'registration': int(expected_minor),  # Only minors go through registration
            'exam': int(expected_minor),           # Only minors go through examination
            'trauma': int(expected_trauma),         # Only trauma patients
            'cubicle_1': int(expected_minor),       # Non-trauma cubicle beds
            'cubicle_2': int(expected_trauma),      # Trauma cubicle beds
        }
        
        return staff_needed



    def mso_shift_planner(self):
        """Full MSO pipeline: forecast arrivals -> estimate service -> solve optimal staffing."""
        expected_minor, expected_trauma = self.forecast_arrivals()
        staff_needed = self.estimate_service_demand(expected_minor, expected_trauma)
        allocation = self.solve_shift_optimization(staff_needed)
        return allocation
        

    
    def solve_shift_optimization(self, staff_needed):
        """Fatigue-aware MSO: choose staffing considering service load and fatigue."""
        allocation = []
        for unit in self.resource_targets:
            max_available = self.total_staff_pool[unit]
            recommended = staff_needed.get(unit, 0)
            
            # Fatigue adjustment
            avg_fatigue = self._average_fatigue_of_unit(unit)
            fatigue_penalty_factor = (avg_fatigue / 100.0)  # e.g., 0.0 to 1.0
            adjusted_staff = int(recommended * (1.0 + fatigue_penalty_factor))  # recommend more staff if fatigue is high

            # Final choice
            assigned = min(adjusted_staff, max_available)
            allocation.append(assigned)

        return np.array(allocation, dtype=np.int32)


    def _count_shift_discharges(self):
        """Count discharges during the last shift window."""
        start_time = self.current_time - self.shift_duration
        discharges = sum(1 for e in self.model.full_event_log if e['event'] == 'depart' and start_time < e['time'] <= self.current_time)
        return discharges
    
    # -----------------------------
    # FATIGUE
    # -----------------------------

    def _average_fatigue_of_unit(self, unit_name):
        """Helper: average fatigue of resources in a unit."""
        store = getattr(self.model.args, unit_name)
        fatigue_sum = 0.0
        count = 0
        if hasattr(store, "items"):
            for resource in store.items:
                if hasattr(resource, "fatigue"):
                    fatigue_sum += resource.fatigue
                    count += 1
        return (fatigue_sum / count) if count > 0 else 0.0

    def _average_fatigue_by_unit(self):
        """Returns a dictionary of average fatigue per unit."""
        avg_fatigue = {}
        for unit_name in self.resource_targets:
            fatigue_sum = 0.0
            count = 0
            store = getattr(self.model.args, unit_name)
            if hasattr(store, "items"):
                for resource in store.items:
                    if hasattr(resource, "fatigue"):
                        fatigue_sum += resource.fatigue
                        count += 1
            avg_fatigue[unit_name] = (fatigue_sum / count) if count > 0 else 0.0
        return avg_fatigue

    
    def _update_fatigue(self):
        """
        Improved fatigue model:
        - Staff accumulate fatigue even if idle (small base rate).
        - Staff accumulate extra fatigue if busy (higher rate).
        - Resting staff recover fatigue naturally.
        """
        base_fatigue_rate_per_min = 0.02   # Low fatigue even if idle
        busy_fatigue_rate_per_min = 0.1    # Higher fatigue if busy
        recovery_rate_per_min = 0.2        # Recovery when off shift

        # Update active on-shift staff
        for res_name in self.resource_targets:
            store = getattr(self.model.args, res_name)
            if hasattr(store, "items"):
                for resource in store.items:
                    if hasattr(resource, "fatigue"):
                        # Always gain small base fatigue
                        resource.fatigue = min(100.0, resource.fatigue + base_fatigue_rate_per_min * self.step_size)
                        resource.time_on_shift += self.step_size

                        # If busy, gain extra fatigue
                        if getattr(resource, "busy", False):
                            resource.fatigue = min(100.0, resource.fatigue + busy_fatigue_rate_per_min * self.step_size)

        # Simulate fatigue recovery for resting staff
        for res_name, count in self.resting_staff.items():
            if count > 0:
                pass




    

    # -----------------------------
    # PROCESSES
    # -----------------------------
        
    
    def _process_pending_actions(self):
        """Execute any resource changes that are now due."""
        to_execute = [item for item in self.pending_resource_changes if item[0] <= self.current_time]
        self.pending_resource_changes = [item for item in self.pending_resource_changes if item[0] > self.current_time]

        for activation_time, kind, res in to_execute:
            if kind == 'add':
                store = getattr(self.model.args, res)
                new_id = self.resource_capacity[res] + 1
                store.put(CustomResource(self.model.env, capacity=1, id_attribute=new_id))
                self.resource_capacity[res] += 1
            elif kind == 'remove':
                store = getattr(self.model.args, res)
                if isinstance(store, simpy.Store) and store.items:
                    store.items.pop(0)
                    self.resource_capacity[res] = max(0, self.resource_capacity[res]-1)


    def _apply_action(self, action):
        """Queue the chosen action for delayed execution."""
        kind, res = self.action_mapping.get(action, ('noop', None))
        if kind in ['add', 'remove'] and res:
            # Queue resource changes with a delay
            activation_time = self.current_time + self.resource_change_delay
            self.pending_resource_changes.append((activation_time, kind, res))
        else:
            # No-op or unknown: do nothing
            pass


    def _print_shift_summary(self, shift_reward):
        print("\n" + "="*40)
        print(f"=== SHIFT COMPLETED ===")
        print(f"Time: {self.current_time:.0f} minutes")

        # Active Staff
        print("Active Staff:")
        for unit, count in self.active_staff.items():
            print(f"  {unit}: {count} assigned")

        # Resting Staff
        print("Resting Staff:")
        for unit, count in self.resting_staff.items():
            print(f"  {unit}: {count} resting")

        # Discharges
        discharges = self._count_shift_discharges()
        print(f"Patients Discharged This Shift: {discharges}")

        # Fatigue
        avg_fatigue = self._average_fatigue_by_unit()
        avg_fatigue_shift = np.mean(list(avg_fatigue.values()))
        print("\n=== Fatigue Statistics ===")
        print(f"Average Fatigue Across Units: {avg_fatigue_shift:.2f}")
        print("Fatigue Breakdown by Unit:")
        for unit, fatigue in avg_fatigue.items():
            print(f"  {unit}: {fatigue:.1f}")


        # Active and Resting Staff Counts
        print("\n=== Staffing Summary ===")
        print("Active Staff per Unit:")
        for unit, count in self.active_staff.items():
            print(f"  {unit}: {count}")
        print("Resting Staff per Unit:")
        for unit, count in self.resting_staff.items():
            print(f"  {unit}: {count}")

        print("="*40)
        print(f"Shift Reward: {shift_reward:.2f}")
        print("="*40)
        print("="*40)

        # Reset wait trackers after shift
        self.wait_time_totals = {
            "triage": 0.0,
            "registration": 0.0,
            "exam": 0.0,
            "trauma": 0.0,
            "cubicle_1": 0.0,
            "cubicle_2": 0.0
        }
        self.wait_time_counts = {
            "triage": 0,
            "registration": 0,
            "exam": 0,
            "trauma": 0,
            "cubicle_1": 0,
            "cubicle_2": 0
        }


        

    def step(self, action):
        old_time = self.current_time

        # ─── SHIFT START PLANNING ────────────────────────────────────────────────
        if self.is_shift_planning:
            self.latest_mso_plan = self.mso_shift_planner()
            self._apply_shift_action(action)
            self.is_shift_planning = False
            self.time_in_shift = 0
            self.reward_this_shift = 0.0  # Reset reward tracker for this shift

        # ─── SIMULATION ADVANCEMENT ──────────────────────────────────────────────
        next_time = old_time + self.step_size
        self.model.env.run(until=next_time)
        self.current_time = next_time
        self.time_in_shift += self.step_size

        # ─── FATIGUE UPDATE ──────────────────────────────────────────────────────
        self._update_fatigue()

        # ─── REWARD COMPUTATION ───────────────────────────────────────────────────
        reward = self._compute_step_reward(old_time, next_time)
        self.reward_this_shift += reward

        # ─── SHIFT END CHECK ──────────────────────────────────────────────────────
        if self.time_in_shift >= self.shift_duration:
            self.is_shift_planning = True

            # Shift finished, print shift summary
            self._print_shift_summary(self.reward_this_shift)


            # Add shift bonus
            shift_bonus = self._compute_shift_bonus()
            reward += shift_bonus

            # Reset counters for new shift
            self.cur_reward = self.reward_this_shift
            self.reward_this_shift = 0.0


            print("Shift Completed with Bonus Reward", shift_bonus)

            # run results
            summary = SimulationSummary(self.model)
            summary.process_run_results_live() # Tire

            # Now you can access mid-shift wait times, etc:
            mean_triage_wait = summary.results.get('01a_triage_wait', 0.0)
            mean_exam_wait = summary.results.get('03a_examination_wait', 0.0)
            mean_treat_wait_nontrauma = summary.results.get('04a_treatment_wait(non_trauma)', 0.0)
            mean_treat_wait_trauma = summary.results.get('07a_treatment_wait(trauma)', 0.0)
            throughput = summary.results.get('09_throughput', 0.0)
            mean_ward_wait = summary.results.get('10a_ward_wait', 0.0)
            mean_icu_wait = summary.results.get('11a_icu_wait', 0.0)

            # You can now also PRINT or USE these for your reward shaping!
            print(f"[Summary] Mean Triage Wait: {mean_triage_wait:.2f} mins")
            print(f"[Summary] Mean Exam Wait: {mean_exam_wait:.2f} mins")
            print(f"[Summary] Mean Non-Trauma Treat Wait: {mean_treat_wait_nontrauma:.2f} mins")
            print(f"[Summary] Mean Trauma Treat Wait: {mean_treat_wait_trauma:.2f} mins")
            print(f"[Summary] Throughput: {throughput:.2f}")
            print(f"[Summary] Ward Wait: {mean_ward_wait:.2f}")
            print(f"[Summary] ICU Wait: {mean_icu_wait:.2f}")

            self.cumulative_triage_waits.append(summary.results.get('01a_triage_wait', 0.0))
            self.cumulative_registration_waits.append(summary.results.get('02a_registration_wait', 0.0))
            self.cumulative_exam_waits.append(summary.results.get('03a_examination_wait', 0.0))
            self.cumulative_trauma_waits.append(summary.results.get('05a_stabilisation_wait', 0.0))
            self.cumulative_cub1_waits.append(summary.results.get('04a_treatment_wait(non_trauma)', 0.0))
            self.cumulative_cub2_waits.append(summary.results.get('07a_treatment_wait(trauma)', 0.0))
            self.cumulative_fatigues.append(np.mean(list(self._average_fatigue_by_unit().values())))
            self.cumulative_throughput.append(summary.results.get('09_throughput', 0.0))
            
            

        # ─── PENDING RESOURCE CHANGES ─────────────────────────────────────────────
        self._process_pending_actions()

        # ─── OBSERVATION AND RETURN ───────────────────────────────────────────────
        obs = self._get_observation()
        done = (self.current_time >= self.rc_period)
        info = {}


        # ---- Accumulate for External Summary ----
        if not hasattr(self, "cumulative_rewards"):
            self.cumulative_rewards = []
            self.cumulative_triage_waits = []
            self.cumulative_exam_waits = []
            self.cumulative_registration_waits = []
            self.cumulative_trauma_waits = []
            self.cumulative_cub1_waits = []
            self.cumulative_cub2_waits = []
            self.cumulative_fatigues = []
            self.cumulative_throughput = []

        # Log this step reward
        self.cumulative_rewards.append(reward)

       


        return obs, reward, done, info



    
    # -----------------------------
    # RETURNS
    # -----------------------------
    def _get_observation(self):
        """
        Clean bottleneck-aware observation function for learnability and MSO integration.
        """
        summary = self.model.sim_summary
        args = self.model.args
        obs = []

        # --- System metrics ---
        obs += [
            summary.avg_total_wait_time,
            summary.avg_fatigue,
            summary.total_discharged,
        ]

        # --- Bottleneck stats per unit ---
        for unit in self.resource_targets:
            util = summary.utilization.get(unit, 0.0)
            qlen = summary.queue_lengths.get(unit, 0)
            cap = getattr(args, f"n_{unit}", 1)
            obs += [util, qlen / cap if cap > 0 else 0.0]

        # --- MSO plan deviation ---
        for i, unit in enumerate(self.resource_targets):
            planned = self.latest_mso_plan[i] if i < len(self.latest_mso_plan) else 0.0
            actual = summary.utilization.get(unit, 0.0)
            obs.append(actual - planned)

        obs = np.array(obs, dtype=np.float32)
        expected_dim = 3 + 2 * len(self.resource_targets) + len(self.resource_targets)
        assert obs.shape[0] == expected_dim, f"[OBS ERROR] Got shape {obs.shape}, expected {expected_dim}"

        if self.current_time < 500:
            print(f"[DEBUG] Obs shape: {obs.shape}, contents: {obs}")

        return obs








    # -----------------------------
    # REWARD
    # -----------------------------

    def _compute_resource_cost(self):
        """Calculate total resource operational cost."""
        total_cost = 0.0
        for res in self.resource_targets:
            if self.cost_mode == 'equal':
                cost = 1.0
            else:
                cost = self.cost_per_unit.get(res, 1.0)
            total_cost += self.resource_capacity[res] * cost
        return total_cost

    def _compute_step_reward(self, start_time, end_time):
        """
        Computes the reward for this step based on state improvements.
        Assumes we're penalizing for idle time, delay, and rewarding throughput.
        """
        summary = self.model.sim_summary  # updated after env.run
        reward = 0.0

        # Penalty for wait time
        reward -= summary.avg_total_wait_time * 0.1

        # Reward for throughput
        reward += summary.total_discharged * 1.0

        # Penalty for deviation from plan
        plan_error = sum(abs(summary.utilization.get(unit, 0.0) - self.latest_mso_plan[i])
                        for i, unit in enumerate(self.resource_targets))
        reward -= plan_error * 0.5

        # Bonus: Penalty for resource idleness
        idle_resources = 0
        for res in self.resource_targets:
            r = getattr(self.model.args, res)
            if hasattr(r, 'capacity') and hasattr(r, 'count'):
                idle_resources += (r.capacity - r.count)
            elif hasattr(r, 'items'):
                idle_resources += len(r.items)  # Store idle units
        reward -= idle_resources * 0.05

        # Optional: fatigue signal if you have it
        reward -= summary.avg_fatigue * 0.2

        return reward


    def _compute_shift_bonus(self):
        """Compute reward at the end of a shift."""
        discharges = self._count_shift_discharges()

        avg_wait = self.model.get_overall_weighted_wait_time()


        total_fatigue = 0.0
        active_count = 0
        for res_name in self.resource_targets:
            store = getattr(self.model.args, res_name)
            if hasattr(store, "items"):
                for resource in store.items:
                    if hasattr(resource, "fatigue"):
                        total_fatigue += resource.fatigue
                        active_count += 1
        avg_fatigue = (total_fatigue / active_count) if active_count > 0 else 0.0

        resource_cost = sum(self.active_staff[k] * self.cost_per_unit.get(k, 1.0) for k in self.active_staff)

        shift_bonus = (
            + 50 * discharges
            - 2.0 * avg_wait
            - 3.0 * avg_fatigue
            - 1.0 * resource_cost
        )
        return np.clip(shift_bonus, -500, 500)


  






    def render(self, mode='human'):
        pass

    def close(self):
        pass

# hospital_env.py

import gym
from gym import spaces
import numpy as np
import simpy
from model_classes import Scenario, TreatmentCentreModel, CustomResource

class HospitalSimEnv(gym.Env):
    """
    Hospital ED simulation environment for RL (resource allocation & patient flow).
    Continuous-time SimPy model controlled in discrete time steps.
    """
    metadata = {'render.modes': []}

    def __init__(self, sim_config, step_size=60, alpha=5.0, beta=0.2, gamma=0.1, delta=0.01, epsilon=0.05, rc_period=20160, cost_mode='diverse'):

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

        # Resource definitions
        self.resource_targets = ['triage', 'registration', 'exam', 'trauma', 'cubicle_1', 'cubicle_2']
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

        # Shift management
        self.shift_duration = 480  # 8 hours
        self.time_in_shift = 0
        self.is_shift_planning = True
        self.total_staff_pool = {
            'triage': 20,
            'registration': 20,
            'exam': 15,
            'trauma': 10,
            'cubicle_1': 25,
            'cubicle_2': 10
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
        obs_dim = 1 + 2 * len(self.resource_targets) + 2 + len(self.resource_targets)
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

        # Create new scenario
        self.scenario = Scenario(**cfg)
        self.model = TreatmentCentreModel(self.scenario)
        self.model.env.process(self.model.arrivals_generator())

        self.current_time = 0.0

        self.latest_mso_plan = self.mso_shift_planner()  # Generate MSO plan for first shift

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
        print(f"\n=== SHIFT COMPLETED ===")
        print(f"Time: {self.current_time:.0f} minutes")
        print(f"Active Staff:")
        for k, v in self.active_staff.items():
            print(f"  {k}: {v} assigned")
        print(f"Resting Staff:")
        for k, v in self.resting_staff.items():
            print(f"  {k}: {v} resting")
        discharges = self._count_shift_discharges()
        print(f"Patients Discharged This Shift: {discharges}")
        avg_fatigue = self._average_fatigue_by_unit()
        print(f"Average Fatigue by Unit:")
        for unit, fatigue in avg_fatigue.items():
            print(f"  {unit}: {fatigue:.1f}")

        print("="*40)
        print(f"=== SHIFT SUMMARY ===")
        print(f"Shift Time: {self.current_time:.0f} minutes")
        print(f"Total Reward This Shift: {self.reward_this_shift:.2f}")
        avg_fatigue_shift = np.mean(list(avg_fatigue.values()))
        print(f"Average Fatigue Across Units: {avg_fatigue_shift:.2f}")

        print("Fatigue Breakdown by Unit:")
        for unit, fat in avg_fatigue.items():
            print(f"  {unit}: {fat:.1f}")

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
        reward = self._compute_reward(old_time, next_time)
        self.reward_this_shift += reward

        # ─── SHIFT END CHECK ──────────────────────────────────────────────────────
        if self.time_in_shift >= self.shift_duration:
            self.is_shift_planning = True

            # Shift finished, print shift summary
            self._print_shift_summary(self.reward_this_shift)

        # ─── PENDING RESOURCE CHANGES ─────────────────────────────────────────────
        self._process_pending_actions()

        # ─── OBSERVATION AND RETURN ───────────────────────────────────────────────
        obs = self._get_observation()
        done = (self.current_time >= self.rc_period)
        info = {}

        return obs, reward, done, info



    
    # -----------------------------
    # RETURNS
    # -----------------------------

    def _get_observation(self):
        """Construct observation vector."""
        obs = [self.current_time]
        log = self.model.full_event_log

        def queue_len(event_name, start_tag):
            waits = sum(1 for e in log if e['event'] == event_name)
            starts = sum(1 for e in log if e['event'] == start_tag)
            return max(0, waits - starts)

        # Queue + Busy for each resource
        for res, wait_tag, start_tag in zip(
            ['triage', 'registration', 'exam', 'trauma', 'cubicle_1', 'cubicle_2'],
            ['triage_wait_begins', 'MINORS_registration_wait_begins', 'MINORS_examination_wait_begins',
             'TRAUMA_stabilisation_wait_begins', 'MINORS_treatment_wait_begins', 'TRAUMA_treatment_wait_begins'],
            ['triage_begins', 'MINORS_registration_begins', 'MINORS_examination_begins',
             'TRAUMA_stabilisation_begins', 'MINORS_treatment_begins', 'TRAUMA_treatment_begins']):
            q = queue_len(wait_tag, start_tag)
            b = self.resource_capacity[res] - len(getattr(self.model.args, res).items)
            obs += [q, b]

        trauma_count = nontrauma_count = 0
        for e in log:
            if e['time'] > self.current_time - self.step_size and e['time'] <= self.current_time:
                if e['event'] == 'triage_wait_begins':
                    if e['pathway'] == 'Trauma':
                        trauma_count += 1
                    elif e['pathway'] == 'Non-Trauma':
                        nontrauma_count += 1
        obs += [trauma_count, nontrauma_count]

        obs += list(self.latest_mso_plan)
        return np.array(obs, dtype=np.float32)



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

    def _compute_reward(self, t0, t1):
        new_events = [e for e in self.model.full_event_log if t0 < e['time'] <= t1]
        
        # 1. Throughput
        discharges = sum(1 for e in new_events if e['event'] == 'depart')
        active_staff_total = sum(self.active_staff.values())
        efficiency_score = (discharges / active_staff_total) if active_staff_total > 0 else 0

        # 2. Queue load
        total_queue = 0
        for res, wait_tag, start_tag in zip(
            ['triage', 'registration', 'exam', 'trauma', 'cubicle_1', 'cubicle_2'],
            ['triage_wait_begins', 'MINORS_registration_wait_begins', 'MINORS_examination_wait_begins',
            'TRAUMA_stabilisation_wait_begins', 'MINORS_treatment_wait_begins', 'TRAUMA_treatment_wait_begins'],
            ['triage_begins', 'MINORS_registration_begins', 'MINORS_examination_begins',
            'TRAUMA_stabilisation_begins', 'MINORS_treatment_begins', 'TRAUMA_treatment_begins']):
            total_queue += max(0, sum(1 for e in self.model.full_event_log if e['event'] == wait_tag) -
                                sum(1 for e in self.model.full_event_log if e['event'] == start_tag))

        # 3. Cost
        resource_cost = sum(self.cost_per_unit.get(res, 1.0) * self.active_staff.get(res, 0) for res in self.resource_targets)

        # 4. Fatigue
        total_fatigue = 0.0
        total_active = 0
        for res_name in self.resource_targets:
            store = getattr(self.model.args, res_name)
            if hasattr(store, "items"):
                for r in store.items:
                    if hasattr(r, "fatigue"):
                        total_fatigue += r.fatigue
                        total_active += 1
        avg_fatigue = (total_fatigue / total_active) if total_active > 0 else 0.0

        # 5. Surge survival (Optional)
        # For now assume 0
        surge_bonus = 0

        reward = (
            10.0 * efficiency_score
            - 0.5 * total_queue
            - 0.05 * resource_cost
            - 0.1 * avg_fatigue
            + surge_bonus
        )

        return reward





    def render(self, mode='human'):
        pass

    def close(self):
        pass

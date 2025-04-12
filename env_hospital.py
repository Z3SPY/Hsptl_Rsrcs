import simpy
import numpy as np
import random
import time
import datetime
import gym
from gym import spaces

# Import your scenario and model from modelclass
from modelclass import Scenario, WardFlowModel, PatientFlow
import sys




###################################################
# MultiDiscrete Action Wrapper
###################################################
class HospitalActionSpace:
    """
    MultiDiscrete for:
      [ICU_alloc, MedSurg_alloc, nurse_shift, MSO_flag]

    Where:
      - ICU_alloc: 0..max_icu
      - MedSurg_alloc: 0..max_medsurg
      - nurse_shift: 0..max_nurse_shift (0=none, 1=+1, 2=+2, 3=-1 or custom)
      - MSO_flag: 0 or 1
    """
    def __init__(self, max_icu=5, max_medsurg=5, max_nurse_shift=3):
        self.max_icu = max_icu
        self.max_medsurg = max_medsurg
        self.max_nurse_shift = max_nurse_shift
        self.action_space = spaces.MultiDiscrete([
            max_icu + 1,
            max_medsurg + 1,
            max_nurse_shift + 1,
            2
        ])

    def sample(self):
        return self.action_space.sample()

    def contains(self, action):
        return self.action_space.contains(action)

    def decode(self, action):
        icu_alloc, med_alloc, nurse_val, mso_flag = action
        return {
            "icu": int(icu_alloc),
            "medsurg": int(med_alloc),
            "nurse_shift": int(nurse_val),
            "mso": bool(mso_flag)
        }

###################################################
# The Gym Environment
###################################################
class HospitalEnv(gym.Env):
    """
    OpenAI Gym-compatible environment that wraps the SimPy hospital simulation.
    Integrates an MSO planner for periodic planning, DRL agent for short-term control,
    and includes logging enhancements.

    Key expansions:
      - MultiDiscrete action space
      - Snapshot/restore for no-deepcopy rollouts
      - Overfitting fix: clamp or penalize total bed usage
      - Step-based running with a custom reward
    """
    def __init__(self,
                 scenario: Scenario,
                 # max for multi-discrete
                 max_icu=24*24,
                 max_medsurg=24,
                 max_nurse_shift=3,
                 # MSO integration
                 use_mso=False,
                 mso_planner=None,
                 mso_frequency_hours=8,
                 # Simulation step control
                 step_minutes=60,
                 # Logging toggles, if needed
                 debug_logs=False):

        super().__init__()

        # Store scenario & MSO
        self.scenario = scenario
        self.use_mso = use_mso
        self.mso_planner = mso_planner
        self.mso_interval = mso_frequency_hours * 60.0  # convert hours -> minutes
        self.next_plan_time = 0.0
        self.step_minutes = step_minutes
        self.debug_logs = debug_logs


        self.planning_mode = False


        # Build multi-discrete action space
        self.hospital_action = HospitalActionSpace(max_icu, max_medsurg, max_nurse_shift)
        self.action_space = self.hospital_action.action_space

        # Observation space: e.g. [ED_in_use, ICU_free, MedSurg_free, Nurses_free, recommended_icu?]
        # We'll keep it simple: [ed_in_use, icu_available, med_available, nurse_free, placeholder]
        obs_high = np.array([
            scenario.n_ed_beds,
            scenario.n_icu_beds + scenario.n_medsurg_beds,
            scenario.n_icu_beds + scenario.n_medsurg_beds,
            scenario.day_shift_nurses + scenario.night_shift_nurses,
            scenario.n_icu_beds + scenario.n_medsurg_beds
        ], dtype=np.float32)
        self.observation_space = spaces.Box(
            low=0.0,
            high=obs_high,
            shape=(5,),
            dtype=np.float32
        )

        # We'll hold a reference to the WardFlowModel
        self.model = None

        # Additional state variables for nurse cost or bed usage
        self.current_icu_beds = scenario.n_icu_beds
        self.current_medsurg_beds = scenario.n_medsurg_beds
        self.nurse_change_cost = 0.0
        self.cumulative_reward = 0.0
        self.episode_start_time = None

        #Reward    
        self.last_reward_time = 0.0  # This prevents the AttributeError
        self.shift_reward_acc = 0.0  
        self.shift_duration = 480  


    def reset(self):
        """
        Initialize a new simulation episode:
         - Build the underlying WardFlowModel from scenario
         - Start background processes
         - Reset bed usage, nurse costs, logs
        """
        
        # Build the model
        self.model = WardFlowModel(self.scenario, start_datetime=datetime.datetime.now())
        # Run up to 0 to initialize everything

        self.model.run(0)

        # Reset everything
        self.current_icu_beds = self.scenario.n_icu_beds
        self.current_medsurg_beds = self.scenario.n_medsurg_beds
        self.nurse_change_cost = 0.0
        self.cumulative_reward = 0.0
        self.episode_start_time = time.time()

        # If we do MSO planning at time=0, do it
        self.next_plan_time = 0.0
        self.last_reward_time = 0.0
        self.shift_reward_acc = 0.0  

        obs = self._get_obs() 

        print("[DEBUG] Resetting environment")
        if isinstance(obs, np.ndarray):
            print(f"[DEBUG] Reset obs shape: {obs.shape}")
        else:
            print(f"[DEBUG] Reset obs type: {type(obs)} => {obs}")


        return obs
    

    # GET REWARD

    def get_final_state(self):
        final_state = {
            "Time": self.model.env.now,
            "ICU_tokens": len(self.model.icu.items),
            "ICU_capacity": self.current_icu_beds,
            "MedSurg_tokens": len(self.model.medsurg.items),
            "MedSurg_capacity": self.current_medsurg_beds,
            "Nurses_available": self.model.nurses.capacity - self.model.nurses.count,
            "ED_usage": self.model.ed.count,
        }
        # You can add more details depending on your WardFlowModel structure.
        return final_state


    def step(self, action):
        """
        Executes one simulation step based on the given action.
        Aggregates reward over a defined shift_duration.
        """
        # Ensure model is initialized.
        if self.model is None:
            print("[WARNING] 'model' is None; calling reset() automatically.")
            obs = self.reset()
            return obs, 0.0, False, {"auto_reset": True}

        # 1. Planner override logic (as implemented previously) remains.
        if self.use_mso and (self.model.env.now >= self.next_plan_time):
            planned_action, planned_value = self.mso_planner.best_action(self)
            if self.debug_logs:
                print(f"[DEBUG] Planner override activated at time {self.model.env.now}.")
                print(f"[DEBUG] Planner selected action: {planned_action} with estimated value: {planned_value}.")
            action = planned_action
            self.next_plan_time = self.model.env.now + self.mso_interval

        # 2. Decode and apply the action.
        decoded = self.hospital_action.decode(action)
        new_icu = decoded["icu"]
        new_medsurg = decoded["medsurg"]
        nurse_shift_val = decoded["nurse_shift"]

        # Update ICU capacity.
        if new_icu != self.current_icu_beds:
            if new_icu > self.current_icu_beds:
                added = new_icu - self.current_icu_beds
                for i in range(added):
                    self.model.icu.put(f"ICU_Bed_Extra_{i+1}")
            else:
                reduction = self.current_icu_beds - new_icu
                available = len(self.model.icu.items)
                if available >= reduction:
                    for i in range(reduction):
                        self.model.icu.items.pop(0)
                else:
                    print("[WARNING] Not enough ICU tokens available to reduce capacity as requested.")
            self.current_icu_beds = new_icu

        # Update MedSurg capacity.
        if new_medsurg != self.current_medsurg_beds:
            if new_medsurg > self.current_medsurg_beds:
                added = new_medsurg - self.current_medsurg_beds
                for i in range(added):
                    self.model.medsurg.put(f"MedSurg_Bed_Extra_{i+1}")
            else:
                reduction = self.current_medsurg_beds - new_medsurg
                available = len(self.model.medsurg.items)
                if available >= reduction:
                    for i in range(reduction):
                        self.model.medsurg.items.pop(0)
                else:
                    print("[WARNING] Not enough MedSurg tokens available to reduce capacity as requested.")
            self.current_medsurg_beds = new_medsurg

        # Update nurse capacity.
        baseline = self.scenario.day_shift_nurses  # This could be dynamic based on current time.
        new_nurse_capacity = baseline + nurse_shift_val
        self.model.nurses.capacity = new_nurse_capacity
        # Optionally update a current nurse capacity variable.
        # self.current_nurse_capacity = new_nurse_capacity

        # 3. Advance the simulation.
        self.model.run(self.step_minutes)  # Advance simulation by step_minutes.
        current_time = self.model.env.now

        # 4. Compute reward for the time interval since last reward calculation.
        # Get events from the event log with times in (last_reward_time, current_time].
        recent_events = [e for e in self.model.event_log if self.last_reward_time < e["time"] <= current_time]
        step_reward = self.compute_reward_from_events(recent_events)
        self.shift_reward_acc += step_reward

        # 5. Check if the shift duration has elapsed.
        if current_time - self.last_reward_time >= self.shift_duration:
            aggregated_reward = self.shift_reward_acc
            # Reset accumulator and update reward marker.
            self.shift_reward_acc = 0.0
            self.last_reward_time = current_time
            reward_to_return = aggregated_reward
            if self.debug_logs:
                print(f"[SHIFT REWARD] Time: {current_time:.2f}, Aggregated Reward: {aggregated_reward:.2f}")
        else:
            reward_to_return = 0.0  # No aggregated reward until a shift is complete.

        # 6. Get updated observation.
        obs = self._get_obs()
        # 7. Check termination condition.
        done = current_time >= self.scenario.simulation_time
        info = {"current_time": current_time, "shift_reward": reward_to_return}
        self.cumulative_reward += reward_to_return

        # 8. Optionally, at the end of an episode, log final state.
        if done:
            final_state = self.get_final_state()
            print("[FINAL OBSERVATION]", final_state)
            info["final_state"] = final_state

        return obs, reward_to_return, done, info





    def _get_obs(self):
        # Construct an observation vector that includes current resource statuses
        obs = np.array([
            self.scenario.n_ed_beds - self.model.ed.count,  # ED available
            len(self.model.icu.items),                      # ICU available (tokens)
            len(self.model.medsurg.items),                  # MedSurg available (tokens)
            self.model.nurses.capacity - self.model.nurses.count,  # Nurses available
            # Optionally, add more aggregated metrics, e.g., average wait time from the audit.
            np.mean([r.get('ed_in_use', 0) for r in self.model.utilisation_audit])  # simple placeholder
        ], dtype=np.float32)
        return obs


    
    
    def _compute_reward(self):
        current_time = self.model.env.now
        if not hasattr(self, "last_reward_time"):
            self.last_reward_time = 0

        relevant_events = [e for e in self.model.event_log if self.last_reward_time < e['time'] <= current_time]

        r1_discharge = 0
        r2_delay_penalty = 0
        r3_unserved = 0
        r4_resource_penalty = 0
        r5_queue_penalty = 0
        r6_transfer_bonus = 0
        r7_boarding_penalty = 0

        for event in relevant_events:
            if event.get('event') == 'discharge':
                if event.get('los_deviation', 0) <= 0:
                    r1_discharge += 1
                else:
                    r2_delay_penalty += event.get('los_deviation', 0)
            elif event.get('event_type') == 'complication':
                r7_boarding_penalty += 1
            elif event.get('event') == 'transfer':
                r6_transfer_bonus += 1

        # Resource overuse
        extra_icu = max(0, self.current_icu_beds - self.scenario.n_icu_beds)
        extra_med = max(0, self.current_medsurg_beds - self.scenario.n_medsurg_beds)
        base_nurse = self.scenario.day_shift_nurses if (int(self.model.env.now // 60) % 24) in range(7, 19) else self.scenario.night_shift_nurses
        extra_nurses = max(0, self.model.nurses.capacity - base_nurse)
        r4_resource_penalty = extra_icu + extra_med + extra_nurses

        # Queue penalty (ED queue only for now)
        queue_len = len(self.model.ed.queue) if hasattr(self.model.ed, 'queue') else 0
        r5_queue_penalty = queue_len

        reward = (
            2.0 * r1_discharge +
            -0.1 * r2_delay_penalty +
            -5.0 * r3_unserved +
            -0.5 * r4_resource_penalty +
            -0.2 * r5_queue_penalty +
            1.0 * r6_transfer_bonus +
            -10.0 * r7_boarding_penalty
        )


        debug = True
        if self.debug_logs and debug:
            print(f"[REWARD DEBUG] Time={self.model.env.now}, "
              f"Discharges={r1_discharge}, DelayPenalty={r2_delay_penalty}, "
              f"Unserved={r3_unserved}, ResourcePenalty={r4_resource_penalty}, "
              f"QueuePenalty={r5_queue_penalty}, Transfers={r6_transfer_bonus}, "
              f"Boarding={r7_boarding_penalty}, TotalReward={reward}")

        self.last_reward_time = current_time
        return np.clip(reward, -100, 100)

    

    def compute_reward_from_events(self, events):
        r_discharges = sum(1 for e in events if e.get("event") == "discharge" and e.get("los_deviation", 0) <= 0)
        r_delay = sum(e.get("los_deviation", 0) for e in events if e.get("event") == "discharge" and e.get("los_deviation", 0) > 0)
        r_complications = sum(1 for e in events if e.get("event_type") == "complication")
        r_transfers = sum(1 for e in events if e.get("event") == "transfer")
        # For the queue, compute a dynamic penalty, if our model provides a proper ED queue.
        queue_penalty = 0
        if hasattr(self.model, "ed") and hasattr(self.model.ed, "queue"):
            for patient in self.model.ed.queue:
                wait_time = self.model.env.now - getattr(patient, "arrival_time", 0)
                severity = getattr(patient, "severity", 1)
                queue_penalty += severity * (wait_time / 60)
        
        # Resource cost could be a function of extra resources used.
        extra_resources = (max(0, self.current_icu_beds - self.scenario.n_icu_beds) +
                        max(0, self.current_medsurg_beds - self.scenario.n_medsurg_beds))
        
        # Define weight factors
        w_discharges = 1.0
        w_delay = -0.5
        w_complications = -1.5
        w_transfers = 0.5
        w_queue = -0.3
        w_resource = -0.2
        
        reward = (w_discharges * r_discharges +
                w_delay * r_delay +
                w_complications * r_complications +
                w_transfers * r_transfers +
                w_queue * queue_penalty +
                w_resource * extra_resources)
        
        debug = False
        if self.debug_logs and debug:
            print(f"[REWARD DEBUG] Time={self.model.env.now}, "
              f"Discharges={r_discharges}, DelayPenalty={r_delay}, "
              f"Unserved={r_complications}, ResourcePenalty={r_transfers}, "
              f"QueuePenalty={queue_penalty}, Transfers={extra_resources}, "
              f"TotalReward={reward}")

        return reward



    #######################################################
    # Snapshot / Restore for advanced planning or MCTS
    #######################################################
    def snapshot_state(self):
        """
        Minimal but sufficient environment state capturing:
        - sim time
        - bed usage & nurse cost
        - RNG states
        - Patient data
        - If you need store items or resource usage, add them here
        """
        return {
            "sim_time": self.model.env.now,
            "icu_beds": self.current_icu_beds,
            "med_beds": self.current_medsurg_beds,
            "nurse_cost": self.nurse_change_cost,
            "rng_np": np.random.get_state(),
            "rng_py": random.getstate(),
            # Rebuild patients
            "patients": [p.to_dict() for p in self.model.patients],
        }

    def restore_state(self, snap):
        
        #Rebuild the environment from the snapshot dictionary.
        #This is used instead of a heavy `deepcopy(self)`.
        
        self.model.env._now = snap["sim_time"]
        self.current_icu_beds = snap["icu_beds"]
        self.current_medsurg_beds = snap["med_beds"]
        self.nurse_change_cost = snap["nurse_cost"]

        np.random.set_state(snap["rng_np"])
        random.setstate(snap["rng_py"])

        # Rebuild patients
        self.model.patients = []
        for data in snap["patients"]:
            self.model.patients.append(PatientFlow.from_dict(data))

        # If you also want to restore store items for ICU or MedSurg, do that here:
        # e.g., self.model.icu.items = snapshot["icu_items"]
        # That might require you to store them in snapshot_state too.

        # Done
        if self.debug_logs:
            print(f"[RestoreState] Reverted to time={snap['sim_time']} with {len(self.model.patients)} patients")


###################################################
# If you need a quick test:
###################################################
if __name__ == "__main__":
    # Create a scenario with 24 hours simulation time and baseline resource capacities.
    scenario = Scenario(
        simulation_time=60 * 24,  # 24 hours in seconds
        random_number_set=random.randint(0,sys.maxsize),
        n_icu_beds=32,
        n_medsurg_beds=32
    )
    
    # Create the environment with desired maximum capacities (we want to allow an increase)
    env = HospitalEnv(
        scenario,
        max_icu=70,          # maximum ICU capacity that agent can set
        max_medsurg=70,      # maximum MedSurg capacity that agent can set
        max_nurse_shift=10,
        debug_logs=True
    )
    
    # Ensure the environment is reset first
    initial_obs = env.reset()
    print("[TEST] Environment reset complete. Initial observation:", initial_obs)
    
    # For testing, create a test action that sets:
    # - ICU capacity to 6 (an increase from the baseline 4)
    # - MedSurg capacity to 8 (an increase from the baseline 4)
    # - Nurse shift change is set to 0 (no change)
    # - MSO flag is 0 (not triggering the planner)
    test_action = np.array([6, 8, 0, 0])
    print("[TEST] Test action:", test_action)
    
    # Call step() with this test action
    obs, reward, done, info = env.step(test_action)
    
    # Print the number of tokens in the ICU and MedSurg stores to verify dynamic update.
    print("Post-step ICU tokens count:", len(env.model.icu.items))
    print("Post-step MedSurg tokens count:", len(env.model.medsurg.items))
    
    # Optionally, run a simple simulation loop for further testing.
    done = False
    total_reward = 0.0
    step_count = 0
    
    while not done:
        # Using random actions from the action space
        act = env.action_space.sample()
        obs, rew, done, info = env.step(act)
        total_reward += rew
        step_count += 1
    
    print(f"Episode ended. Steps={step_count}, total reward={total_reward}")
    if "episode_summary" in info:
        print("Episode summary:", info["episode_summary"])

    

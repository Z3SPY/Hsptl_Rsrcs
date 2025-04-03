import simpy
import numpy as np
import random
import time
import datetime
import gym
from gym import spaces

# Import your scenario and model from modelclass
from modelclass import Scenario, WardFlowModel, Patient

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
                 max_icu=5,
                 max_medsurg=5,
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

        obs = self._get_obs() 

        if isinstance(obs, np.ndarray):
            print(f"[DEBUG] Reset obs shape: {obs.shape}")
        else:
            print(f"[DEBUG] Reset obs type: {type(obs)} => {obs}")


        return obs

    def step(self, action):
        """
        1) Possibly override 'action' with MSO if use_mso=True and time >= next_plan_time
        2) Decode multi-discrete action
        3) Reallocate beds, track nurse shift cost
        4) Advance simulation by step_minutes or until scenario end
        5) Compute reward, return (obs, reward, done, info)
        """
       

        # If using MSO, override action at intervals
        current_time = self.model.env.now

        if self.use_mso and self.mso_planner is not None and current_time >= self.next_plan_time and not self.planning_mode:
            if hasattr(self.mso_planner, 'best_action'):
                action, _ = self.mso_planner.best_action(self)  # MCTS
            else:
                action = self.mso_planner.plan(self)  # LP/MSO
            self.next_plan_time += self.mso_interval

        # Decode
        decoded = self.hospital_action.decode(action)
        icu_target = decoded["icu"]
        med_target = decoded["medsurg"]
        nurse_shift_val = decoded["nurse_shift"]
        # mso_flag = decoded["mso"] # can be used for logging or ignoring

        # Check for over-capacity usage
        max_capacity = self.scenario.n_icu_beds + self.scenario.n_medsurg_beds
        combined = icu_target + med_target
        if combined > max_capacity:
            # Simple clamp or penalty
            over = combined - max_capacity
            med_target = max(med_target - over, 0)
            if self.debug_logs:
                print(f"[OverCapacity] Clamping ICU={icu_target}, MED={med_target + over} to maintain total <= {max_capacity}")

        # Reallocate
        self.current_icu_beds = icu_target
        self.current_medsurg_beds = med_target

        # Nurse shift changes
        # e.g. 0=none,1=+1,2=+2,3=-1
        if nurse_shift_val == 1:
            self.nurse_change_cost += 1.0
        elif nurse_shift_val == 2:
            self.nurse_change_cost += 2.0
        elif nurse_shift_val == 3:
            self.nurse_change_cost += 0.5

        # Advance simulation
        old_time = self.model.env.now
        next_time = old_time + self.step_minutes
        done = False
        if next_time >= self.scenario.simulation_time:
            next_time = self.scenario.simulation_time
            done = True

        self.model.run(next_time)

        # Build obs
        obs = self._get_obs()
        # Compute reward
        reward = self._compute_reward()
        self.cumulative_reward += reward

        info = {}
        if done:
            # Summaries
            elapsed = time.time() - self.episode_start_time
            info["episode_summary"] = {
                "total_reward": self.cumulative_reward,
                "final_time": self.model.env.now,
                "elapsed_seconds": round(elapsed, 2)
            }
            if self.debug_logs:
                print(f"[EpisodeEnd] Steps done. Reward={self.cumulative_reward}")


        # ADDED THIS ====================================================================
        
        # (Manual 2 and 3) Check action shape and type
        if not isinstance(action, (list, np.ndarray)):
            raise ValueError(f"[ERROR] Expected MultiDiscrete vector, got: {type(action)} => {action}")
        # Expect length=4 for your [6,6,4,2] space
        if len(action) != 4:
            raise ValueError(f"[ERROR] Action length mismatch. Got len={len(action)} != 4")
        
        
        #FOR DEBUGGING
        #print(f"[DEBUG] PPO selected action: {action}")

        # ADDED THIS ====================================================================


        return obs, reward, done, info

    def _get_obs(self):
        """
        Observation includes ED usage, ICU availability, MedSurg availability,
        nurse availability, and a placeholder dimension.
        """
        ed_in_use = self.model.ed.count
        icu_avail = len(self.model.icu.items)
        med_avail = len(self.model.medsurg.items)
        nurse_free = self.model.nurses.capacity - self.model.nurses.count
        placeholder = 0.0
        return np.array([ed_in_use, icu_avail, med_avail, nurse_free, placeholder], dtype=np.float32)

    def _compute_reward(self):
        """
        Example reward:
         + base
         - penalty if total bed usage exceeds scenario
         - nurse shift cost
         - possibly more (like ED usage penalty, or positivity if throughput is high)
        """
        R_base = 10.0

        # Over capacity penalty
        total_alloc = self.current_icu_beds + self.current_medsurg_beds
        capacity = self.scenario.n_icu_beds + self.scenario.n_medsurg_beds
        over_capacity_pen = 0.0
        if total_alloc > capacity:
            over_capacity_pen = 5.0 * (total_alloc - capacity)

        nurse_cost = self.nurse_change_cost
        self.nurse_change_cost = 0.0  # reset each step

        reward = R_base - over_capacity_pen - nurse_cost
        
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
        """
        Rebuild the environment from the snapshot dictionary.
        This is used instead of a heavy `deepcopy(self)`.
        """
        self.model.env._now = snap["sim_time"]
        self.current_icu_beds = snap["icu_beds"]
        self.current_medsurg_beds = snap["med_beds"]
        self.nurse_change_cost = snap["nurse_cost"]

        np.random.set_state(snap["rng_np"])
        random.setstate(snap["rng_py"])

        # Rebuild patients
        self.model.patients = []
        for data in snap["patients"]:
            self.model.patients.append(Patient.from_dict(data))

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
    from modelclass import Scenario
    scenario = Scenario(simulation_time=24*60, n_icu_beds=4, n_medsurg_beds=4)
    env = HospitalEnv(scenario, max_icu=4, max_medsurg=4, max_nurse_shift=3, debug_logs=True)
    obs = env.reset()

    done = False
    total_r = 0
    step_count = 0

    while not done:
        # random action
        act = env.action_space.sample()
        obs, rew, done, info = env.step(act)
        total_r += rew
        step_count += 1

    print(f"Episode ended. Steps={step_count}, total reward={total_r}")
    if "episode_summary" in info:
        print(info["episode_summary"])

"""
Refactored Hospital Simulation Module
---------------------------------------
This module implements:
  • A Scenario class that holds simulation parameters.
  • A set of resource classes (CareUnit, NurseGroup) and a Patient class.
  • A HospitalFlowModel that orchestrates patient arrivals, ED/ICU/MedSurg flows,
    boarding, transfers, and discharge management.
  • A HospitalActionSpace class defining an 8-dimensional action space for DRL.
  • An apply_action() method to modify system parameters based on the action.
  • A __main__ block for debugging/testing that runs a simulation and prints a snapshot.
  
This module is designed to integrate with the front end (basic_app.py) without changes.
"""    

import json
import simpy
import numpy as np
from gymnasium import spaces
import json
import pandas as pd
import os
from collections import defaultdict
from scipy.stats import norm  
from functions.helper import convert_mean_std_to_lognorm, load_empirical_los, load_transition_matrix


# ─────────────────────────────────────────────────────────────────────────────
# 0. Load empirical parameters
# ─────────────────────────────────────────────────────────────────────────────
with open("DataAnalysis/empirical_params.json") as f:
    _js = json.load(f)
LAMBDA_HOUR    = _js["lambda_hour"]        # 24‑vector of λ arrivals/hr
SEVERITY_PROBS = _js["acuity_probs"]        

LOS_PARAMS     = {
    1: tuple(_js["lognorm_params"]["1"]),
    2: tuple(_js["lognorm_params"]["2"]),
    3: tuple(_js["lognorm_params"]["3"]),
    4: tuple(_js["lognorm_params"]["4"]),
    5: tuple(_js["lognorm_params"]["5"])
}



# ─────────────────────────────────────────────────────────────────────────────
# 1. Scenario
# ─────────────────────────────────────────────────────────────────────────────
class Scenario:
    def __init__(
        self,
        simulation_time: float       = 60*24*7,
        shift_length:    float       = 8*60,
        n_ed_beds:       int         = 15,
        n_icu_beds:      int         = 4,
        n_medsurg_beds:  int         = 10,
        day_shift_nurses:int         = 10,
        night_shift_nurses:int       = 5,
        n_observation_beds:int       = 10,
        arrival_profile: list[float] = LAMBDA_HOUR,   # λ per hour
        severity_probs:  list[float] = SEVERITY_PROBS,   # P(severity=1,2,3)
        acuity_los_csv:  str         = "DataAnalysis/graphss/acuity_duration_stats.csv",
        within_stay_csv: str         = "DataAnalysis/graph/P_within_stay.csv",
        exit_probs_csv:  str         = "DataAnalysis/graph/P_exit_by_group.csv",
        los_cap_quantile:float       = 0.99
    ):
        # 1. Core simulation parameters
        self.simulation_time    = simulation_time
        self.shift_length       = shift_length
        self.n_ed_beds          = n_ed_beds
        self.n_icu_beds         = n_icu_beds
        self.n_medsurg_beds     = n_medsurg_beds
        self.day_shift_nurses   = day_shift_nurses
        self.night_shift_nurses = night_shift_nurses
        self.n_observation_beds = n_observation_beds  # Add this line


        # 2. Arrival & severity profile
        self.arrival_profile = arrival_profile
        self.severity_probs  = severity_probs 

        # ────────────────────────────────────────────────
        # 3. Empirical LOS: load CSV & compute log‑normal
        # ────────────────────────────────────────────────
        df = pd.read_csv(acuity_los_csv)
        # Only keep ED (“Emergency”), ICU, Ward (“Medical Ward”)
        unit_map = {"Emergency": "ED", "ICU": "ICU", "Medical Ward": "Ward"}
        self.los_params = load_empirical_los("DataAnalysis/empirical_params.json")

        
        for _, row in df.iterrows():
            raw_unit = row["unit_group"]
            if raw_unit not in unit_map:
                continue

            unit = unit_map[raw_unit]
            acuity = int(float(row["acuity"]))  # ensures 1.0 → 1, etc.
            mean = row["mean_min"]
            var  = row["var_min"]

            if np.isnan(mean) or np.isnan(var) or var <= 0 or mean <= 0:
                continue

            sigma2 = np.log(1 + var / mean**2)
            mu = np.log(mean / np.sqrt(1 + var / mean**2))
            sigma = np.sqrt(sigma2)

            if unit not in self.los_params:
                self.los_params[unit] = {}

            self.los_params[unit][acuity] = (mu, sigma)

        # After parsing LOS from CSV, ensure all 5 acuities exist for each unit
        default_mu, default_sigma = np.log(120), 1.0  # default fallback

        for unit in ["ED", "ICU", "Ward"]:
            for acuity in range(1, 6):  # 1 to 5
                if acuity not in self.los_params[unit]:
                    self.los_params[unit][acuity] = (default_mu, default_sigma)


        # 4. Within-stay transition matrix (unchanged)
        if os.path.isfile(within_stay_csv):
            df_w = pd.read_csv(within_stay_csv, index_col=0)
            self.within_stay_probs = df_w.to_dict(orient="index")
        else:
            self.within_stay_probs = {
                "ED":       {"Discharge": 0.20, "ICU": 0.30, "Ward": 0.50},
                "ICU":      {"Discharge": 0.60, "Ward": 0.40},
                "Ward":     {"Discharge": 1.00},
                "Discharge":{"Discharge": 1.00},
            }

        # 5. Exit probabilities (unchanged)
        if os.path.isfile(exit_probs_csv):
            df_e = pd.read_csv(exit_probs_csv, index_col=0)
            if "P_exit" in df_e.columns:
                self.exit_probs = df_e["P_exit"].to_dict()
            else:
                self.exit_probs = {g: 1/len(df_e) for g in df_e.index}
        else:
            fallback = 1.0 / len(self.within_stay_probs)
            self.exit_probs = {g: fallback for g in self.within_stay_probs}

        # 6. LOS capping (unchanged)
        ed_rate = 1.0 / 60.0
        self.ed_max = -np.log(1.0 - los_cap_quantile) / ed_rate
        z = norm.ppf(los_cap_quantile)
        self.los_max = {}
        # Now uses the new nested self.los_params for all units? 
        # If you want to cap per unit, you can extend this loop accordingly.
        for acuity, (mu, sigma) in self.los_params["ED"].items():
            self.los_max[("ED", acuity)] = float(np.exp(mu + sigma*z))
        for acuity, (mu, sigma) in self.los_params["ICU"].items():
            self.los_max[("ICU", acuity)] = float(np.exp(mu + sigma*z))
        for acuity, (mu, sigma) in self.los_params["Ward"].items():
            self.los_max[("Ward", acuity)] = float(np.exp(mu + sigma*z))

    def sample_acuity(self) -> int:
        """
        Sample a patient severity level using the loaded empirical acuity distribution.
        Returns an integer 1–5 (1 = most severe).
        """
        return np.random.choice([1, 2, 3, 4, 5], p=self.severity_probs)



# ─────────────────────────────────────────────────────────────────────────────
# 2. Resource Classes
# ─────────────────────────────────────────────────────────────────────────────
    
class NurseGroup:
    def __init__(self, num_nurses, shift_hours=8, name=""):
        self.num_nurses = num_nurses  # Number of nurses available
        self.shift_hours = shift_hours  # Length of the shift
        self.fatigue_level = 0.0  # Initial fatigue level
        self.patients_attended = 0  # Number of patients attended in a shift
        self.name = name  # Name to identify the nurse group (Day/Night)

    def attend_patient(self):
        """Increases fatigue when a nurse attends a patient."""
        self.patients_attended += 1
        self.fatigue_level += 0.1  # Each patient attended adds to fatigue
        if self.fatigue_level > 10:
            self.fatigue_level = 10  # Maximum fatigue level

    def work(self):
        """Nurses work for a full shift (affects fatigue)."""
        self.fatigue_level += 0.2  # Increase fatigue from working a full shift
        if self.fatigue_level > 10:
            self.fatigue_level = 10  # Cap the fatigue level

    def reset(self):
        """Reset fatigue after each shift."""
        self.fatigue_level = 0  # Reset fatigue level
        self.patients_attended = 0  # Reset patients attended

    def is_available(self):
        """Check if the nurse group is available to attend patients."""
        return self.num_nurses > 0  # Can check if there are any nurses left for the shift


class CareUnit:
    """
    Represents a hospital unit (ED, ICU, or MedSurg) using real queues.
    Uses SimPy Resource to allow queuing and visibility into blocked patients.
    """
    def __init__(self, env, unit_name, capacity):
        self.env = env
        self.unit_name = unit_name
        self.capacity = capacity
        self.resource = simpy.Resource(self.env, capacity=self.capacity)

    def bed_available(self):
        return self.resource.count < self.capacity

    def beds_in_use(self):
        return self.capacity - len(self.resource.users)  

    def available_beds(self):
        return self.capacity - self.resource.count

    def occupy_bed(self):
        if self.bed_available():
            return self.resource.request()
        return None

    def free_bed(self, bed):
        bed.release(bed.users[0])  # Free up the bed after treatment

    @property
    def queue(self):
        return self.resource.queue





class Patient:
    def __init__(self, pid, severity, arrival_time):
        self.pid = pid
        self.severity = severity
        self.arrival_time = arrival_time
        self.status = "waiting"
        self.current_unit = "ED"
        self.start_treatment = None
        self.end_treatment = None
        self.discharge_time = None
        self.event_log = []


# ─────────────────────────────────────────────────────────────────────────────
# 3. HospitalActionSpace Class
# ─────────────────────────────────────────────────────────────────────────────
class HospitalActionSpace:
    """
    A two‑tier (strategic + tactical) action space using a Dict.
    
    Strategic actions (updated once per shift):
      • icu_delta      — continuous in [–2.0, +2.0] beds
      • medsurg_delta  — continuous in [–2.0, +2.0] beds
      • nurse_day_pct  — continuous in [0.0, 2.0], fraction of base nurses
      • nurse_night_pct— continuous in [0.0, 2.0]
    
    Tactical actions (updated every decision step):
      • overflow_flag    — discrete {0,1}
      • transfer_flag    — discrete {0,1}
      • triage_threshold — continuous [0.0,1.0] (e.g. percentile cutoff)
      • diversion_rate   — continuous [0.0,1.0] fraction of arrivals
    """
    def __init__(self, scenario):
        self.scenario = scenario

        # Strategic: continuous adjustments per shift
        self.strategic = spaces.Box(
            low=np.array([-2.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([ 2.0, 2.0, 2.0], dtype=np.float32),
            shape=(3,),
            dtype=np.float32
        )


        # Tactical: flags + thresholds each call
        self.tactical = spaces.Dict({
            "overflow_flag":    spaces.Discrete(2),
            "transfer_flag":    spaces.Discrete(2),
            "triage_threshold": spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
            "diversion_rate":   spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
        })

        self.action_space = spaces.Dict({
            "strategic": self.strategic,
            "tactical":  self.tactical
        })

    def sample(self):
        return self.action_space.sample()

    def decode(self, action):
        strat = action["strategic"]
        tact  = action["tactical"]

        # Compute integer bed changes
        icu_adj     = int(np.round(strat[0]))
        ms_adj      = int(np.round(strat[1]))
        # Compute nurse counts as a fraction of base
        day_nurse   = int(np.round(self.scenario.day_shift_nurses * strat[2]))
        night_nurse = int(np.round(self.scenario.night_shift_nurses * strat[3]))

        return {
            # Strategic
            "icu_adjustment":        icu_adj,
            "ms_adjustment":         ms_adj,
            "nurse_day_adjustment":  day_nurse   - self.scenario.day_shift_nurses,
            "nurse_night_adjustment":night_nurse - self.scenario.night_shift_nurses,

            # Tactical
            "overflow_flag":    bool(tact["overflow_flag"]),
            "transfer_flag":    bool(tact["transfer_flag"]),
            "triage_threshold": float(tact["triage_threshold"][0]),
            "ambulance_diversion_rate": float(tact["diversion_rate"][0])
        }

        


# ------------------------------------------------------------------
# 7.  Baseline / Heuristic Policies
# ------------------------------------------------------------------
def reactive_policy(state, scenario):
    """
    Simple queue‑aware heuristic that toggles overflow, diversion,
    and capacity deltas when ED congestion exceeds thresholds.
    """
    ed_in_use     = state[0]
    icu_in_use    = state[1]
    ward_in_use   = state[2]
    ed_queue_len  = state[3]

    icu_avail  = scenario.n_icu_beds  - icu_in_use
    ward_avail = scenario.n_medsurg_beds - ward_in_use

    return {
        # strategic
        "icu_adjustment":  +1 if icu_avail == 0 else 0,
        "ms_adjustment":   +1 if ward_avail < 2 else 0,
        "nurse_day_adjustment":  +3 if ed_queue_len > 10 else 0,
        "nurse_night_adjustment": 0,
        # tactical
        "overflow_flag":            ed_queue_len > 5,
        "transfer_flag":            ed_queue_len > 8,
        "triage_threshold":         0.4 if ed_queue_len > 8 else 0.2,
        "ambulance_diversion_rate": min(1.0, 0.05 * max(ed_queue_len - 5, 0)),
        "discharge_acceleration":   2 if ward_avail < 2 else 0,
    }

"""import random

transition_matrix = load_transition_matrix()

def sample_next_unit(current_unit: str, severity: int):
        transitions = transition_matrix.get(severity, {}).get(current_unit, {})
        if not transitions:
            return "Discharge"

        units, probs = zip(*transitions.items())
        r = random.random()
        cumulative = 0.0
        for unit, p in zip(units, probs):
            cumulative += p
            if r < cumulative:
                return unit
        return units[-1]

if __name__ == "__main__":
    next_unit = sample_next_unit("Emergency", severity=1)
    print(next_unit)"""

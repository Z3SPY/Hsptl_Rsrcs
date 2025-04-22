#!/usr/bin/env python
# -*- coding: utf-8 -*-
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

import simpy
import numpy as np
import datetime
import random
import itertools
from gymnasium import spaces
import json
import pandas as pd
import os
from collections import defaultdict


class NurseGroup:
    def __init__(self, base_nurses, name='Nurses'):
        self.base_nurses = base_nurses
        self.current_nurses = base_nurses  # This might decrease with fatigue or shift changes

    def get_fatigue_factor(self):
        # Example: if nurses are less than the base, the fatigue is higher
        if self.current_nurses < self.base_nurses:
            diff = self.base_nurses - self.current_nurses
            return 1.0 + 0.1 * diff
        return 1.0


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



#############################
# 1. Scenario
#############################
import os
import numpy as np
import pandas as pd
from scipy.stats import norm  # for log‑normal quantile

import os
import pandas as pd
import numpy as np
from scipy.stats import norm

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

        # 2. Arrival & severity profile
        self.arrival_profile = arrival_profile
        self.severity_probs  = severity_probs 

        # ────────────────────────────────────────────────
        # 3. Empirical LOS: load CSV & compute log‑normal
        # ────────────────────────────────────────────────
        df = pd.read_csv(acuity_los_csv)
        # Only keep ED (“Emergency”), ICU, Ward (“Medical Ward”)
        unit_map = {"Emergency": "ED", "ICU": "ICU", "Medical Ward": "Ward"}
        self.los_params = {u: {} for u in unit_map.values()}
        
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



#############################
# 2. Resource Classes
#############################
class CareUnit:
    """
    Represents a hospital unit (ED, ICU, or MedSurg).
    Uses a SimPy Store to manage bed tokens.
    """
    def __init__(self, env, name, capacity):
        self.env = env
        self.name = name
        self.capacity = capacity
        self.store = simpy.Store(env, capacity=capacity)
        for i in range(capacity):
            self.store.put(f"{name}_Bed_{i+1}")

    def bed_available(self):
        return len(self.store.items) > 0
    
    def beds_in_use(self):
        return self.capacity - len(self.store.items)
    
    def available_beds(self):
        # NEW: returns the number of available beds
        return len(self.store.items)

    def occupy_bed(self):
        return self.store.get()

    def free_bed(self, token):
        return self.store.put(token)
    
    def request(self):
        return self.store.get()

    def release(self, token):
        return self.store.put(token)


class NurseGroup:
    """
    Manages a group of nurses. Provides a capacity and computes a fatigue factor.
    """
    def __init__(self, base_nurses, name=''):
        self.base_nurses = base_nurses
        self.name = name
        self.current_nurses = base_nurses

    def set_nurses(self, new_count):
        self.current_nurses = max(0, new_count)

    def get_fatigue_factor(self, scenario: Scenario):
        if self.current_nurses < self.base_nurses:
            diff = self.base_nurses - self.current_nurses
            return 1.0 + (scenario.nurse_fatigue_factor * diff)
        else:
            return 1.0

#############################
# 3. Patient Class
#############################
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


#############################
# 4. HospitalFlowModel
#############################
class HospitalFlowModel:
    """
    Orchestrates the simulation:
      - Generates arrivals (time-aware)
      - Routes patients through ED → ICU/MedSurg / discharge with boarding & transfers.
      - Contains an apply_action() method to adjust operational parameters dynamically.
      - Supports snapshot_state() for debugging.
    """
    def __init__(self, scenario: Scenario, start_datetime=None, debug=False):
        self.scenario = scenario
        self.debug = debug
        self.start_datetime = start_datetime if start_datetime else datetime.datetime.now()
        self.env = simpy.Environment()

        # PATIENTS AND EVENTS
        self.event_log = []
        self.patients = []
        self.pid_counter = 0
        self.utilisation_log = []

        self.hourly_arrivals = [0] * 24              # index = hour of day
        self.daily_arrivals  = defaultdict(int)      # key = day index (0,1,2,…)
        self.arrival_scale = .5
        #self.scaled_lambda = [l * self.arrival_scale for l in scenario.arrival_profile]
        self.raw_arrivals = 0

        # REMOVE IF POSSIBOLE?    
        self.trans_probs = scenario.within_stay_probs
        self.exit_probs  = scenario.exit_probs

        self.boarding_registry = {}  # Tracks patients waiting for ICU/Ward
        self.overflow_allowed = False  # Controlled by DRL agent or scenario toggle

        # Create CareUnits:
        self.ed = CareUnit(self.env, "ED", scenario.n_ed_beds)
        self.icu = CareUnit(self.env, "ICU", scenario.n_icu_beds)
        self.medsurg = CareUnit(self.env, "MedSurg", scenario.n_medsurg_beds)

        # Nurse resource will be modeled separately (using NurseGroup with a dummy simpy.Resource for simplicity)
        self.nurse_day = NurseGroup(scenario.day_shift_nurses, name="DayNurses")
        self.nurse_night = NurseGroup(scenario.night_shift_nurses, name="NightNurses")
        # We start with the day shift resource for simplicity
        self.nurses = simpy.Resource(self.env, capacity=scenario.day_shift_nurses)

        # For MSO/planner flags used in actions
        self.overflow_allowed = False
        self.transfer_allowed = False
        self.discharge_acceleration = 0
        self.ambulance_diversion = False
        self.elective_deferral = 0
        self.icu_threshold = 0.5

        # Statistics
        self.lwbs_count = 0
        self.discharge_count = 0
        self.boarded_count = 0
        self.total_boarding_time = 0.0
        self.transferred_out = 0
        self.pid_counter = 1
        self.patients = []
        self.event_log = []

        self.icu_admits = 0           # Initialize ICU admissions counter
        self.ward_admits = 0          # Initialize Ward admissions counter

        # For background initialization
        self.background_started = False

        
        # ---- NOW seed the first utilisation record -----------
        initial_audit = self.utilisation_audit(current_time=0.0)
        self.utilisation_log.append(initial_audit)

        # Testing
        self.env.process(self.arrivals_generator())
        self.env.process(self.audit_util(interval=60))


    @property
    def nurse_fatigue(self):
        """
        Compute a composite nurse fatigue value by weighting the fatigue factors
        from day and night nurse groups. This aggregated value reflects the overall
        staffing strain in the hospital.
        """
        total_base = self.nurse_day.base_nurses + self.nurse_night.base_nurses
        fatigue_day = self.nurse_day.get_fatigue_factor(self.scenario)
        fatigue_night = self.nurse_night.get_fatigue_factor(self.scenario)
        # Compute weighted average based on the base numbers:
        weighted_fatigue = (
            (fatigue_day * self.nurse_day.base_nurses) +
            (fatigue_night * self.nurse_night.base_nurses)
        ) / total_base if total_base > 0 else 1.0
        return weighted_fatigue



    def get_severity_multiplier(self, severity):
        return {1: 0.6, 2: 0.8, 3: 1.0, 4: 1.3, 5: 1.6}.get(severity, 1.0)

    def try_transfer_or_board(self, patient, target_unit: str):
        """
        Try to admit the patient to the target unit.
        If unavailable, board the patient and wait until a bed is free.
        Returns: SimPy resource (bed) or None if transferred out.
        """
        self.log_event(patient, f"{target_unit.lower()}_request")

        if target_unit == "ICU":
            resource = self.icu
        elif target_unit == "Ward":
            resource = self.medsurg
        else:
            raise ValueError(f"Unknown unit: {target_unit}")

        # ✅ Try immediate bed assignment
        if resource.bed_available():
            bed = yield resource.occupy_bed()
            return bed

        # ❌ No bed available: board OR transfer logic
        # Boarding now waits until a bed becomes free and returns it
        bed = yield from self._handle_boarding(patient, target_unit=target_unit)

        if bed is None:
            self.transferred_out += 1
            self.log_event(patient, "transfer_out", unit=target_unit)
            return None

        return bed


    
    
    def _handle_boarding(self, patient, target_unit: str):
        """
        If no bed is available, board the patient until available.
        """
        self.boarded_count += 1
        patient.boarding_start_time = self.env.now
        self.log_event(patient, "boarding", unit=target_unit)

        resource = self.icu if target_unit == "ICU" else self.medsurg
        bed = yield resource.occupy_bed()
        patient.boarding_end_time = self.env.now

        wait_time = self.env.now - patient.boarding_start_time
        self.total_boarding_time += wait_time

        self.log_event(patient, f"end_boarding_{target_unit}")
        return bed
    
    def handle_icu_boarding(self, patient, icu_request):
        boarding_start = self.env.now
        result = yield icu_request | self.env.timeout(360)  # max 6hr wait
        waited = self.env.now - boarding_start
        patient.icu_wait_time = waited

        if icu_request not in result:
            self.transferred_out += 1
            self.log_event(patient, "transfer_due_to_boarding")
            return

        self.log_event(patient, "icu_admit")
        yield self.env.timeout(np.random.exponential(180))  # example LOS
        self.log_event(patient, "icu_discharge")






 


    # -------------------------------
    # 4A. Action Application
    # -------------------------------
    def apply_action(self, action: dict):
        """
        Apply a hierarchical action dict with both strategic and tactical controls.
        Expects keys:
        • icu_adjustment           (int): Δ in ICU beds
        • ms_adjustment            (int): Δ in MedSurg beds
        • nurse_day_adjustment     (int): Δ in day‑shift nurses
        • nurse_night_adjustment   (int): Δ in night‑shift nurses
        • overflow_flag           (bool)
        • transfer_flag           (bool)
        • triage_threshold        (float, 0.0–1.0)
        • ambulance_diversion_rate(float, 0.0–1.0)
        • discharge_acceleration  (int, optional)
        """
        # 1) Unpack everything, with safe defaults
        icu_adj     = int(action.get("icu_adjustment", 0))
        ms_adj      = int(action.get("ms_adjustment", 0))
        day_adj     = int(action.get("nurse_day_adjustment", 0))
        night_adj   = int(action.get("nurse_night_adjustment", 0))
        overflow    = bool(action.get("overflow_flag", False))
        transfer    = bool(action.get("transfer_flag", False))
        triage_thr  = float(action.get("triage_threshold", 0.0))
        diversion   = float(action.get("ambulance_diversion_rate", 0.0))
        disc_acc    = int(action.get("discharge_acceleration", 0))

        if self.debug:
            print("[ACTION APPLY] Strategic  → ICU Δ:", icu_adj,
                "MS Δ:", ms_adj,
                "| Nurses Δ:", day_adj, night_adj)
            print("[ACTION APPLY] Tactical  → overflow:", overflow,
                "transfer:", transfer,
                "triage_thr:", triage_thr,
                "diversion_rate:", diversion,
                "disc_acc:", disc_acc)

        # 2) ICU capacity
        new_icu_cap = max(0, self.scenario.n_icu_beds + icu_adj)
        diff = new_icu_cap - self.icu.capacity
        if diff > 0:
            for i in range(diff):
                self.icu.store.put(f"ICU_Bed_Extra_{i+100}")
        elif diff < 0:
            for _ in range(-diff):
                if self.icu.store.items:
                    self.icu.store.items.pop()
        self.icu.capacity = new_icu_cap

        # 3) MedSurg capacity
        new_ms_cap = max(0, self.scenario.n_medsurg_beds + ms_adj)
        diff = new_ms_cap - self.medsurg.capacity
        if diff > 0:
            for i in range(diff):
                self.medsurg.store.put(f"MedSurg_Bed_Extra_{i+100}")
        elif diff < 0:
            for _ in range(-diff):
                if self.medsurg.store.items:
                    self.medsurg.store.items.pop()
        self.medsurg.capacity = new_ms_cap

        # 4) Nurse staffing
        current_hour = int((self.env.now // 60) % 24)
        if 7 <= current_hour < 19:
            base = self.scenario.day_shift_nurses
            new_nurses = base + day_adj
        else:
            base = self.scenario.night_shift_nurses
            new_nurses = base + night_adj
        new_nurses = max(0, new_nurses)
        self.nurses = simpy.Resource(self.env, capacity=new_nurses)

        # 5) Tactical flags & thresholds
        self.overflow_allowed            = overflow
        self.transfer_allowed            = transfer
        self.triage_threshold            = triage_thr
        self.ambulance_diversion_rate    = diversion
        # If you still need a boolean ambulance_diversion flag:
        self.ambulance_diversion         = (diversion > 0.5)
        self.discharge_acceleration      = disc_acc

        if self.debug:
            print("[ACTION APPLY] New capacities → ICU:", self.icu.capacity,
                "MedSurg:", self.medsurg.capacity,
                "Nurses:", self.nurses.capacity)
            print("[ACTION APPLY] overflow_allowed:", self.overflow_allowed,
                "transfer_allowed:", self.transfer_allowed,
                "triage_threshold:", self.triage_threshold,
                "diversion_rate:", self.ambulance_diversion_rate,
                "discharge_acceleration:", self.discharge_acceleration)

    # -------------------------------
    # 4B. Arrivals Generator
    # -------------------------------
    def arrivals_generator(self):
        """
        Continuous, non‑homogeneous Poisson arrival stream.
        Samples number of arrivals each hour, schedules each within hour.
        """
        pid_counter = itertools.count(1)
        while self.env.now < self.scenario.simulation_time:
            hr = int((self.env.now // 60) % 24)
            lam = self.scenario.arrival_profile[hr]
            eff_lam = lam * (1.0 - getattr(self, "ambulance_diversion_rate", 0.0))

            n_arrivals = np.random.poisson(eff_lam)

            if self.debug:
                print(f"[ARRIVAL] t={self.env.now:.1f} min, hr={hr}, λ={lam:.2f}, eff={eff_lam:.2f}, count={n_arrivals}")

            for _ in range(n_arrivals):
                delay = np.random.uniform(0, 60 / max(n_arrivals, 1))
                yield self.env.timeout(delay)

                severity = np.random.choice([1, 2, 3, 4, 5], p=self.scenario.severity_probs)

                if severity == 1 and random.random() < getattr(self, "triage_threshold", 0.0):
                    self.lwbs_count += 1
                    dummy = Patient(pid=self.pid_counter, arrival_time=self.env.now, severity=severity)
                    self.log_event(dummy, "triage_reject")
                    self.pid_counter += 1
                    continue

                # Optional triage delay (if triage bay exists)
                if hasattr(self, "triage_bay"):
                    with self.triage_bay.request() as tri_req:
                        yield tri_req
                        yield self.env.timeout(np.random.exponential(self.triage_mean))
                        self.log_event(dummy, "triage_complete")

                # Create and admit patient
                pat = Patient(pid=self.pid_counter, arrival_time=self.env.now, severity=severity)
                self.patients.append(pat)
                self.log_event(pat, "arrival")
                self.raw_arrivals += 1
                self.hourly_arrivals[hr] += 1

                day = int(self.env.now // (60 * 24))
                self.daily_arrivals[day] += 1

                self.env.process(self.process_ed_flow(pat))
                self.pid_counter += 1

            # Advance to next hour
            yield self.env.timeout(60 - (self.env.now % 60))






    # HELPER
    def sample_los(self, unit: str, severity: int) -> float:
        """
        Draw a log‑normal LOS for the given 'unit' and patient 'severity',
        then scale by severity multiplier, nurse fatigue, and discharge acceleration (if non-ED).
        """
        # 1) Get empirical (mu, sigma)
        mu, sigma = self.scenario.los_params[unit][severity]
        base_los = np.random.lognormal(mu, sigma)

        # 2) Apply severity & nurse fatigue
        los = base_los * self.get_severity_multiplier(severity) * self.nurse_fatigue

        # 3) If not ED, also apply discharge acceleration factor
        if unit != "ED":
            accel_factor = max(0.5, 1.0 - 0.1 * self.discharge_acceleration)
            los *= accel_factor

        return los        


    #FLOWS
    def process_ed_flow(self, patient: Patient):
        """
        ED → treatment → discharge/ICU/Ward routing.
        """
        arrival_time = self.env.now
        req = self.ed.request()
        result = yield req | self.env.timeout(120)  # max wait before LWBS

        waited = self.env.now - arrival_time
        patient.wait_time = waited

        if req not in result:
            self.lwbs_count += 1
            self.patients.append(patient)  # Track LWBS too
            self.log_event(patient, "lwbs")
            return

        self.log_event(patient, "start_treatment")
        patient.start_treatment = self.env.now

        ed_los = self.sample_los("ED", patient.severity)
        yield self.env.timeout(ed_los)

        patient.end_treatment = self.env.now
        self.ed.release(req)
        self.patients.append(patient)
        self.log_event(patient, "ed_complete")

        p_discharge, p_icu, p_ward = self._get_disposition_probs(patient.severity)
        r = np.random.random()

        if r < p_discharge:
            yield from self.complete_discharge(patient, source="ed")
        elif r < p_discharge + p_icu:
            yield from self.process_icu_flow(patient)
        else:
            yield from self.process_ward_flow(patient)




    def process_icu_flow(self, patient: Patient):
        """
        ICU processing: request/board if needed, LOS, and disposition: 
        → Ward, Transfer-Out, or Discharge (probabilistically).
        """
        patient.current_unit = "ICU"
        self.log_event(patient, "icu_request")

        icu_req_time = self.env.now
        bed = yield from self.try_transfer_or_board(patient, target_unit="ICU")
        if bed is None:
            self.transferred_out += 1
            self.log_event(patient, "transfer_out", unit="ICU")
            return

        self.icu_admits += 1
        icu_wait = self.env.now - icu_req_time
        patient.icu_wait_time = icu_wait
        self.log_event(patient, "icu_admit")

        icu_los = self.sample_los("ICU", patient.severity)
        yield self.env.timeout(icu_los)

        yield self.icu.free_bed(bed)
        self.log_event(patient, "icu_discharge")

        r = np.random.random()
        if r < 0.6:
            self.log_event(patient, "icu_to_ward")
            yield from self.process_ward_flow(patient)
        elif r < 0.6 + 0.25:
            self.transferred_out += 1
            self.log_event(patient, "transfer_out", unit="ICU")
            self.log_event(patient, "icu_to_transfer")
            yield self.env.timeout(0)
        else:
            self.log_event(patient, "icu_to_discharge")
            yield from self.complete_discharge(patient, source="icu")



    def process_ward_flow(self, patient: Patient):
        """
        Ward (MedSurg) processing: request/board if needed, LOS, and disposition:
        → Discharge or Transfer-Out (probabilistically).
        """
        patient.current_unit = "Ward"
        self.log_event(patient, "ward_request")

        ward_req_time = self.env.now
        bed = yield from self.try_transfer_or_board(patient, target_unit="Ward")
        if bed is None:
            self.transferred_out += 1
            self.log_event(patient, "transfer_out", unit="Ward")
            return

        ward_wait = self.env.now - ward_req_time
        patient.ward_wait_time = ward_wait
        self.log_event(patient, "ward_admit")

        ward_los = self.sample_los("Ward", patient.severity)
        yield self.env.timeout(ward_los)

        yield self.medsurg.free_bed(bed)
        self.log_event(patient, "ward_discharge")

        r = np.random.random()
        if r < 0.8:
            self.log_event(patient, "ward_to_discharge")
            yield from self.complete_discharge(patient, source="ward")
        else:
            self.transferred_out += 1
            self.log_event(patient, "transfer_out", unit="Ward")
            self.log_event(patient, "ward_to_transfer")
            yield self.env.timeout(0)




    #STEPPING
        
    def step(self, action):
        self.apply_action(action)
        next_shift = self.env.now + self.scenario.shift_length
        self.run(until=next_shift)

        obs    = self.get_obs()
        reward = self.calculate_multiobj_reward()   # now a vector
        done   = self.env.now >= self.scenario.simulation_time
        info   = self.get_patient_flow_summary()

        return obs, reward, done, False, info

    
    def calculate_multiobj_reward(self):
        # Objective 1: Throughput (discharges)
        throughput = float(self.discharge_count)

        # Objective 2: Wait penalties (LWBS + boarded)
        wait_penalty = float(self.lwbs_count + self.boarded_count)

        # Objective 3: Staff fatigue penalty
        fatigue_penalty = self.nurse_fatigue  # composite factor

        # Objective 4: Average utilization (ED)
        utils = [u['ed_in_use'] for u in self.utilisation_log]
        avg_util = float(sum(utils) / max(len(utils), 1))

        # Return as a vector: higher is better for first & fourth, lower is better for penalties
        return np.array([throughput, -wait_penalty, -fatigue_penalty, avg_util], dtype=np.float32)





    # -------------------------------
    # Discharge 
    # ------------------------------- 
                                       
    def complete_discharge(self, patient: Patient, source="discharge"):
        """
        Mark the patient as discharged, update counters, and log the event.
        """
        patient.status = "discharged"
        patient.current_unit = "Exit"
        patient.discharge_time = self.env.now
        self.discharge_count += 1
        self.log_event(patient, f"{source}_discharge")
        # Yield a 0-time event so it becomes a generator
        yield self.env.timeout(0)





    def _get_disposition_probs(self, severity):
        return {
            1: (0.95, 0.02, 0.03),  # 95% discharged
            2: (0.6, 0.15, 0.25),   # some go to ward
            3: (0.3, 0.4, 0.3),     # mixed outcomes
            4: (0.1, 0.7, 0.2),     # mostly ICU
            5: (0.05, 0.9, 0.05)    # ICU-critical
        }.get(severity, (0.3, 0.4, 0.3))  # fallback




    # -------------------------------
    # Log 
    # -------------------------------
    def log_event(self, patient, event, **kwargs):
        """
        Logs a structured event for a patient with optional extra fields.
        """
        log = {
            "time": self.env.now,
            "pid": patient.pid,
            "event": event,
            "severity": patient.severity,
            "unit": getattr(patient, "current_unit", None)
        }
        log.update(kwargs)  # <-- Accept extra keyword arguments

        if hasattr(patient, "event_log"):
            patient.event_log.append(log)

        if self.debug:
            print(log)

        self.event_log.append(log)



    def get_obs(self):
        now = self.env.now

        # Bed occupancy and queue lengths
        ed_in_use   = self.ed.beds_in_use()
        icu_in_use  = self.icu.beds_in_use()
        ward_in_use = self.medsurg.beds_in_use()
        ed_queue    = self.scenario.n_ed_beds   - self.ed.available_beds()
        icu_queue   = self.scenario.n_icu_beds  - self.icu.available_beds()
        ward_queue  = self.scenario.n_medsurg_beds - self.medsurg.available_beds()
        ed_queue_len  = len(self.ed.store.get_queue)
        icu_queue_len = len(self.icu.store.get_queue)
        ward_queue_len= len(self.medsurg.store.get_queue)


        # Recent arrival rate (last 60 min)
        arrivals_last_hour = sum(
            1 for e in self.event_log
            if e["event"] == "arrival" and now - e["time"] <= 60
        )
        avg_arrival_rate = arrivals_last_hour / 60.0  # per minute

        # Rolling utilization (mean ED occupancy over last N audits)
        recent_utils = [
            u['ed_in_use'] for u in self.utilisation_log
            if now - u['time'] <= 120  # last 2 h
        ]
        avg_ed_util = sum(recent_utils) / max(len(recent_utils), 1)

        # Time‑of‑day feature
        hour_of_day = (now // 60) % 24

        return np.array([
            ed_in_use, icu_in_use, ward_in_use,
            ed_queue_len, icu_queue_len, ward_queue_len,
            avg_arrival_rate, avg_ed_util, hour_of_day
        ], dtype=np.float32)


    
    def get_patient_flow_summary(self):
        """
        Summarize patient flow metrics:
          - Counts: total, discharged, boarded, LWBS, transferred
          - Average and percentile wait times in each unit (ED, ICU, Ward)
          - Average boarding time and total length of stay
        """
        summary = {
            "n_total": len(self.patients),
            "n_discharged": self.discharge_count,
            "n_transferred": self.transferred_out,
            "n_boarded": self.boarded_count,
            "n_lwbs": self.lwbs_count,
            "avg_total_time": 0.0,
            "avg_boarding_time": 0.0,
            # unit-specific waits
            "avg_ed_wait": 0.0,
            "avg_icu_wait": 0.0,
            "avg_ward_wait": 0.0,
            # overall
            "avg_wait_overall": 0.0,
            "median_ed_wait": 0.0,
            "p95_ed_wait": 0.0,
        }

        durations = []
        boarding_times = []
        ed_waits = []
        icu_waits = []
        ward_waits = []

        for p in self.patients:
            # total time in system
            if getattr(p, 'discharge_time', None) and p.arrival_time is not None:
                durations.append(p.discharge_time - p.arrival_time)

            # boarding durations
            if hasattr(p, 'boarding_start_time') and hasattr(p, 'boarding_end_time'):
                boarding_times.append(p.boarding_end_time - p.boarding_start_time)

            # ED wait
            if hasattr(p, 'wait_time'):
                ed_waits.append(p.wait_time)

            if hasattr(p, 'ward_wait_time'):
                ward_waits.append(p.ward_wait_time)
            
            if hasattr(p, 'icu_wait_time'):
                icu_waits.append(p.icu_wait_time)



        # aggregate
        if durations:
            summary["avg_total_time"] = float(np.mean(durations))
        if boarding_times:
            summary["avg_boarding_time"] = float(np.mean(boarding_times))
        if ed_waits:
            summary["avg_ed_wait"] = float(np.mean(ed_waits))
            summary["median_ed_wait"] = float(np.median(ed_waits))
            summary["p95_ed_wait"] = float(np.percentile(ed_waits, 95))
        if icu_waits:
            summary["avg_icu_wait"] = float(np.mean(icu_waits))
        if ward_waits:
            summary["avg_ward_wait"] = float(np.mean(ward_waits))

        # overall wait across all queues
        all_waits = ed_waits + icu_waits + ward_waits
        if all_waits:
            summary["avg_wait_overall"] = float(np.mean(all_waits))

        # print arrival counts
        print("=== Arrival Counts ===")
        print(f"  raw_arrivals   : {self.raw_arrivals}")
        print(f"  admitted (n_total) : {len(self.patients)}")
        print(f"  discharged     : {self.discharge_count}")
        print(f"  left w/o svc   : {self.lwbs_count}")
        print(f"  transferred    : {self.transferred_out}")

        return summary 





    # -------------------------------
    # 4C. Audit and Snapshot Functions
    # -------------------------------
                
    def utilisation_audit(self, current_time=None):
        return {
            'time': current_time or self.env.now,
            'ed_in_use': self.ed.beds_in_use(),
            'icu_available': self.icu.capacity - len(self.icu.store.items),
            'medsurg_available': self.medsurg.capacity - len(self.medsurg.store.items),
            'nurses_available': self.nurses.capacity - len(self.nurses.queue) - self.nurses.count,
            'discharges': self.discharge_count,
            'lwbs': self.lwbs_count,
            'boarded': self.boarded_count,
            'transfers': self.transferred_out,
        }


    def audit_util(self, interval=60):
        while True:
            record = self.utilisation_audit()
            self.utilisation_log.append(record)
            yield self.env.timeout(interval)

   
    
    def snapshot_state(self):
        """
        Returns a snapshot of the current simulation state to be used for front-end visualization 
        or by planners. Includes key metrics and logs.
        """
        return {
            "time": self.env.now,
            "patients": [vars(p) for p in self.patients],
            "ed_in_use": self.ed.beds_in_use(),  # Number of ED beds in use
            "icu_occupancy": self.icu.beds_in_use(),  # ICU beds in use
            "ward_occupancy": self.medsurg.capacity - len(self.medsurg.store.items) if hasattr(self.medsurg, "store") else None,
            "nurse_fatigue": self.nurse_fatigue,  # Assuming composite property defined elsewhere
            "discharge_count": self.discharge_count,
            "lwbs": self.lwbs_count,
            "boarded": self.boarded_count,
            "transfers": self.transferred_out,
            "event_log": self.event_log[-100:]
        }
    
    

    # -------------------------------
    # RESET
    # -------------------------------
                

    def reset(self):
        """
        Reinitializes the entire model simulation state.
        Must be called at the start of every episode.
        """
        self.__init__(self.scenario, start_datetime=self.start_datetime, debug=self.debug)
        return self.get_obs()



    # -------------------------------
    # 4D. Run Function
    # -------------------------------
    def run(self, until):
        horizon = until if until is not None else self.scenario.simulation_time
        self.env.run(until=horizon)

#############################
# 5. HospitalActionSpace Class
#############################
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
            low=np.array([-2.0, -2.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([ 2.0,  2.0, 2.0, 2.0], dtype=np.float32),
            shape=(4,),
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

# ------------------------------------------------------------------
# Debug runner – heuristic policy vs. no‑control baseline
# ------------------------------------------------------------------
if __name__ == "__main__":
    from pprint import pprint
    import datetime, time

    print("=== Hospital Simulation Debug Runner (30‑day test) ===")

    scenario = Scenario(
        simulation_time    = 30 * 24 * 60,
        n_ed_beds          = 5,
        n_icu_beds         = 5,
        n_medsurg_beds     = 10,
        day_shift_nurses   = 10,
        night_shift_nurses = 5,
        arrival_profile    = LAMBDA_HOUR,
        severity_probs     = SEVERITY_PROBS,
    )

    model = HospitalFlowModel(scenario, start_datetime=datetime.datetime.now(), debug=True)

    # 3) Run the heuristic policy, one shift at a time
    shift_len = scenario.shift_length     # 8 h ⇒ 480 min
    while model.env.now < scenario.simulation_time:
        # a) Observe
        obs = model.get_obs()

        # b) Choose action (reactive heuristic)
        action = reactive_policy(obs, scenario)

        # c) Apply action
        model.apply_action(action)

        # d) Advance one shift
        model.run(until = model.env.now + shift_len)
        

    # 4) Snapshot results
    snap = model.snapshot_state()
    print(f"\n=== Simulation Snapshot at t = {snap['time']:.0f} min ===")
    print("ED in use:", snap['ed_in_use'])
    print("ICU in use:", snap['icu_occupancy'])
    print("Ward in use:", snap['ward_occupancy'])
    print("Nurse‑fatigue index:", snap['nurse_fatigue'])
    print("Discharged:", snap['discharge_count'],
          "| LWBS:", snap['lwbs'],
          "| Boarded:", snap['boarded'],
          "| Transfers:", snap['transfers'])

    print("\n=== Recent Event Log (last 10) ===")
    for event in snap['event_log'][-10:]:
        pprint(event)

    # 5) Observation & flow summary
    obs_now = model.get_obs()
    print("\n=== Final Observation Vector ===")
    pprint(obs_now)

    flow_summary = model.get_patient_flow_summary()
    print("\n=== Patient‑flow Summary ===")
    pprint(flow_summary)

    # 6) Reset sanity‑check
    print("\n=== Reset Test ===")
    print("Obs before reset:", obs_now)
    obs_reset = model.reset()
    print("Obs after  reset:", obs_reset)

    # Pause so you can read the terminal
    time.sleep(2)

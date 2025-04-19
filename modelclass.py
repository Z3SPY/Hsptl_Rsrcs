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
SEVERITY_PROBS = _js["class_probs"]        # [p_G1‑2, p_G3, p_G4‑5]
# Map severity 1→G1‑2, 2→G3, 3→G4‑5
LOS_PARAMS     = {
    1: tuple(_js["lognorm_params"]["G1-2"]),
    2: tuple(_js["lognorm_params"]["G3"]),
    3: tuple(_js["lognorm_params"]["G4-5"])
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
        arrival_profile: list[float] = None,   # λ per hour
        severity_probs:  list[float] = None,   # P(severity=1,2,3)
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
        self.arrival_profile = arrival_profile or [1.0] * 24
        self.severity_probs  = severity_probs  or [1/3, 1/3, 1/3]

        # ────────────────────────────────────────────────
        # 3. Empirical LOS: load CSV & compute log‑normal
        # ────────────────────────────────────────────────
        df = pd.read_csv(acuity_los_csv)
        # Only keep ED (“Emergency”), ICU, Ward (“Medical Ward”)
        unit_map = {"Emergency":"ED", "ICU":"ICU", "Medical Ward":"Ward"}
        df = df[df["unit_group"].isin(unit_map)]
        
        # Nested dict: self.los_params[unit][acuity] = (mu, sigma)
        self.los_params = {"ED":{}, "ICU":{}, "Ward":{}}
        for _, row in df.iterrows():
            unit      = unit_map[row["unit_group"]]
            acuity    = int(row["acuity"])
            mean_min  = row["mean_min"]
            var_min   = row["var_min"]
            # log-normal formulas so that mean = mean_min, var = var_min
            sigma2 = np.log(1 + var_min / mean_min**2)
            mu     = np.log(mean_min / np.sqrt(1 + var_min / mean_min**2))
            sigma  = np.sqrt(sigma2)
            self.los_params[unit][acuity] = (mu, sigma)

        # Ensure every acuity 1–3 exists for each unit (fallback)
        default_mu, default_sigma = np.log(120), 1.0
        for u in ("ED","ICU","Ward"):
            for a in (1,2,3):
                self.los_params[u].setdefault(a, (default_mu, default_sigma))

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

        self.event_log = []
        self.patients = []
        self.pid_counter = 0
        self.utilisation_log = []

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
        return {1: 0.8, 2: 1.0, 3: 1.5}.get(severity, 1.0)

    def try_transfer_or_board(self, patient, target_unit: str):
        """
        Try to admit the patient to the target unit.
        If unavailable, either board the patient or transfer out.
        Returns: SimPy resource (bed) or None if transferred.
        """
        self.log_event(patient, f"{target_unit.lower()}_request")

        if target_unit == "ICU":
            resource = self.icu
        elif target_unit == "Ward":
            resource = self.medsurg
        else:
            raise ValueError(f"Unknown unit: {target_unit}")

        if not resource.bed_available():
            yield from self._handle_boarding(patient, target_unit=target_unit)
            return None  # Transferred out or abandoned

        bed = yield resource.occupy_bed()
        return bed
    
    
    def _handle_boarding(self, patient, target_unit: str):
        """
        If no bed is available, board the patient until available, or escalate.
        """
        self.log_event(patient, f"start_boarding_{target_unit}")
        self.boarded_count += 1

        wait_start = self.env.now
        resource = self.icu if target_unit == "ICU" else self.medsurg
        bed = yield resource.occupy_bed()
        wait_time = self.env.now - wait_start

        boarding_time = self.env.now - patient.boarding_start_time
        self.total_boarding_time += boarding_time

        self.log_event(patient, f"end_boarding_{target_unit}")
        return bed



 


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
        pid_counter = itertools.count(1)
        for _ in pid_counter:
            if self.env.now >= self.scenario.simulation_time:
                break

            # 1) Sample inter‑arrival from empirical λ-hour
            hr      = int((self.env.now // 60) % 24)
            lam     = self.scenario.arrival_profile[hr]
            eff_lam = lam * (1.0 - getattr(self, "ambulance_diversion_rate", 0.0))
            iat     = np.random.exponential(60.0 / max(eff_lam, 1e-6))
            if self.debug:
                print(f"[ARRIVAL] hour={hr}, λ={lam:.2f}, eff={eff_lam:.2f}, iat={iat:.1f}")

            yield self.env.timeout(iat)

            # 2) Sample acuity from empirical distribution
            severity = np.random.choice([1, 2, 3], p=self.scenario.severity_probs)

            # 3) Triage gate (low‐acuity reject)
            if severity == 1 and random.random() < getattr(self, "triage_threshold", 0.0):
                self.lwbs_count += 1
                self.event_log.append({
                    'time':       self.env.now,
                    'patient_id': self.pid_counter,
                    'event':      'triage_reject',
                    'severity':   severity
                })
                self.pid_counter += 1
                continue

            # 4) Admit patient to ED flow
            pat = Patient(pid=self.pid_counter,
                          arrival_time=self.env.now,
                          severity=severity)
            self.pid_counter += 1
            self.patients.append(pat)
            self.env.process(self.process_ed_flow(pat))
            self.event_log.append({
                'time':       self.env.now,
                'patient_id': pat.pid,
                'event':      'arrival',
                'severity':   severity
            })


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
        # 1) Attempt to get an ED bed (timeout ⇒ LWBS)
        req = self.ed.request()
        result = yield req | self.env.timeout(120)
        if req not in result:
            self.lwbs_count += 1
            self.log_event(patient, "lwbs")
            return

        # 2) Admit & log start of treatment
        self.log_event(patient, "start_treatment")
        patient.start_treatment = self.env.now

        # 3) Sample ED LOS (empirical)
        ed_los = self.sample_los("ED", patient.severity)
        yield self.env.timeout(ed_los)

        # 4) Complete treatment
        patient.end_treatment = self.env.now
        self.ed.release(req)
        self.log_event(patient, "ed_complete")

        # 5) Route onward
        p_discharge, p_icu, _ = self._get_disposition_probs(patient.severity)
        if np.random.random() < p_discharge:
            yield from self.complete_discharge(patient, source="ed")
        elif np.random.random() < (p_discharge + p_icu):
            yield from self.process_icu_flow(patient)
        else:
            yield from self.process_ward_flow(patient)



    def process_icu_flow(self, patient: Patient):
        """
        ICU processing: request/board if needed, then LOS and step‐down to Ward.
        """
        patient.current_unit = "ICU"
        self.log_event(patient, "icu_request")

        # 1) Admit or board out
        bed = yield from self.try_transfer_or_board(patient, target_unit="ICU")
        if bed is None:
            return  # transferred out

        # 2) Log admission
        self.icu_admits += 1
        self.log_event(patient, "icu_admit")

        # 3) Sample ICU LOS
        icu_los = self.sample_los("ICU", patient.severity)
        yield self.env.timeout(icu_los)

        # 4) Discharge from ICU
        yield self.icu.free_bed(bed)
        self.log_event(patient, "icu_discharge")

        # 5) Step‐down to Ward
        yield from self.process_ward_flow(patient)



    def process_ward_flow(self, patient: Patient):
        """
        MedSurg/Ward processing: request/board, LOS, then discharge.
        """
        patient.current_unit = "Ward"
        self.log_event(patient, "ward_request")

        # 1) Admit or board out
        bed = yield from self.try_transfer_or_board(patient, target_unit="Ward")
        if bed is None:
            return  # transferred out

        # 2) Log admission
        self.ward_admits += 1
        self.log_event(patient, "ward_admit")

        # 3) Sample Ward LOS
        ward_los = self.sample_los("Ward", patient.severity)
        yield self.env.timeout(ward_los)

        # 4) Discharge
        yield self.medsurg.free_bed(bed)
        yield from self.complete_discharge(patient, source="ward")



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
        """
        Return (discharge_prob, icu_prob, ward_prob) for a given severity.
        Must sum to ~1.0
        Example logic:
        severity=1 => mostly discharge, rarely ward or icu
        severity=2 => moderate chance discharge, some ward, some icu
        severity=3 => bigger chance ICU or ward, smaller discharge
        """
        if severity == 1:
            # Low acuity
            return (0.85, 0.00, 0.15)
        elif severity == 2:
            return (0.60, 0.15, 0.25)
        else:  # severity=3
            return (0.30, 0.50, 0.20)



    # -------------------------------
    # Log 
    # -------------------------------
    def log_event(self, patient, event):
        """Log an event with timestamp, event name, patient id, severity, and current unit."""
        log = {
            "time": round(self.env.now, 2),
            "event": event,
            "pid": patient.pid,
            "severity": patient.severity,
            "unit": patient.current_unit
        }
        patient.event_log.append(log)
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
        summary = {
            "n_total": len(self.patients),
            "n_discharged": self.discharge_count,
            "n_transferred": self.transferred_out,
            "n_boarded": self.boarded_count,
            "n_lwbs": self.lwbs_count,
            "avg_total_time": 0.0,
            "avg_boarding_time": 0.0,
            "avg_wait_time": 0.0
        }

        durations, boarding_times, wait_times = [], [], []

        for p in self.patients:
            if hasattr(p, "discharge_time") and p.discharge_time and p.arrival_time:
                durations.append(p.discharge_time - p.arrival_time)

            arrival, treatment_start, boarding_start = None, None, None
            for e in p.event_log:
                if e["event"] == "arrival":
                    arrival = e["time"]
                elif e["event"] == "start_treatment":
                    treatment_start = e["time"]
                elif e["event"] == "boarding":
                    boarding_start = e["time"]
                elif e["event"] == "transfer" and boarding_start is not None:
                    boarding_times.append(e["time"] - boarding_start)

            if arrival and treatment_start:
                wait_times.append(treatment_start - arrival)

        if durations:
            summary["avg_total_time"] = np.mean(durations)
        if boarding_times:
            summary["avg_boarding_time"] = np.mean(boarding_times)
        if wait_times:
            summary["avg_wait_time"] = np.mean(wait_times)

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
        if not self.background_started:
            self.env.process(self.arrivals_generator())
            self.env.process(self.audit_util(interval=60))
            self.background_started = True
        self.env.run(until=until)

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

    print("=== Hospital Simulation Debug Runner (7‑day test) ===")

    scenario = Scenario(
        simulation_time    = 60 * 24 * 7,
        n_ed_beds          = 15,
        n_icu_beds         = 4,
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

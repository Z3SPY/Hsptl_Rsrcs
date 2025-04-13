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


#############################
# 1. Scenario
#############################
class Scenario:
    def __init__(self,
                 simulation_time=60*24*7,  # total simulation time in minutes (e.g., 7 days)
                 shift_length=8*60,       # 8-hour shift in minutes
                 n_ed_beds=15,
                 n_icu_beds=4,
                 n_medsurg_beds=10,
                 day_shift_nurses=10,
                 night_shift_nurses=5,
                 ed_treatment_mean=2.0,   # mean treatment time in ED in hours
                 icu_los_mean=2.0,        # mean ICU length-of-stay in days
                 ms_los_mean=3.5,         # mean ward length-of-stay in days
                 nurse_fatigue_factor=0.1, 
                 arrival_profile=None):
        self.simulation_time = simulation_time
        self.shift_length = shift_length
        self.n_ed_beds = n_ed_beds
        self.n_icu_beds = n_icu_beds
        self.n_medsurg_beds = n_medsurg_beds
        self.day_shift_nurses = day_shift_nurses
        self.night_shift_nurses = night_shift_nurses
        self.ed_treatment_mean = ed_treatment_mean  # in hours
        self.icu_los_mean = icu_los_mean * 24 * 60    # convert days to minutes
        self.ms_los_mean = ms_los_mean * 24 * 60
        self.nurse_fatigue_factor = nurse_fatigue_factor
        # A 24-element profile for hourly arrivals (patients per hour)
        self.arrival_profile = arrival_profile or [1, 1, 2, 2, 3, 4, 6, 8, 10, 10, 9, 9,
                                                     8, 8, 7, 6, 5, 5, 4, 3, 3, 2, 2, 1]

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
    """
    Tracks an individual patient’s flow and times.
    """
    def __init__(self, pid, arrival_time, severity):
        self.pid = pid
        self.arrival_time = arrival_time
        self.severity = severity
        self.state = "WAITING"   # initial state
        self.start_treatment = None
        self.end_treatment = None
        self.discharge_time = None

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
        self.discharge_count = 0
        self.lwbs_count = 0
        self.icu_admits = 0
        self.ward_admits = 0
        self.transferred_out = 0
        self.boarded_count = 0

        # For background initialization
        self.background_started = False

    # -------------------------------
    # 4A. Action Application
    # -------------------------------
    def apply_action(self, action_vector):
        """
        Decode an 8-dim action vector and apply adjustments.
        Action vector indices:
          0: ICU bed adjustment (integer, e.g. -2..+2)
          1: MedSurg bed adjustment (integer, e.g. -2..+2)
          2: Day nurse adjustment (integer, e.g. -5..+5)
          3: Night nurse adjustment (integer, e.g. -3..+3)
          4: Overflow flag (0 or 1)
          5: Transfer flag (0 or 1)
          6: Discharge acceleration (0-3)
          7: Ambulance diversion flag (0 or 1)
        """
        if len(action_vector) < 8:
            raise ValueError("Action vector must have at least 8 dimensions.")
        decoded = {
            'icu_adjustment': int(action_vector[0]),
            'ms_adjustment': int(action_vector[1]),
            'nurse_day_adjustment': int(action_vector[2]),
            'nurse_night_adjustment': int(action_vector[3]),
            'overflow_flag': bool(action_vector[4]),
            'transfer_flag': bool(action_vector[5]),
            'discharge_acceleration': int(action_vector[6]),
            'ambulance_diversion': bool(action_vector[7])
        }
        if self.debug:
            print("[ACTION APPLY] Decoded action:", decoded)
        
        # Apply adjustments for ICU:
        new_icu_capacity = self.scenario.n_icu_beds + decoded['icu_adjustment']
        new_icu_capacity = max(0, new_icu_capacity)
        diff_icu = new_icu_capacity - self.icu.capacity
        if diff_icu > 0:
            for i in range(diff_icu):
                self.icu.store.put(f"ICU_Bed_Extra_{i+100}")
        elif diff_icu < 0:
            remove_count = abs(diff_icu)
            for _ in range(remove_count):
                if len(self.icu.store.items) > 0:
                    self.icu.store.get_nowait()
        self.icu.capacity = new_icu_capacity

        # MedSurg adjustments:
        new_ms_capacity = self.scenario.n_medsurg_beds + decoded['ms_adjustment']
        new_ms_capacity = max(0, new_ms_capacity)
        diff_ms = new_ms_capacity - self.medsurg.capacity
        if diff_ms > 0:
            for i in range(diff_ms):
                self.medsurg.store.put(f"MedSurg_Bed_Extra_{i+100}")
        elif diff_ms < 0:
            remove_count = abs(diff_ms)
            for _ in range(remove_count):
                if len(self.medsurg.store.items) > 0:
                    self.medsurg.store.get_nowait()
        self.medsurg.capacity = new_ms_capacity

        # Adjust nurse resource.
        current_hour = int((self.env.now // 60) % 24)
        if 7 <= current_hour < 19:
            new_nurse = self.scenario.day_shift_nurses + decoded['nurse_day_adjustment']
        else:
            new_nurse = self.scenario.night_shift_nurses + decoded['nurse_night_adjustment']
        new_nurse = max(0, new_nurse)
        self.nurses = simpy.Resource(self.env, capacity=new_nurse)

        # Set remaining flags:
        self.overflow_allowed = decoded['overflow_flag']
        self.transfer_allowed = decoded['transfer_flag']
        self.discharge_acceleration = decoded['discharge_acceleration']
        self.ambulance_diversion = decoded['ambulance_diversion']
        
        # Log application:
        if self.debug:
            print("[ACTION APPLY] Updated ICU capacity:", self.icu.capacity)
            print("[ACTION APPLY] Updated MedSurg capacity:", self.medsurg.capacity)
            print("[ACTION APPLY] Updated Nurse capacity:", self.nurses.capacity)

    # -------------------------------
    # 4B. Arrivals Generator
    # -------------------------------
    def arrivals_generator(self):
        for pid in itertools.count(1):
            if self.env.now >= self.scenario.simulation_time:
                break
            current_hour = int((self.env.now // 60) % 24)
            base_rate = self.scenario.arrival_profile[current_hour]
            effective_rate = base_rate * (0.5 if self.ambulance_diversion else 1.0)
            mean_iat = 60.0 / effective_rate if effective_rate > 0 else 9999
            iat = np.random.exponential(mean_iat)
            if self.debug:
                print(f"[ARRIVAL] Hour {current_hour}, rate={base_rate} ({'diversion' if self.ambulance_diversion else 'normal'}), IAT={iat:.2f} min")
            yield self.env.timeout(iat)
            severity = np.random.randint(1, 4)
            pat = Patient(pid=self.pid_counter, arrival_time=self.env.now, severity=severity)
            self.pid_counter += 1
            self.patients.append(pat)
            # For now, we simulate a direct discharge from ED after ED treatment time.
            self.env.process(self.process_ed_flow(pat))
            self.event_log.append({
                'time': self.env.now,
                'patient_id': pat.pid,
                'event': 'arrival',
                'severity': severity
            })

    def process_ed_flow(self, patient: Patient):
        """
        ED processing that differentiates severity 1, 2, and 3:
        - severity 3 => Attempt ICU if bed is free, else board/transfer
        - severity 1 or 2 => typical 70% discharge, 30% ward
        """
        # Predefine severity_factor to avoid errors
        severity_factor = {1: 0.8, 2: 1.0, 3: 1.5}

        # Request an ED bed (simple resource request)
        req = self.ed.request()
        result = yield req | self.env.timeout(120)
        if req not in result:
            self.lwbs_count += 1
            self.event_log.append({
                'time': self.env.now,
                'patient_id': patient.pid,
                'event': 'lwbs'
            })
            return

        # Got an ED bed; simulate ED treatment:
        patient.start_treatment = self.env.now

        base_treatment = np.random.exponential(self.scenario.ed_treatment_mean * 60)
        adjusted_treatment = base_treatment * severity_factor.get(patient.severity, 1.0)
        treatment_time = adjusted_treatment * max(0, 1 - 0.1 * self.discharge_acceleration)

        yield self.env.timeout(treatment_time)
        patient.end_treatment = self.env.now

        # Release ED bed:
        self.ed.release(req)

        # Decision logic:
        if patient.severity == 3:
            # Attempt ICU
            if self.icu.bed_available():
                bed = yield self.icu.occupy_bed()
                self.icu_admits += 1
                # ICU LOS
                icu_time = np.random.exponential(self.scenario.icu_los_mean) * severity_factor[3]
                icu_time *= max(0, 1 - 0.1 * self.discharge_acceleration)
                yield self.env.timeout(icu_time)
                yield self.icu.free_bed(bed)

                # Finally discharge
                patient.discharge_time = self.env.now
                self.discharge_count += 1
                self.event_log.append({
                    'time': self.env.now,
                    'patient_id': patient.pid,
                    'event': 'icu_discharge'
                })
            else:
                # Board or Transfer
                self.boarded_count += 1
                self.event_log.append({
                    'time': self.env.now,
                    'patient_id': patient.pid,
                    'event': 'boarding'
                })
                # For demonstration, board X minutes, then forced transfer
                board_time = 60 * severity_factor[3]  # e.g. 90 if severity 3
                yield self.env.timeout(board_time)

                self.transferred_out += 1
                self.event_log.append({
                    'time': self.env.now,
                    'patient_id': patient.pid,
                    'event': 'transfer_out'
                })
        else:
            # severity 1 or 2 => typical 70% discharge, 30% ward
            if np.random.random() < 0.7:
                discharge_delay = np.random.exponential(15 * severity_factor.get(patient.severity, 1.0))
                patient.discharge_time = self.env.now + discharge_delay
                self.discharge_count += 1
                self.event_log.append({
                    'time': self.env.now,
                    'patient_id': patient.pid,
                    'event': 'discharge'
                })
            else:
                if self.medsurg.bed_available():
                    bed = yield self.medsurg.store.get()
                    self.ward_admits += 1

                    ward_time = np.random.exponential(self.scenario.ms_los_mean)
                    ward_time *= severity_factor.get(patient.severity, 1.0)
                    ward_time *= max(0, (1 - 0.1 * self.discharge_acceleration))
                    yield self.env.timeout(ward_time)

                    yield self.medsurg.store.put(bed)
                    patient.discharge_time = self.env.now
                    self.discharge_count += 1
                    self.event_log.append({
                        'time': self.env.now,
                        'patient_id': patient.pid,
                        'event': 'ward_discharge'
                    })
                else:
                    self.boarded_count += 1
                    self.event_log.append({
                        'time': self.env.now,
                        'patient_id': patient.pid,
                        'event': 'boarding'
                    })
                    yield self.env.timeout(60)
                    self.transferred_out += 1
                    self.event_log.append({
                        'time': self.env.now,
                        'patient_id': patient.pid,
                        'event': 'transfer'
                    })


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


    def _handle_boarding(self, patient: Patient, target_unit_name: str):
        """
        For demonstration: patient boards for some time scaled by severity,
        then is 'transferred_out'. In a real flow, they'd wait for bed frees.
        """
        severity_factor = {1: 0.8, 2: 1.0, 3: 1.5}
        self.boarded_count += 1

        self.event_log.append({
            'time': self.env.now,
            'patient_id': patient.pid,
            'event': 'boarding',
            'target_unit': target_unit_name
        })

        board_time = 60.0 * severity_factor.get(patient.severity, 1.0)  # e.g. 60-90 min
        yield self.env.timeout(board_time)

        self.transferred_out += 1
        self.event_log.append({
            'time': self.env.now,
            'patient_id': patient.pid,
            'event': 'transfer'
        })




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
        snap = {
            'time': self.env.now,
            'ed_in_use': self.ed.beds_in_use(),
            'icu_available': len(self.icu.store.items),
            'medsurg_available': len(self.medsurg.store.items),
            'nurses_available': self.nurses.capacity - self.nurses.count,
            'discharge_count': self.discharge_count,
            'lwbs_count': self.lwbs_count,
            'boarded_count': self.boarded_count,
            'transferred_out': self.transferred_out
        }
        return snap

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
    Defines an 8-dimensional action space.
    Dimensions:
      0: ICU bed adjustment (-2 to +2)
      1: MedSurg bed adjustment (-2 to +2)
      2: Day nurse adjustment (-5 to +5)
      3: Night nurse adjustment (-3 to +3)
      4: Overflow flag (0 or 1)
      5: Transfer flag (0 or 1)
      6: Discharge acceleration (0–3)
      7: Ambulance diversion flag (0 or 1)
    """
    def __init__(self):
        icu_range = (-2,2)
        ms_range = (-2,2)
        nurse_day_range = (-5,5)
        nurse_night_range = (-3,3)
        self.icu_dim = icu_range[1] - icu_range[0] + 1
        self.ms_dim = ms_range[1] - ms_range[0] + 1
        self.nurse_day_dim = nurse_day_range[1] - nurse_day_range[0] + 1
        self.nurse_night_dim = nurse_night_range[1] - nurse_night_range[0] + 1
        self.icu_offset = -icu_range[0]
        self.ms_offset = -ms_range[0]
        self.nurse_day_offset = -nurse_day_range[0]
        self.nurse_night_offset = -nurse_night_range[0]
        self.action_space = spaces.MultiDiscrete([
            self.icu_dim,
            self.ms_dim,
            self.nurse_day_dim,
            self.nurse_night_dim,
            2,  # overflow flag
            2,  # transfer flag
            4,  # discharge acceleration (0-3)
            2   # ambulance diversion flag
        ])

    def sample(self):
        return self.action_space.sample()

    def decode(self, action):
        return {
            'icu_adjustment': int(action[0]) - self.icu_offset,
            'ms_adjustment': int(action[1]) - self.ms_offset,
            'nurse_day_adjustment': int(action[2]) - self.nurse_day_offset,
            'nurse_night_adjustment': int(action[3]) - self.nurse_night_offset,
            'overflow_flag': bool(action[4]),
            'transfer_flag': bool(action[5]),
            'discharge_acceleration': int(action[6]),
            'ambulance_diversion': bool(action[7])
        }

#############################
# 6. Main Runner for Debugging
#############################
if __name__ == "__main__":
    # For debugging purposes:
    print("=== Hospital Simulation Debug Runner ===")
    # Create a scenario with a shorter simulation time (e.g., 10 hours)
    scenario = Scenario(simulation_time=60*10, n_icu_beds=4, n_medsurg_beds=10,
                        day_shift_nurses=10, night_shift_nurses=5, ed_treatment_mean=2.0)
    # Create the HospitalFlowModel instance
    model = HospitalFlowModel(scenario, start_datetime=datetime.datetime.now(), debug=True)
    
    # Create the action space object and sample an action.
    action_space_obj = HospitalActionSpace()
    sample_action = action_space_obj.sample()
    decoded_action = action_space_obj.decode(sample_action)
    print("Sample raw action vector:", sample_action)
    print("Decoded action:", decoded_action)
    
    # Apply the action to the model.
    model.apply_action(sample_action)
    
    # Run the model until 3 days (4320 minutes) for testing.
    model.run(until=60*24*3)
    
    # Print snapshot of the system state.
    snap = model.snapshot_state()
    print("=== Simulation Snapshot at t =", snap['time'], "minutes ===")
    print("ED in use:", snap['ed_in_use'])
    print("ICU available:", snap['icu_available'])
    print("MedSurg available:", snap['medsurg_available'])
    print("Nurses available:", snap['nurses_available'])
    print("Discharge count:", snap['discharge_count'])
    print("LWBS count:", snap['lwbs_count'])
    print("Boarded count:", snap['boarded_count'])
    print("Transferred out:", snap['transferred_out'])
    print("=== Event Log Summary (first 10 events) ===")
    for event in model.event_log[:10]:
        print(event)

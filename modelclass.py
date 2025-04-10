#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Refactored Hospital Ward Simulation (PreemptiveResource with preempt disabled)
------------------------------------------------------------------------------
This script defines:
  - Distribution classes (Exponential, Lognormal, Bernoulli)
  - Patient class with minimal placeholders for triage/ED/ICU usage
  - Scenario class storing simulation config (capacities, service times, probabilities)
  - WardFlowModel that manages the SimPy environment, resources, nurse scheduling,
    feedback controller, arrivals, etc.
"""

import simpy
import numpy as np
import pandas as pd
import itertools
import datetime
from math import sqrt
import math
import random





class DynamicPreemptiveResource(simpy.PreemptiveResource):
    @property
    def capacity(self):
        # Return the internal capacity stored in _capacity.
        return self._capacity

    @capacity.setter
    def capacity(self, new_capacity):
        # Optionally, add a safety check here:
        if new_capacity < self.count:
            raise ValueError("New capacity cannot be less than current active usage.")
        # Update the internal capacity
        self._capacity = new_capacity


###########################################
# DISTRIBUTION CLASSES
###########################################
class Exponential:
    def __init__(self, mean, random_seed=None):
        self.mean = mean
        self.rng = np.random.default_rng(random_seed)
    def sample(self):
        return self.rng.exponential(self.mean)

class Lognormal:
    def __init__(self, mean, sigma, random_seed=None):
        self.mean = mean    # scale in log-space
        self.sigma = sigma  # standard deviation in log-space
        self.rng = np.random.default_rng(random_seed)
    def sample(self):
        return self.rng.lognormal(mean=np.log(self.mean), sigma=self.sigma)

class Bernoulli:
    def __init__(self, p, random_seed=None):
        self.p = p
        self.rng = np.random.default_rng(random_seed)
    def sample(self):
        return self.rng.random() < self.p

def trace(msg, TRACE=False):
    if TRACE:
        print(msg)

###########################################
# Summary 
###########################################

class SimulationSummary:
    def __init__(self, model):
        self.model = model
        self.patients = model.patients
        self.event_log = model.event_log
        self.utilisation = model.utilisation_audit
        self.summary = {}

    def process_run_results(self):
        times = [p.total_time for p in self.patients if p.total_time and p.total_time > 0]
        triage_waits = [p.triage_wait for p in self.patients if p.triage_wait is not None]
        icu_waits = [p.icu_wait for p in self.patients if p.icu_wait is not None]
        ms_waits = [p.medsurg_wait for p in self.patients if p.medsurg_wait is not None]
        
        for p in self.patients[:5]:
            print(f"Patient {p.pid}:")
            print(f"  Arrival: {p.arrival}")
            print(f"  Total Time: {p.total_time}")
            print(f"  Triage Wait: {p.triage_wait}")
            print(f"  ICU Wait: {p.icu_wait}")
            print(f"  MedSurg Wait: {p.medsurg_wait}")
        # Summary statistics
        # Note: np.nanmean is used to ignore NaN values in the list
      


        self.summary = {
            'n_patients': len(self.patients),
            'avg_total_time': np.mean(times) if times else 0,
            'avg_triage_wait': np.nanmean([p.triage_wait for p in self.patients]),
            'avg_icu_wait': np.nanmean([p.icu_wait if hasattr(p, "icu_wait") else np.nan for p in self.patients]),
            'avg_medsurg_wait': np.nanmean([p.medsurg_wait if hasattr(p, "medsurg_wait") else np.nan for p in self.patients]),
        }

    def summary_frame(self):
        return self.summary


def extract_event_log_summary(event_log):
    df = pd.DataFrame(event_log)
    return df.groupby(['event_type', 'event']).size().unstack(fill_value=0).to_dict()


def extract_utilisation_summary(utilisation_audit):
    df = pd.DataFrame(utilisation_audit)
    return {
        'avg_ed_usage': df['ed_in_use'].mean() if 'ed_in_use' in df else None,
        'avg_triage_usage': df['triage_in_use'].mean() if 'triage_in_use' in df else None,
        'avg_icu_available': df['icu_available'].mean() if 'icu_available' in df else None,
        'avg_medsurg_available': df['medsurg_available'].mean() if 'medsurg_available' in df else None,
        'avg_nurse_avail': df['nurses_available'].mean() if 'nurses_available' in df else None
    }


###########################################
# CUSTOM PATIENT CLASS (THIS IS PROBLEMATIC)
###########################################
class Patient:
    """
    Represents a patient in the system with placeholders for multi-phase ED approach.
    Steps might be:
      1) Triage
      2) Nurse evaluation
      3) ED evaluation
      4) Optional imaging
      5) Decision: ICU / MedSurg / direct discharge
      6) Discharge
    """
    def __init__(self, pid, env, scenario, event_log, start_datetime, severity):
        self.pid = pid
        self.env = env
        self.scenario = scenario
        self.event_log = event_log
        self.start_datetime = start_datetime
        self.severity = severity  # 1=highest priority, 3=lowest
        # Track times
        self.arrival = -np.inf
        self.total_time = -np.inf
        # Track waits
        self.triage_wait = np.nan
        self.icu_wait = np.nan
        self.medsurg_wait = np.nan

    def process(self, triage, ed, icu, medsurg, nurse):
        """
        Placeholder process for patient flows:
        - Arrive
        - Triage (request)
        - Nurse evaluation
        - ED evaluation
        - Optional imaging
        - Decide: ICU, MedSurg, or discharge
        - Final discharge
        """
        # 0) Arrival
        self.arrival = self.env.now
        self.event_log.append({
            'patient': self.pid,
            'severity': self.severity,
            'event_type': 'arrival_departure',
            'event': 'arrival',
            'time': self.env.now
        })

        # 1) Triage Phase (priority request)
        triage_wait_start = self.env.now

        try:
            with triage.request(priority=self.severity, preempt=False) as req:
                yield req
                self.triage_wait = self.env.now - triage_wait_start
                triage_time = self.scenario.triage_dist.sample()
                yield self.env.timeout(triage_time)
        except:
            # Edge case: patient interrupted or skipped
            self.triage_wait = self.env.now - triage_wait_start
            return


        # 2) Nurse evaluation (placeholder)
        with nurse.request(priority=self.severity, preempt=False) as req:
            yield req
            nurse_eval_time = np.random.exponential(5) 
            yield self.env.timeout(nurse_eval_time)

        # 3) ED evaluation
        with ed.request(priority=self.severity, preempt=False) as req:
            yield req
            ed_eval_time = self.scenario.ed_eval_dist.sample()
            yield self.env.timeout(ed_eval_time)

        # 4) Optional imaging (50% chance)
        if np.random.random() < 0.5:
            imaging_time = self.scenario.ed_imaging_dist.sample()
            yield self.env.timeout(imaging_time)

        # 5) Decision: ICU, MedSurg, or direct discharge
        if self.scenario.icu_prob_dist.sample():
            # Wait for ICU bed
            icu_wait_start = self.env.now
            icu_bed = yield icu.get()
            self.icu_wait = self.env.now - icu_wait_start
            if self.scenario.icu_proc_prob_dist.sample():
                proc_time = self.scenario.icu_proc_dist.sample()
                yield self.env.timeout(proc_time)
            icu_stay = self.scenario.icu_stay_dist.sample()
            yield self.env.timeout(icu_stay)
            yield icu.put(icu_bed)
        elif self.scenario.medsurg_prob_dist.sample():
            ms_wait_start = self.env.now
            ms_bed = yield medsurg.get()
            self.medsurg_wait = self.env.now - ms_wait_start
            ms_stay = self.scenario.medsurg_stay_dist.sample()
            yield self.env.timeout(ms_stay)
            yield medsurg.put(ms_bed)
        else:
            pass  # direct discharge

        # 6) Final discharge
        dd = self.scenario.discharge_delay_dist.sample()
        yield self.env.timeout(dd)

        if self.env.now is not None and self.arrival is not None:
            # Calculate total time in system
            if self.arrival != -np.inf:
                self.total_time = self.env.now - self.arrival
            else:
                self.total_time = np.nan

        self.event_log.append({
            'patient': self.pid,
            'event_type': 'arrival_departure',
            'event': 'discharge',
            'time': self.env.now,
            'total_time': self.total_time
        })

    

    def execute(self, triage, ed, icu, medsurg, nurse):
        yield from self.process(triage, ed, icu, medsurg, nurse)

    # For snapshot & restore:
    def to_dict(self):
        return {
            'pid': self.pid,
            'severity': self.severity,
            'arrival': self.arrival,
            'total_time': self.total_time,
            'triage_wait': self.triage_wait,
            'icu_wait': self.icu_wait,
            'medsurg_wait': self.medsurg_wait
            # Add more if needed for your flows
        }

    @classmethod
    def from_dict(cls, data):
        # We can't fully re-create the SimPy references, but let's store placeholders
        p = cls(pid=data['pid'], env=None, scenario=None, event_log=[], start_datetime=None, severity=data['severity'])
        p.arrival = data['arrival']
        p.total_time = data['total_time']
        p.triage_wait = data['triage_wait']
        p.icu_wait = data['icu_wait']
        p.medsurg_wait = data['medsurg_wait']
        return p

###########################################
# SCENARIO CLASS
###########################################
class Scenario:
    """
    Defines config:
      - Resource capacities
      - Service time means (triage, ed_eval, imaging, icu_stay, etc.)
      - Probability of ICU or MedSurg
      - Nurse scheduling
    Also constructs distribution objects for sample() usage.
    """
    def __init__(self,
                 simulation_time=7*24*60,
                 random_number_set=42,
                 n_triage=10,
                 n_ed_beds=32,
                 n_icu_beds=32,
                 n_medsurg_beds=32,

                 triage_mean=10.0,
                 ed_eval_mean=60.0,
                 ed_imaging_mean=30.0,
                 icu_stay_mean=360.0,
                 medsurg_stay_mean=240.0,
                 discharge_delay_mean=15.0,
                 icu_proc_mean=30.0,

                 p_icu=0.3,
                 p_medsurg=0.5,
                 p_icu_procedure=0.5,

                 day_shift_nurses=10,
                 night_shift_nurses=5,
                 shift_length=12,
                 model="simplified-ed-flow"):

        self.simulation_time = simulation_time
        self.random_number_set = random_number_set

        # Resource capacities
        self.n_triage = n_triage
        self.n_ed_beds = n_ed_beds
        self.n_icu_beds = n_icu_beds
        self.n_medsurg_beds = n_medsurg_beds

        # Means for service times
        self.triage_mean = triage_mean
        self.ed_eval_mean = ed_eval_mean
        self.ed_imaging_mean = ed_imaging_mean
        self.icu_stay_mean = icu_stay_mean
        self.medsurg_stay_mean = medsurg_stay_mean
        self.discharge_delay_mean = discharge_delay_mean
        self.icu_proc_mean = icu_proc_mean

        self.p_icu = p_icu
        self.p_medsurg = p_medsurg
        self.p_icu_procedure = p_icu_procedure

        self.day_shift_nurses = day_shift_nurses
        self.night_shift_nurses = night_shift_nurses
        self.shift_length = shift_length
        self.model = model

        # Build distributions
        self.triage_dist = Exponential(self.triage_mean, random_seed=self.random_number_set+1)
        self.ed_eval_dist = Exponential(self.ed_eval_mean, random_seed=self.random_number_set+2)
        self.ed_imaging_dist = Exponential(self.ed_imaging_mean, random_seed=self.random_number_set+3)
        self.icu_stay_dist = Exponential(self.icu_stay_mean, random_seed=self.random_number_set+4)
        self.medsurg_stay_dist = Exponential(self.medsurg_stay_mean, random_seed=self.random_number_set+5)
        self.discharge_delay_dist = Exponential(self.discharge_delay_mean, random_seed=self.random_number_set+6)
        self.icu_proc_dist = Exponential(self.icu_proc_mean, random_seed=self.random_number_set+7)

        self.icu_prob_dist = Bernoulli(self.p_icu, random_seed=self.random_number_set+8)
        self.medsurg_prob_dist = Bernoulli(self.p_medsurg, random_seed=self.random_number_set+9)
        self.icu_proc_prob_dist = Bernoulli(self.p_icu_procedure, random_seed=self.random_number_set+10)

    def set_random_no_set(self, random_no):
        self.random_number_set = random_no
        # Reinitialize with the new random seed
        self.__init__(
            simulation_time=self.simulation_time,
            random_number_set=self.random_number_set,
            n_triage=self.n_triage,
            n_ed_beds=self.n_ed_beds,
            n_icu_beds=self.n_icu_beds,
            n_medsurg_beds=self.n_medsurg_beds,
            triage_mean=self.triage_mean,
            ed_eval_mean=self.ed_eval_mean,
            ed_imaging_mean=self.ed_imaging_mean,
            icu_stay_mean=self.icu_stay_mean,
            medsurg_stay_mean=self.medsurg_stay_mean,
            discharge_delay_mean=self.discharge_delay_mean,
            icu_proc_mean=self.icu_proc_mean,
            p_icu=self.p_icu,
            p_medsurg=self.p_medsurg,
            p_icu_procedure=self.p_icu_procedure,
            day_shift_nurses=self.day_shift_nurses,
            night_shift_nurses=self.night_shift_nurses,
            shift_length=self.shift_length,
            model=self.model
        )

###########################################
# WARD FLOW MODEL
###########################################
HOURLY_ARRIVAL_RATES = [1, 1, 1, 1, 2, 4, 6, 8, 10, 9, 8, 7, 7, 6, 6, 5, 4, 3, 2, 2, 1, 1, 1, 1]        
    
class WardFlowModel:
    """
    Manages the simulation environment, resources, nurse scheduling,
    feedback controller, and patient arrivals.
    """

    def __init__(self, scenario, start_datetime):
        self.env = simpy.Environment()
        self.scenario = scenario
        self.start_datetime = start_datetime
        self.utilisation_audit = []
        self.event_log = []
        self.patients = []
        self.patient_count = 0


        # NEW: Set up the hourly arrival profile.
        self.hourly_arrival_profile = HOURLY_ARRIVAL_RATES

        # Triage & ED as PreemptiveResource
        self.triage = DynamicPreemptiveResource(self.env, capacity=self.scenario.n_triage)
        self.ed = DynamicPreemptiveResource(self.env, capacity=self.scenario.n_ed_beds)

        # ICU & MedSurg as simple Stores
        self.icu = simpy.Store(self.env)
        for i in range(self.scenario.n_icu_beds):
            self.icu.put(f"ICU_Bed_{i+1}")
        self.medsurg = simpy.Store(self.env)
        for i in range(self.scenario.n_medsurg_beds):
            self.medsurg.put(f"MedSurg_Bed_{i+1}")

        # Nurse resource
        self.nurses = DynamicPreemptiveResource(self.env, capacity=self.scenario.day_shift_nurses)

        # Start background process list
        self.background_processes = []


    #######################################
    # NEW: Helper Functions for Time-Based Arrivals
    #######################################
    def get_current_hour(self):
        """
        Returns the current simulation hour as an integer between 0 and 23.
        Assumes self.env.now is in minutes.
        """
        return int((self.env.now // 60) % 24)

    def sample_time_aware_interarrival(self):
        """
        Samples an interarrival time (in minutes) based on the current hour's arrival rate.
        Uses an exponential distribution where the mean is 60 divided by the arrival rate.
        """
        current_hour = self.get_current_hour()
        base_rate = self.hourly_arrival_profile[current_hour]
        if base_rate <= 0:
            # If no patients are expected this hour, return a very long interarrival time.
            return 9999  
        mean_interval = 60 / base_rate  # Mean time between arrivals in minutes

        iat = np.random.exponential(mean_interval)
        print(f"[DEBUG] At hour {current_hour}, base rate = {base_rate}, sampled IAT = {iat:.2f} minutes")
        return np.random.exponential(mean_interval)

    def arrivals_generator(self):
        """
        Generates patients with time-varying interarrival times based on an hourly profile.
        """
        for pid in itertools.count(1):
            if self.env.now >= self.scenario.simulation_time:
                break
            # Sample the interarrival time using the time-aware function.
            iat = self.sample_time_aware_interarrival()
            yield self.env.timeout(iat)
            
            severity = np.random.randint(low=1, high=4)
            p = PatientFlow(pid, self.env, self.scenario, self.event_log, self.start_datetime, severity)
            self.patients.append(p)
            self.env.process(p.execute(self.triage, self.ed, self.icu, self.medsurg, self.nurses))
            
            self.event_log.append({
                'patient': pid,
                'severity': severity,
                'event_type': 'arrival_departure',
                'event': 'arrival',
                'time': self.env.now
            })





    def nurse_shift_scheduler(self):
        while True:
            # Day shift
            self.nurses = DynamicPreemptiveResource(self.env, capacity=self.scenario.day_shift_nurses) 
            print(f"[DEBUG] Day shift started at {self.env.now}, nurse capacity set to {self.scenario.day_shift_nurses}.")           
            yield self.env.timeout(self.scenario.shift_length * 60)
            # Night shift
            self.nurses = DynamicPreemptiveResource(self.env, capacity=self.scenario.night_shift_nurses)
            print(f"[DEBUG] Day shift started at {self.env.now}, nurse capacity set to {self.scenario.night_shift_nurses}.")              
            yield self.env.timeout(self.scenario.shift_length * 60)
            
            # Log shift change
            print(f"Shift change at {self.env.now} minutes")
            self.event_log.append({
                'event_type': 'shift_change',
                'time': self.env.now,
                'day_shift_nurses': self.scenario.day_shift_nurses,
                'night_shift_nurses': self.scenario.night_shift_nurses
            })


    def feedback_controller(self, check_interval=60, ed_threshold=5):
        while True:
            yield self.env.timeout(check_interval)
            used_ed = self.ed.count

            print(f"ED usage at {self.env.now} minutes: {used_ed}")
            print(f"Event log size: {len(self.event_log)}")

            if used_ed >= ed_threshold:
                self.event_log.append({
                    'event_type': 'feedback',
                    'action': 'ED overload detected',
                    'time': self.env.now,
                    'used_ed': used_ed
                })

    def audit_utilisation(self, interval=1):
        while True:
            record = {
                'time': self.env.now,
                'triage_in_use': self.triage.count,
                'ed_in_use': self.ed.count,
                'icu_available': len(self.icu.items),
                'medsurg_available': len(self.medsurg.items),
                'nurses_available': self.nurses.capacity - self.nurses.count
            }
            self.utilisation_audit.append(record)

            #print(f"Utilisation audit at {self.env.now} minutes: {record}")
            yield self.env.timeout(interval)

    def run(self, rc_period):
        # Ensure simulation runs at least a tiny step
        if rc_period <= self.env.now:
            rc_period = self.env.now + 0.01

        # Start background processes if not already
        if not self.background_processes:
            print("===========================\nStarting background processes...\n===========================")
            self.background_processes.append(self.env.process(self.nurse_shift_scheduler()))
            self.background_processes.append(self.env.process(self.feedback_controller(check_interval=60, ed_threshold=5)))
            self.background_processes.append(self.env.process(self.audit_utilisation(interval=1)))
            self.background_processes.append(self.env.process(self.arrivals_generator()))

        self.env.run(until=rc_period)

        # Optionally cancel background at the end
        # for proc in self.background_processes:
        #     proc.interrupt()
        # self.env.run(until=self.env.now + 0.01)

###########################################
# PATIENT FLOW (Detailed)
###########################################
class PatientFlow:
    """
    Enhanced PatientFlow process including:
      - Transfer conditions with boarding.
      - Expected LOS and LOS deviation calculation.
      - Nurse throughput effect.
      - Failure escalation if boarding time exceeds threshold.
    """
    def __init__(self, pid, env, scenario, event_log, start_datetime, severity):
        self.pid = pid
        self.env = env
        self.scenario = scenario
        self.event_log = event_log
        self.start_datetime = start_datetime
        self.severity = severity  # e.g. 1 = critical, 3 = low priority
        
        self.arrival = -np.inf
        self.total_time = -np.inf
        self.triage_wait = np.nan
        self.icu_wait = np.nan
        self.medsurg_wait = np.nan
        self.expected_LOS = None  # Expected length-of-stay (to be set upon transfer)
        self.complication = False  # [CHANGE] New: flag for failure escalation

    def get_nurse_effect(self, nurse):
        # [CHANGE] Compute a nurse effect: if nurse capacity is low relative to a baseline, service times increase.
        baseline = 10  # Example baseline nurse capacity
        current = nurse.capacity if nurse.capacity > 0 else 1
        effect = max(1.0, baseline / current)
        return effect

    def process(self, triage, ed, icu, medsurg, nurse):
        # ----- 0. Arrival -----
        self.arrival = self.env.now
        self.event_log.append({
            'patient': self.pid,
            'severity': self.severity,
            'event_type': 'arrival_departure',
            'event': 'arrival',
            'time': self.env.now
        })

        # ----- 1. Triage with Nurse Assistance -----
        triage_wait_start = self.env.now
        with triage.request(priority=self.severity, preempt=False) as req:
            yield req
            self.triage_wait = self.env.now - triage_wait_start
            base_triage_time = self.scenario.triage_dist.sample()
            # Nurse influence: if nurse pool is under pressure, triage takes longer.
            nurse_effect = self.get_nurse_effect(nurse)  # [CHANGE] Nurse throughput factor
            triage_time = base_triage_time * nurse_effect
            yield self.env.timeout(triage_time)

        # ----- 2. Nurse Evaluation (using nurse pool tokens) -----
        with nurse.request(priority=self.severity, preempt=False) as req:
            yield req
            base_nurse_eval = np.random.exponential(5)
            nurse_eval_time = base_nurse_eval * nurse_effect  # [CHANGE] Scale nurse eval time
            yield self.env.timeout(nurse_eval_time)

        # ----- 3. ED Evaluation -----
        ed_wait_start = self.env.now
        with ed.request(priority=self.severity, preempt=False) as req:
            yield req
            base_ed_eval = self.scenario.ed_eval_dist.sample()
            ed_eval_time = base_ed_eval * nurse_effect
            yield self.env.timeout(ed_eval_time)
            ed_wait = self.env.now - ed_wait_start
            self.event_log.append({
                'patient': self.pid,
                'event_type': 'ed_eval',
                'event': 'completed',
                'time': self.env.now,
                'wait_time': ed_wait
            })

        # ----- 4. Optional Imaging -----
        if np.random.random() < 0.5:
            imaging_time = self.scenario.ed_imaging_dist.sample()
            yield self.env.timeout(imaging_time)

        # ----- 5. Transfer Decision: ICU vs. MedSurg vs. Direct Discharge -----
        # Here, we determine the route based on severity and probability.
        transfer_to_icu = self.scenario.icu_prob_dist.sample() and (self.severity <= 2)
        transfer_to_medsurg = self.scenario.medsurg_prob_dist.sample() and (self.severity > 1)
        
        # Set expected LOS from pre-defined means (can be refined using data)
        if transfer_to_icu:
            self.expected_LOS = getattr(self.scenario.icu_stay_dist, 'mean', self.scenario.icu_stay_dist.sample()) # LOOK INTO THIS 
            icu_wait_start = self.env.now
            max_board_time = 240  # e.g., 4 hours max boarding time
            icu_bed = None
            # Formal queueing: try to get an ICU bed token
            while (self.env.now - icu_wait_start) < max_board_time:
                if len(icu.items) > 0:
                    icu_bed = yield icu.get()
                    break
                else:
                    yield self.env.timeout(1)  # Wait 1 minute and try again.
            if not icu_bed:
                # Failure Escalation: patient deteriorates if waiting too long
                self.complication = True
                self.event_log.append({
                    'patient': self.pid,
                    'event_type': 'complication',
                    'event': 'icu_board_timeout',
                    'time': self.env.now
                })
                # End the process (or proceed with a heavy penalty)
                return
            self.icu_wait = self.env.now - icu_wait_start
            if self.scenario.icu_proc_prob_dist.sample():
                proc_time = self.scenario.icu_proc_dist.sample()
                yield self.env.timeout(proc_time)
            icu_stay = self.scenario.icu_stay_dist.sample() * nurse_effect  # extended if nurse shortage
            yield self.env.timeout(icu_stay)
            yield icu.put(icu_bed)

        elif transfer_to_medsurg:
            self.expected_LOS = getattr(self.scenario.medsurg_stay_dist, 'mean', self.scenario.medsurg_stay_dist.sample())
            ms_wait_start = self.env.now
            medsurg_bed = None
            while True:
                if len(medsurg.items) > 0:
                    medsurg_bed = yield medsurg.get()
                    break
                else:
                    yield self.env.timeout(1)
            self.medsurg_wait = self.env.now - ms_wait_start
            ms_stay = self.scenario.medsurg_stay_dist.sample() * nurse_effect
            yield self.env.timeout(ms_stay)
            yield medsurg.put(medsurg_bed)
        else:
            # Direct discharge; assign a lower expected LOS
            self.expected_LOS = getattr(self.scenario.discharge_delay_dist, 'mean', self.scenario.discharge_delay_dist.sample())

        # ----- 6. Final Discharge and LOS Evaluation -----
        dd = self.scenario.discharge_delay_dist.sample()
        yield self.env.timeout(dd)
        self.total_time = self.env.now - self.arrival
        los_deviation = self.total_time - self.expected_LOS

        self.event_log.append({
            'patient': self.pid,
            'event_type': 'arrival_departure',
            'event': 'discharge',
            'time': self.env.now,
            'total_time': self.total_time,
            'expected_LOS': self.expected_LOS,
            'los_deviation': los_deviation
        })
        
        

    def execute(self, triage, ed, icu, medsurg, nurse):
        yield from self.process(triage, ed, icu, medsurg, nurse)

    # For snapshot usage if needed
    def to_dict(self):
        return {
            'pid': self.pid,
            'severity': self.severity,
            'arrival': self.arrival,
            'total_time': self.total_time,
            'triage_wait': self.triage_wait,
            'icu_wait': self.icu_wait,
            'medsurg_wait': self.medsurg_wait
        }

    @classmethod
    def from_dict(cls, data):
        pf = cls(pid=data['pid'], env=None, scenario=None, event_log=[], start_datetime=None, severity=data['severity'])
        pf.arrival = data['arrival']
        pf.total_time = data['total_time']
        pf.triage_wait = data['triage_wait']
        pf.icu_wait = data['icu_wait']
        pf.medsurg_wait = data['medsurg_wait']
        return pf

###########################################
# Helper Functions for Single/Multiple Runs
###########################################
def single_run(scenario, rc_period, random_no_set=42, return_detailed_logs=False):
    scenario.set_random_no_set(random_no_set)
    start_datetime = datetime.datetime.now()
    model = WardFlowModel(scenario, start_datetime)
    model.run(rc_period)
    if return_detailed_logs:
        return {
            'model': model,
            'event_log': pd.DataFrame(model.event_log),
            'util_audit': pd.DataFrame(model.utilisation_audit),
            'patients': model.patients
        }
    else:
        return {'model': model}

def multiple_replications(scenario, rc_period, n_reps=3, return_detailed_logs=False):
    outputs = []
    for rep in range(n_reps):
        out = single_run(scenario, rc_period, random_no_set=(scenario.random_number_set+rep), return_detailed_logs=return_detailed_logs)
        outputs.append(out)
    return outputs

if __name__ == "__main__":
    # Basic usage
    scenario = Scenario(
        simulation_time=7*24*60,
        random_number_set=42
    )
    # single run test
    result = single_run(scenario, rc_period=scenario.simulation_time, random_no_set=42, return_detailed_logs=True)
    print("Single run finished. Event log size:", len(result['event_log']))
    print("Number of patients:", len(result['patients']))

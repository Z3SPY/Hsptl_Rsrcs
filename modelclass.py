#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Refactored Hospital Ward Simulation (PreemptiveResource with preempt disabled)
----------------------------------------------------------------------------
Key Features:
  - Uses PreemptiveResource for triage, ED, and nurses (with preempt=False) so that once a patient's request is granted,
    it won't be interrupted by a higher-priority request.
  - Patients are assigned a severity (1 = highest, 3 = lowest).
  - Multi-phase ED process: triage, nurse evaluation, ED evaluation, optional imaging, then decision (ICU/MedSurg/direct discharge),
    followed by a discharge delay.
  - Basic nurse shift scheduling and a feedback controller for ED overload logging.
  - Designed as a foundation for DRL integration.
"""

import simpy
import numpy as np
import pandas as pd
import itertools
import datetime
from math import sqrt

# -----------------------------
# DISTRIBUTION CLASSES
# -----------------------------
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

# -----------------------------
# CUSTOM RESOURCE
# -----------------------------
class CustomResource:
    """
    Represents a single resource unit (like a bed or staff member).
    """
    def __init__(self, env, capacity=1, id_attribute=None):
        self.env = env
        self.capacity = capacity
        self.id_attribute = id_attribute if id_attribute is not None else 0
    def __repr__(self):
        return f"CustomResource(id={self.id_attribute}, cap={self.capacity})"

# -----------------------------
# SCENARIO CLASS
# -----------------------------
class Scenario:
    """
    A simplified ED/ICU/Medsurg scenario:
      - Resource capacities: n_triage, n_ed_beds, n_icu_beds, n_medsurg_beds.
      - Service time means: triage_mean, ed_eval_mean, ed_imaging_mean, icu_stay_mean, medsurg_stay_mean, discharge_delay_mean, icu_proc_mean.
      - Transition probabilities: p_icu, p_medsurg, p_icu_procedure.
      - Nurse scheduling: day_shift_nurses, night_shift_nurses, shift_length.
    """
    def __init__(self,
                 simulation_time=7*24*60,
                 random_number_set=42,
                 n_triage=2,
                 n_ed_beds=4,
                 n_icu_beds=2,
                 n_medsurg_beds=4,

                 # Means for service times (in minutes)
                 triage_mean=10.0,
                 ed_eval_mean=60.0,
                 ed_imaging_mean=30.0,  # NEW: Imaging phase parameter
                 icu_stay_mean=360.0,
                 medsurg_stay_mean=240.0,
                 discharge_delay_mean=15.0,
                 icu_proc_mean=30.0,

                 # Transition probabilities
                 p_icu=0.3,
                 p_medsurg=0.5,
                 p_icu_procedure=0.5,

                 # Nurse scheduling
                 day_shift_nurses=10,
                 night_shift_nurses=5,
                 shift_length=12,   # hours per shift

                 # Model name
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
        self.ed_imaging_mean = ed_imaging_mean  # Included
        self.icu_stay_mean = icu_stay_mean
        self.medsurg_stay_mean = medsurg_stay_mean
        self.discharge_delay_mean = discharge_delay_mean
        self.icu_proc_mean = icu_proc_mean

        # Transition probabilities
        self.p_icu = p_icu
        self.p_medsurg = p_medsurg
        self.p_icu_procedure = p_icu_procedure

        # Nurse scheduling
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
        # Reinitialize with the new random seed (ensure ed_imaging_mean is included)
        self.__init__(
            simulation_time=self.simulation_time,
            random_number_set=self.random_number_set,
            n_triage=self.n_triage,
            n_ed_beds=self.n_ed_beds,
            n_icu_beds=self.n_icu_beds,
            n_medsurg_beds=self.n_medsurg_beds,
            triage_mean=self.triage_mean,
            ed_eval_mean=self.ed_eval_mean,
            ed_imaging_mean=self.ed_imaging_mean,  # added here
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

# -----------------------------
# PATIENT FLOW
# -----------------------------
class PatientFlow:
    """
    Multi-phase ED approach with priority queueing.
    Steps:
      1) Triage (priority request)
      2) Nurse Evaluation (newly added)
      3) ED evaluation (priority request)
      4) Optional imaging
      5) Decision: ICU, MedSurg, or direct discharge
      6) Discharge delay
    """
    def __init__(self, pid, env, scenario, event_log, start_datetime, severity):
        self.pid = pid
        self.env = env
        self.scenario = scenario
        self.event_log = event_log
        self.start_datetime = start_datetime
        self.severity = severity  # 1 = highest priority, 3 = lowest
        self.arrival = -np.inf
        self.total_time = -np.inf
        self.triage_wait = np.nan
        self.icu_wait = np.nan
        self.medsurg_wait = np.nan

    def process(self, triage, ed, icu, medsurg, nurse):
        # 0. Arrival
        self.arrival = self.env.now
        self.event_log.append({
            'patient': self.pid,
            'severity': self.severity,
            'event_type': 'arrival_departure',
            'event': 'arrival',
            'time': self.env.now
        })

        # 1. Triage Phase
        triage_wait_start = self.env.now
        self.event_log.append({
            'event_type': 'queue',
            'event': 'triage_wait_begins',
            'time': triage_wait_start
        })
        with triage.request(priority=self.severity, preempt=False) as req:
            yield req
            self.triage_wait = self.env.now - triage_wait_start
            self.event_log.append({
                'patient': self.pid,
                'event_type': 'resource_use',
                'event': 'triage_begins',
                'time': self.env.now,
                'resource_id': 'triage'
            })
            triage_time = self.scenario.triage_dist.sample()
            yield self.env.timeout(triage_time)
            self.event_log.append({
                'patient': self.pid,
                'event_type': 'resource_use_end',
                'event': 'triage_complete',
                'time': self.env.now,
                'duration': triage_time
            })

        # 2. Nurse Evaluation Phase (new)
        with nurse.request(priority=self.severity, preempt=False) as req:
            yield req
            self.event_log.append({
                'patient': self.pid,
                'event_type': 'resource_use',
                'event': 'nurse_eval_begins',
                'time': self.env.now,
                'resource_id': 'nurse'
            })
            nurse_eval_time = np.random.exponential(5)
            yield self.env.timeout(nurse_eval_time)
            self.event_log.append({
                'patient': self.pid,
                'event_type': 'resource_use_end',
                'event': 'nurse_eval_ends',
                'time': self.env.now,
                'duration': nurse_eval_time
            })

        # 3. ED Evaluation Phase
        ed_wait_start = self.env.now
        self.event_log.append({
            'event_type': 'queue',
            'event': 'ED_wait_begins',
            'time': ed_wait_start
        })
        with ed.request(priority=self.severity, preempt=False) as req:
            yield req
            ed_wait = self.env.now - ed_wait_start
            self.event_log.append({
                'patient': self.pid,
                'event_type': 'resource_use',
                'event': 'ed_admit',
                'time': self.env.now,
                'severity': self.severity,
                'ed_wait': ed_wait
            })
            ed_eval_time = self.scenario.ed_eval_dist.sample()
            yield self.env.timeout(ed_eval_time)
            self.event_log.append({
                'patient': self.pid,
                'event_type': 'resource_use_end',
                'event': 'ed_discharge',
                'time': self.env.now,
                'duration': ed_eval_time
            })

        # 4. Optional Imaging Phase (50% chance)
        if np.random.random() < 0.5:
            imaging_time = self.scenario.ed_imaging_dist.sample()
            self.event_log.append({
                'patient': self.pid,
                'event_type': 'resource_use',
                'event': 'imaging_start',
                'time': self.env.now
            })
            yield self.env.timeout(imaging_time)
            self.event_log.append({
                'patient': self.pid,
                'event_type': 'resource_use_end',
                'event': 'imaging_end',
                'time': self.env.now,
                'duration': imaging_time
            })

        # 5. Decision Phase: ICU, MedSurg, or Direct Discharge
        if self.scenario.icu_prob_dist.sample():
            icu_wait_start = self.env.now
            icu_bed = yield icu.get()
            self.icu_wait = self.env.now - icu_wait_start
            self.event_log.append({
                'patient': self.pid,
                'event_type': 'resource_use',
                'event': 'icu_admit',
                'time': self.env.now,
                'resource_id': icu_bed.id_attribute
            })
            if self.scenario.icu_proc_prob_dist.sample():
                proc_time = self.scenario.icu_proc_dist.sample()
                yield self.env.timeout(proc_time)
                self.event_log.append({
                    'patient': self.pid,
                    'event_type': 'resource_use_end',
                    'event': 'icu_proc_end',
                    'time': self.env.now,
                    'duration': proc_time
                })
            icu_stay = self.scenario.icu_stay_dist.sample()
            yield self.env.timeout(icu_stay)
            self.event_log.append({
                'patient': self.pid,
                'event_type': 'resource_use_end',
                'event': 'icu_discharge',
                'time': self.env.now,
                'duration': icu_stay
            })
            yield icu.put(icu_bed)
        elif self.scenario.medsurg_prob_dist.sample():
            ms_wait_start = self.env.now
            ms_bed = yield medsurg.get()
            self.medsurg_wait = self.env.now - ms_wait_start
            self.event_log.append({
                'patient': self.pid,
                'event_type': 'resource_use',
                'event': 'medsurg_admit',
                'time': self.env.now,
                'resource_id': ms_bed.id_attribute
            })
            ms_stay = self.scenario.medsurg_stay_dist.sample()
            yield self.env.timeout(ms_stay)
            self.event_log.append({
                'patient': self.pid,
                'event_type': 'resource_use_end',
                'event': 'medsurg_discharge',
                'time': self.env.now,
                'duration': ms_stay
            })
            yield medsurg.put(ms_bed)
        else:
            self.event_log.append({
                'patient': self.pid,
                'event_type': 'arrival_departure',
                'event': 'discharge',
                'time': self.env.now
            })

        # 6. Final Discharge Phase
        dd = self.scenario.discharge_delay_dist.sample()
        yield self.env.timeout(dd)
        self.total_time = self.env.now - self.arrival
        self.event_log.append({
            'patient': self.pid,
            'event_type': 'arrival_departure',
            'event': 'discharge',
            'time': self.env.now,
            'total_time': self.total_time
        })

    def execute(self, triage, ed, icu, medsurg, nurse):
        yield from self.process(triage, ed, icu, medsurg, nurse)

# -----------------------------
# WARD FLOW MODEL
# -----------------------------
class WardFlowModel:
    """
    Manages the simulation environment, resource creation, nurse scheduling,
    feedback controller, and patient arrivals.
    """
    def __init__(self, scenario, start_datetime):
        self.env = simpy.Environment()
        self.scenario = scenario
        self.start_datetime = start_datetime
        self.utilisation_audit = []
        self.event_log = []
        # ... [your existing code] ...
        self.patients = []
        self.patient_count = 0

        # Start background process list
        self.background_processes = []

        # Priority-based triage & ED using PreemptiveResource
        self.triage = simpy.PreemptiveResource(self.env, capacity=self.scenario.n_triage)
        self.ed = simpy.PreemptiveResource(self.env, capacity=self.scenario.n_ed_beds)

        # ICU & MedSurg as simple Stores
        self.icu = simpy.Store(self.env)
        for i in range(self.scenario.n_icu_beds):
            self.icu.put(CustomResource(self.env, id_attribute=i+1))
        self.medsurg = simpy.Store(self.env)
        for i in range(self.scenario.n_medsurg_beds):
            self.medsurg.put(CustomResource(self.env, id_attribute=i+1))

        # Nurses as PreemptiveResource
        self.nurses = simpy.PreemptiveResource(self.env, capacity=self.scenario.day_shift_nurses)


    def nurse_shift_scheduler(self):
        while True:
            # Set nurse resource for day shift
            self.nurses = simpy.PreemptiveResource(self.env, capacity=self.scenario.day_shift_nurses)
            yield self.env.timeout(self.scenario.shift_length * 60)
            # Set nurse resource for night shift
            self.nurses = simpy.PreemptiveResource(self.env, capacity=self.scenario.night_shift_nurses)
            yield self.env.timeout(self.scenario.shift_length * 60)

    def feedback_controller(self, check_interval=60, ed_threshold=5):
        while True:
            yield self.env.timeout(check_interval)
            used_ed = self.ed.count
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
            yield self.env.timeout(interval)

    def arrivals_generator(self):
        rng = np.random.default_rng(self.scenario.random_number_set)
        for pid in itertools.count(1):
            if self.env.now >= self.scenario.simulation_time:
                break
            # Dynamic mean interarrival time with additional noise
            base_iat = 5.0 + 2.0 * np.sin(2 * np.pi * (self.env.now % 1440) / 1440)
            noisy_iat = base_iat + np.random.uniform(-1.0, 1.0)
            mean_iat = max(3.0, noisy_iat)  # Ensures mean_iat doesn't drop below 3 minutes
            yield self.env.timeout(rng.exponential(mean_iat))
            
            self.patient_count = pid
            severity = rng.integers(low=1, high=4)
            patient = PatientFlow(pid, self.env, self.scenario, self.event_log, self.start_datetime, severity)
            self.patients.append(patient)
            self.env.process(patient.execute(self.triage, self.ed, self.icu, self.medsurg, self.nurses))


    def run(self, rc_period):
        # Start background processes and store their references
        self.background_processes.append(self.env.process(self.nurse_shift_scheduler()))
        self.background_processes.append(self.env.process(self.feedback_controller(check_interval=60, ed_threshold=5)))
        self.background_processes.append(self.env.process(self.audit_utilisation(interval=1)))
        self.background_processes.append(self.env.process(self.arrivals_generator()))
        
        # Run the simulation until the rc_period is reached
        self.env.run(until=rc_period)
        
        # At episode end, cancel all background processes so no pending events remain
        #for proc in self.background_processes:
        #    proc.interrupt("Episode ended")
        
        # Run a tiny bit more to process the cancellations
        self.env.run(until=self.env.now + 0.001)

# -----------------------------
# SIMULATION SUMMARY
# -----------------------------
class SimulationSummary:
    def __init__(self, model):
        self.model = model
        self.args = model.scenario
        self.results = {}
        self.full_event_log = model.event_log
        self.patient_log = []
        self.utilisation_audit = model.utilisation_audit

    def process_run_results(self):
        patients = self.model.patients
        valid_times = [p.total_time for p in patients if p.total_time > 0]
        self.results['arrivals'] = len(patients)
        self.results['throughput'] = len(valid_times)
        self.results['mean_total_time'] = np.mean(valid_times) if valid_times else np.nan

        for p in patients:
            if p.total_time > 0:
                self.patient_log.append({
                    'pid': p.pid,
                    'severity': p.severity,
                    'arrival': p.arrival,
                    'total_time': p.total_time,
                    'triage_wait': p.triage_wait,
                    'icu_wait': p.icu_wait,
                    'medsurg_wait': p.medsurg_wait
                })

        twaits = [x['triage_wait'] for x in self.patient_log if not np.isnan(x['triage_wait'])]
        self.results['mean_triage_wait'] = np.mean(twaits) if twaits else np.nan

        audit_df = pd.DataFrame(self.utilisation_audit)
        if not audit_df.empty:
            self.results['triage_in_use_avg'] = audit_df['triage_in_use'].mean()
            self.results['ed_in_use_avg'] = audit_df['ed_in_use'].mean()
            self.results['icu_available_avg'] = audit_df['icu_available'].mean()
            self.results['medsurg_available_avg'] = audit_df['medsurg_available'].mean()
            self.results['nurses_available_avg'] = audit_df['nurses_available'].mean()
        else:
            self.results['triage_in_use_avg'] = np.nan
            self.results['ed_in_use_avg'] = np.nan
            self.results['icu_available_avg'] = np.nan
            self.results['medsurg_available_avg'] = np.nan
            self.results['nurses_available_avg'] = np.nan

    def summary_frame(self):
        if not self.results:
            self.process_run_results()
        return pd.DataFrame({'metrics': self.results}).T

    def detailed_logs(self):
        return {
            'full_event_log': pd.DataFrame(self.full_event_log),
            'patient_log': pd.DataFrame(self.patient_log),
            'utilisation_audit': pd.DataFrame(self.utilisation_audit),
            'summary_df': self.summary_frame()
        }

# -----------------------------
# HELPER FUNCTIONS
# -----------------------------
def single_run(scenario, rc_period, random_no_set=42, return_detailed_logs=False):
    scenario.set_random_no_set(random_no_set)
    scenario.simulation_time = rc_period
    start_datetime = datetime.datetime(2025, 3, 21, 6, 0)
    model = WardFlowModel(scenario, start_datetime)
    model.run(rc_period)
    summary = SimulationSummary(model)
    summary.process_run_results()
    if return_detailed_logs:
        return {
            'results': {
                'summary_df': summary.summary_frame(),
                'full_event_log': pd.DataFrame(model.event_log),
                'patient_log': pd.DataFrame(summary.patient_log)
            }
        }
    return summary.summary_frame()

def multiple_replications(scenario, rc_period, n_reps=3, return_detailed_logs=False):
    outputs = []
    for rep in range(n_reps):
        out = single_run(scenario, rc_period, random_no_set=scenario.random_number_set + rep, return_detailed_logs=True)
        out["results"]["full_event_log"] = out["results"]["full_event_log"].assign(rep=rep+1)
        out["results"]["summary_df"]["rep"] = rep+1
        outputs.append(out)
    if return_detailed_logs:
        return outputs
    df_results = pd.concat([o["results"]["summary_df"] for o in outputs])
    df_results.index = range(1, len(df_results)+1)
    df_results.index.name = "rep"
    return df_results

# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":
    scenario = Scenario(
        simulation_time=7*24*60,
        random_number_set=42,
        n_triage=2,
        n_ed_beds=4,
        n_icu_beds=2,
        n_medsurg_beds=4,

        # Means for service times
        ed_imaging_mean=30.0,
        triage_mean=10.0,
        ed_eval_mean=60.0,
        icu_stay_mean=360.0,
        medsurg_stay_mean=240.0,
        discharge_delay_mean=15.0,
        icu_proc_mean=30.0,

        # Transition probabilities
        p_icu=0.3,
        p_medsurg=0.5,
        p_icu_procedure=0.5,

        # Nurse scheduling
        day_shift_nurses=10,
        night_shift_nurses=5,
        shift_length=12,
        model="simplified-ed-flow"
    )
    
    df_summary = multiple_replications(scenario, scenario.simulation_time, n_reps=3)
    pd.set_option('display.max_columns', None)
    print("Summary of Multiple Replications:")
    print(df_summary)

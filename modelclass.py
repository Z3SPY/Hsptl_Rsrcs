#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Three-Ward Hospital Simulation (Refactored + Improved)
------------------------------------------------------
Key Changes:
  - Shift-based nurse scheduling
  - Feedback mechanism for ED surges
  - Example: lognormal & exponential service times
  - Hints for future DRL integration
"""

import simpy
import numpy as np
import pandas as pd
import itertools
import datetime

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
    """
    You can use lognormal to capture skewed distributions often seen in hospital data.
    mean = desired mean *of the distribution*, sigma = standard deviation (log-space).
    """
    def __init__(self, mean, sigma, random_seed=None):
        self.mean = mean
        self.sigma = sigma
        self.rng = np.random.default_rng(random_seed)
        
        # The 'mean' param in a lognormal can be tricky to interpret; it might be
        # the *median* or we might adjust for log-space. We'll keep it simple here.
    def sample(self):
        return self.rng.lognormal(mean=np.log(self.mean), sigma=self.sigma)

# Example gamma distribution, if you want to use it:
# from scipy.stats import gamma
# class GammaDist:
#     def __init__(self, shape, scale, random_seed=None):
#         # shape = k, scale = theta
#         self.shape = shape
#         self.scale = scale
#         self.rng = np.random.default_rng(random_seed)
#     def sample(self):
#         return gamma.rvs(self.shape, scale=self.scale, random_state=self.rng)

class Bernoulli:
    def __init__(self, p, random_seed=None):
        self.p = p
        self.rng = np.random.default_rng(random_seed)
    def sample(self):
        return self.rng.random() < self.p

# -----------------------------
# UTILITY
# -----------------------------

def trace(msg, TRACE=False):
    if TRACE:
        print(msg)

# -----------------------------
# CUSTOM RESOURCE
# -----------------------------

class CustomResource:
    def __init__(self, env, capacity, id_attribute=None):
        self.env = env
        self.capacity = capacity  # For reference, not used in Store logic directly.
        self.id_attribute = id_attribute if id_attribute is not None else 0
    def __repr__(self):
        return f"CustomResource(id={self.id_attribute}, cap={self.capacity})"

# -----------------------------
# SCENARIO CLASS
# -----------------------------

class Scenario:
    """
    Holds the parameters for the simulation: resource capacities, distribution objects, etc.
    """
    def __init__(self,
                 simulation_time=7*24*60,  # total run time in minutes
                 random_number_set=42,
                 n_triage=4,
                 n_ed_beds=6,
                 n_icu_beds=4,
                 n_medsurg_beds=8,
                 # Service time means (we can override with lognormal or other distributions).
                 triage_mean=5.0,
                 ed_stay_mean=60.0,
                 icu_stay_mean=180.0,
                 medsurg_stay_mean=120.0,
                 p_icu=0.3,
                 p_medsurg=0.5,
                 p_icu_procedure=0.5,
                 icu_proc_mean=30.0,
                 # Registration / exam resources
                 n_reg=1,
                 n_exam=3,
                 reg_mean=8.0,
                 reg_var=2.0,
                 exam_mean=16.0,
                 exam_var=3.0,
                 # Nurse scheduling
                 day_shift_nurses=12,
                 night_shift_nurses=8,
                 shift_length=12,  # hours for each shift block
                 model="3-ward-flow"):

        self.simulation_time = simulation_time
        self.random_number_set = random_number_set
        
        # Resource capacities
        self.n_triage = n_triage
        self.n_ed_beds = n_ed_beds
        self.n_icu_beds = n_icu_beds
        self.n_medsurg_beds = n_medsurg_beds
        self.n_reg = n_reg
        self.n_exam = n_exam
        
        # Nurse shifts
        self.day_shift_nurses = day_shift_nurses
        self.night_shift_nurses = night_shift_nurses
        self.shift_length = shift_length  # in hours

        # Service time parameters (we'll attach distributions)
        self.triage_mean = triage_mean
        self.ed_stay_mean = ed_stay_mean
        self.icu_stay_mean = icu_stay_mean
        self.medsurg_stay_mean = medsurg_stay_mean
        self.icu_proc_mean = icu_proc_mean
        
        # Probabilities
        self.p_icu = p_icu
        self.p_medsurg = p_medsurg
        self.p_icu_procedure = p_icu_procedure
        
        self.model = model

        # Build distributions (you can replace them with lognormal if you want)
        # Currently mixing an Exponential for triage, ED with a Lognormal for ICU, etc. as an example:
        self.triage_dist = Exponential(self.triage_mean, random_seed=self.random_number_set+1)
        self.ed_stay_dist = Exponential(self.ed_stay_mean, random_seed=self.random_number_set+2)
        # Example switch to lognormal for ICU:
        self.icu_stay_dist = Lognormal(mean=self.icu_stay_mean, sigma=0.5, random_seed=self.random_number_set+3)
        self.medsurg_stay_dist = Exponential(self.medsurg_stay_mean, random_seed=self.random_number_set+4)
        self.icu_proc_dist = Exponential(self.icu_proc_mean, random_seed=self.random_number_set+5)
        
        # Probabilities
        self.icu_prob_dist = Bernoulli(self.p_icu, random_seed=self.random_number_set+6)
        self.medsurg_prob_dist = Bernoulli(self.p_medsurg, random_seed=self.random_number_set+7)
        self.icu_proc_prob_dist = Bernoulli(self.p_icu_procedure, random_seed=self.random_number_set+8)

        # For non-trauma phases (registration, exam), use lognormal or normal
        # We'll show just lognormal for demonstration
        self.reg_mean = reg_mean
        self.reg_var = reg_var
        self.exam_mean = exam_mean
        self.exam_var = exam_var
        from math import sqrt
        self.reg_dist = Lognormal(self.reg_mean, sqrt(self.reg_var), random_seed=self.random_number_set+9)
        self.exam_dist = Lognormal(self.exam_mean, sqrt(self.exam_var), random_seed=self.random_number_set+10)

    def set_random_no_set(self, random_number_set):
        """
        Re-seed the scenario if needed, e.g. for multiple replications.
        """
        self.random_number_set = random_number_set
        self.__init__(simulation_time=self.simulation_time,
                      random_number_set=self.random_number_set,
                      n_triage=self.n_triage,
                      n_ed_beds=self.n_ed_beds,
                      n_icu_beds=self.n_icu_beds,
                      n_medsurg_beds=self.n_medsurg_beds,
                      triage_mean=self.triage_mean,
                      ed_stay_mean=self.ed_stay_mean,
                      icu_stay_mean=self.icu_stay_mean,
                      medsurg_stay_mean=self.medsurg_stay_mean,
                      p_icu=self.p_icu,
                      p_medsurg=self.p_medsurg,
                      p_icu_procedure=self.p_icu_procedure,
                      icu_proc_mean=self.icu_proc_mean,
                      n_reg=self.n_reg,
                      n_exam=self.n_exam,
                      reg_mean=self.reg_mean,
                      reg_var=self.reg_var,
                      exam_mean=self.exam_mean,
                      exam_var=self.exam_var,
                      day_shift_nurses=self.day_shift_nurses,
                      night_shift_nurses=self.night_shift_nurses,
                      shift_length=self.shift_length,
                      model=self.model)

# -----------------------------
# PATIENT FLOW
# -----------------------------

class PatientFlow:
    """
    A patient goes through:
      1. Triage & Nurse Evaluation
      2. ED Stay
      3. Potentially ICU or MedSurg
      4. Discharge
    """
    def __init__(self, pid, env, scenario, event_log, start_datetime):
        self.pid = pid
        self.env = env
        self.scenario = scenario
        self.event_log = event_log
        self.start_datetime = start_datetime
        self.arrival = -np.inf
        self.total_time = -np.inf
        
        # Optional wait metrics
        self.triage_wait = np.nan
        self.registration_wait = np.nan
        self.examination_wait = np.nan
        self.icu_wait = np.nan
        self.medsurg_wait = np.nan

    def process(self, triage_store, ed_store, icu_store, medsurg_store, nurse_store):
        # 1. Arrival
        self.arrival = self.env.now
        self.event_log.append({
            'patient': self.pid,
            'pathway': 'Unified',
            'event_type': 'arrival_departure',
            'event': 'arrival',
            'time': self.env.now
        })
        
        # 1a. Triage Wait
        triage_wait_start = self.env.now
        self.event_log.append({
            'patient': self.pid,
            'event_type': 'queue',
            'event': 'triage_wait_begins',
            'time': self.env.now
        })
        triage_resource = yield triage_store.get()
        self.triage_wait = self.env.now - triage_wait_start
        self.event_log.append({
            'patient': self.pid,
            'event_type': 'resource_use',
            'event': 'triage_begins',
            'time': self.env.now,
            'resource_id': triage_resource.id_attribute
        })
        triage_duration = self.scenario.triage_dist.sample()
        yield self.env.timeout(triage_duration)
        self.event_log.append({
            'patient': self.pid,
            'event_type': 'resource_use_end',
            'event': 'triage_complete',
            'time': self.env.now,
            'duration': triage_duration,
            'resource_id': triage_resource.id_attribute
        })
        triage_store.put(triage_resource)

        # 1b. Nurse Evaluation
        nurse = yield nurse_store.get()
        eval_duration = np.random.exponential(2)  # or some other distribution
        self.event_log.append({
            'patient': self.pid,
            'event_type': 'resource_use',
            'event': 'nurse_evaluation_begins',
            'time': self.env.now,
            'resource_id': nurse.id_attribute
        })
        yield self.env.timeout(eval_duration)
        self.event_log.append({
            'patient': self.pid,
            'event_type': 'resource_use_end',
            'event': 'nurse_evaluation_ends',
            'time': self.env.now,
            'duration': eval_duration,
            'resource_id': nurse.id_attribute
        })
        nurse_store.put(nurse)

        # 2. ED Wait
        self.event_log.append({
            'patient': self.pid,
            'event_type': 'queue',
            'event': 'ed_wait_begins',
            'time': self.env.now
        })
        ed_resource = yield ed_store.get()
        self.event_log.append({
            'patient': self.pid,
            'event_type': 'resource_use',
            'event': 'ed_admit',
            'time': self.env.now,
            'resource_id': ed_resource.id_attribute
        })
        ed_duration = self.scenario.ed_stay_dist.sample()
        yield self.env.timeout(ed_duration)
        self.event_log.append({
            'patient': self.pid,
            'event_type': 'resource_use_end',
            'event': 'ed_discharge',
            'time': self.env.now,
            'duration': ed_duration,
            'resource_id': ed_resource.id_attribute
        })
        ed_store.put(ed_resource)

        # 3. Decision: ICU, MedSurg, or Discharge
        if self.scenario.icu_prob_dist.sample():
            # ICU
            self.event_log.append({
                'patient': self.pid,
                'event_type': 'transfer',
                'event': 'transfer_to_icu',
                'time': self.env.now
            })
            icu_resource = yield icu_store.get()
            self.icu_wait = self.env.now  # For demonstration
            self.event_log.append({
                'patient': self.pid,
                'event_type': 'resource_use',
                'event': 'icu_admit',
                'time': self.env.now,
                'resource_id': icu_resource.id_attribute
            })
            if self.scenario.icu_proc_prob_dist.sample():
                # ICU Procedure
                proc_time = self.scenario.icu_proc_dist.sample()
                self.event_log.append({
                    'patient': self.pid,
                    'event_type': 'resource_use',
                    'event': 'icu_procedure_start',
                    'time': self.env.now
                })
                yield self.env.timeout(proc_time)
                self.event_log.append({
                    'patient': self.pid,
                    'event_type': 'resource_use_end',
                    'event': 'icu_procedure_end',
                    'time': self.env.now,
                    'duration': proc_time
                })
            icu_stay_time = self.scenario.icu_stay_dist.sample()
            yield self.env.timeout(icu_stay_time)
            self.event_log.append({
                'patient': self.pid,
                'event_type': 'resource_use_end',
                'event': 'icu_discharge',
                'time': self.env.now,
                'duration': icu_stay_time
            })
            icu_store.put(icu_resource)
        elif self.scenario.medsurg_prob_dist.sample():
            # MedSurg
            self.event_log.append({
                'patient': self.pid,
                'event_type': 'transfer',
                'event': 'transfer_to_medsurg',
                'time': self.env.now
            })
            ms_resource = yield medsurg_store.get()
            self.medsurg_wait = self.env.now
            self.event_log.append({
                'patient': self.pid,
                'event_type': 'resource_use',
                'event': 'medsurg_admit',
                'time': self.env.now,
                'resource_id': ms_resource.id_attribute
            })
            ms_stay_time = self.scenario.medsurg_stay_dist.sample()
            yield self.env.timeout(ms_stay_time)
            self.event_log.append({
                'patient': self.pid,
                'event_type': 'resource_use_end',
                'event': 'medsurg_discharge',
                'time': self.env.now,
                'duration': ms_stay_time
            })
            medsurg_store.put(ms_resource)
        else:
            # Direct discharge
            self.event_log.append({
                'patient': self.pid,
                'event_type': 'transfer',
                'event': 'no_transfer_direct_discharge',
                'time': self.env.now
            })

        # 4. Discharge
        self.event_log.append({
            'patient': self.pid,
            'event_type': 'arrival_departure',
            'event': 'discharge',
            'time': self.env.now
        })
        self.total_time = self.env.now - self.arrival

    def execute(self, triage_store, ed_store, icu_store, medsurg_store, nurse_store):
        yield from self.process(triage_store, ed_store, icu_store, medsurg_store, nurse_store)

# -----------------------------
# MODEL CLASS
# -----------------------------

class WardFlowModel:
    """
    Manages the simpy environment, resources, arrivals, and auditing.
    """
    def __init__(self, scenario, start_datetime):
        self.env = simpy.Environment()
        self.scenario = scenario
        self.start_datetime = start_datetime

        self.utilisation_audit = []  # For logging resource usage over time

        # Triage, ED, ICU, MedSurg as simpy.Store
        self.triage = simpy.Store(self.env)
        for i in range(self.scenario.n_triage):
            self.triage.put(CustomResource(self.env, capacity=1, id_attribute=i+1))

        self.ed_beds = simpy.Store(self.env)
        for i in range(self.scenario.n_ed_beds):
            self.ed_beds.put(CustomResource(self.env, capacity=1, id_attribute=i+1))

        self.icu_beds = simpy.Store(self.env)
        for i in range(self.scenario.n_icu_beds):
            self.icu_beds.put(CustomResource(self.env, capacity=1, id_attribute=i+1))

        self.medsurg_beds = simpy.Store(self.env)
        for i in range(self.scenario.n_medsurg_beds):
            self.medsurg_beds.put(CustomResource(self.env, capacity=1, id_attribute=i+1))

        # Nurse resource (uses a Store)
        self.nurses = simpy.Store(self.env)

        # Additional resources for registration / exam if needed
        self.scenario.registration = simpy.Store(self.env)
        for i in range(self.scenario.n_reg):
            self.scenario.registration.put(CustomResource(self.env, capacity=1, id_attribute=i+1))

        self.scenario.exam = simpy.Store(self.env)
        for i in range(self.scenario.n_exam):
            self.scenario.exam.put(CustomResource(self.env, capacity=1, id_attribute=i+1))

        self.event_log = []
        self.patient_count = 0
        self.patients = []

    def nurse_shift_scheduler(self):
        """
        Periodically resets the nurse_store to mimic day/night shifts.
        For simplicity, day shift => scenario.day_shift_nurses, night shift => scenario.night_shift_nurses.
        """
        while True:
            # 1. Clear all nurses from store
            while len(self.nurses.items) > 0:
                yield self.nurses.get()

            # 2. Add day shift nurses
            for i in range(self.scenario.day_shift_nurses):
                self.nurses.put(CustomResource(self.env, capacity=1, id_attribute=i+1))

            # Run for 'shift_length' hours
            yield self.env.timeout(self.scenario.shift_length * 60)

            # 3. Clear out day shift nurses
            while len(self.nurses.items) > 0:
                yield self.nurses.get()

            # 4. Add night shift nurses
            for i in range(self.scenario.night_shift_nurses):
                self.nurses.put(CustomResource(self.env, capacity=1, id_attribute=i+1))

            yield self.env.timeout(self.scenario.shift_length * 60)

    def feedback_controller(self, check_interval=60, ed_threshold=5):
        """
        A simple feedback loop that checks ED queue length. If it exceeds 'ed_threshold',
        we temporarily add an extra nurse for that hour, if not already added.
        """
        extra_nurse_in_place = False
        nurse_id = 999  # ID for the "surge" nurse

        while True:
            yield self.env.timeout(check_interval)  # check every 'check_interval' minutes

            # We can approximate ED queue length by seeing how many get() processes are blocked.
            # Alternatively, you might track it via an event log or use a placeholder variable.
            # For demonstration, let's assume the ED queue length is the difference between capacity and items:
            ed_available = len(self.ed_beds.items)
            ed_capacity = self.scenario.n_ed_beds
            # This is simplistic because 'ed_available' doesn't directly tell us how many are waiting.
            # A more accurate approach: track how many "requests" are waiting to get an ed_bed.
            # We'll do a naive approach here:
            ed_waiters = ed_capacity - ed_available  # how many are in use

            if ed_waiters >= ed_threshold and not extra_nurse_in_place:
                # Add an extra nurse
                self.nurses.put(CustomResource(self.env, capacity=1, id_attribute=nurse_id))
                extra_nurse_in_place = True
                self.event_log.append({
                    'event_type': 'feedback_action',
                    'event': 'added_surge_nurse',
                    'time': self.env.now
                })
            elif ed_waiters < ed_threshold and extra_nurse_in_place:
                # Remove that extra nurse
                # We find the nurse with id = nurse_id
                # We'll remove it if it's in the store.
                removed_count = 0
                new_nurse_list = []
                while len(self.nurses.items) > 0:
                    n = yield self.nurses.get()
                    if n.id_attribute == nurse_id and removed_count == 0:
                        # Skip re-adding this nurse, effectively removing them
                        removed_count += 1
                    else:
                        new_nurse_list.append(n)
                # Re-add the others
                for n in new_nurse_list:
                    self.nurses.put(n)
                extra_nurse_in_place = False
                self.event_log.append({
                    'event_type': 'feedback_action',
                    'event': 'removed_surge_nurse',
                    'time': self.env.now
                })

    def audit_utilisation(self, interval=1):
        """
        Periodic logging of resource availability.
        """
        while True:
            record = {
                'time': self.env.now,
                'triage_available': len(self.triage.items),
                'ed_beds_available': len(self.ed_beds.items),
                'icu_beds_available': len(self.icu_beds.items),
                'medsurg_beds_available': len(self.medsurg_beds.items),
                'nurses_available': len(self.nurses.items)
            }
            self.utilisation_audit.append(record)
            yield self.env.timeout(interval)

    def arrivals_generator(self):
        """
        Generates patient arrivals. For now, we'll keep a simple exponential with mean=5 minutes.
        You could replace this with a day/night rate or a distribution derived from data.
        """
        rng = np.random.default_rng(self.scenario.random_number_set)
        mean_iat = 5.0  # minutes
        for pid in itertools.count(1):
            if self.env.now >= self.scenario.simulation_time:
                break

            iat = rng.exponential(mean_iat)
            yield self.env.timeout(iat)
            self.patient_count = pid
            patient = PatientFlow(pid, self.env, self.scenario, self.event_log, self.start_datetime)
            self.patients.append(patient)
            self.env.process(patient.execute(
                self.triage,
                self.ed_beds,
                self.icu_beds,
                self.medsurg_beds,
                self.nurses
            ))

    def run(self, results_collection_period):
        """
        Launch all processes and run the simulation.
        """
        # Start the shift scheduler
        self.env.process(self.nurse_shift_scheduler())
        # Start a simple feedback controller
        self.env.process(self.feedback_controller(check_interval=60, ed_threshold=5))
        # Start resource usage audit
        self.env.process(self.audit_utilisation(interval=1))
        # Start arrivals
        self.env.process(self.arrivals_generator())

        # Run the simulation
        self.env.run(until=results_collection_period)

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
    
    def process_run_results(self):
        patients = self.model.patients
        total_times = np.array([p.total_time for p in patients if p.total_time is not None])
        self.results['00_arrivals'] = len(patients)
        self.results['08_total_time_mean'] = np.mean(total_times) if total_times.size > 0 else np.nan
        self.results['09_throughput'] = len([p for p in patients if p.total_time is not None])
        
        # Patient log
        for p in patients:
            self.patient_log.append({
                'pid': p.pid,
                'arrival': p.arrival,
                'total_time': p.total_time,
                'triage_wait': p.triage_wait,
                'icu_wait': p.icu_wait,
                'medsurg_wait': p.medsurg_wait
            })
        
        # Example: average triage wait
        triage_waits = [d['triage_wait'] for d in self.patient_log if not np.isnan(d['triage_wait'])]
        if len(triage_waits) > 0:
            self.results['01_avg_triage_wait'] = np.mean(triage_waits)
        else:
            self.results['01_avg_triage_wait'] = np.nan
        
        # Resource utilisation
        audit_df = pd.DataFrame(self.model.utilisation_audit)
        if not audit_df.empty:
            self.results['triage_util'] = 1 - (audit_df['triage_available'].mean() / self.args.n_triage)
            self.results['ed_util'] = 1 - (audit_df['ed_beds_available'].mean() / self.args.n_ed_beds)
            self.results['icu_util'] = 1 - (audit_df['icu_beds_available'].mean() / self.args.n_icu_beds)
            self.results['nurses_avg'] = audit_df['nurses_available'].mean()
        else:
            self.results['triage_util'] = np.nan
            self.results['ed_util'] = np.nan
            self.results['icu_util'] = np.nan
            self.results['nurses_avg'] = np.nan

    def summary_frame(self):
        if not self.results:
            self.process_run_results()
        df = pd.DataFrame({'metrics': self.results}).T
        return df

    def detailed_logs(self):
        return {
            'full_event_log': pd.DataFrame(self.full_event_log),
            'patient_log': pd.DataFrame(self.patient_log),
            'utilisation_audit': pd.DataFrame(self.model.utilisation_audit),
            'summary_df': self.summary_frame()
        }

# -----------------------------
# RUNNERS
# -----------------------------

def single_run(scenario, rc_period, random_no_set=42, return_detailed_logs=False):
    scenario.set_random_no_set(random_no_set)
    scenario.simulation_time = rc_period
    start_datetime = datetime.datetime(2025, 3, 21, 6, 0)
    model = WardFlowModel(scenario, start_datetime)
    model.run(results_collection_period=rc_period)
    summary_obj = SimulationSummary(model)
    summary_obj.process_run_results()
    if return_detailed_logs:
        return {
            'results': {
                'summary_df': summary_obj.summary_frame(),
                'full_event_log': pd.DataFrame(model.event_log),
                'patient_log': pd.DataFrame(summary_obj.patient_log)
            }
        }
    else:
        return summary_obj.summary_frame()

def multiple_replications(scenario, rc_period, n_reps, return_detailed_logs=False):
    outputs = []
    for rep in range(n_reps):
        out = single_run(scenario, rc_period, random_no_set=scenario.random_number_set + rep,
                         return_detailed_logs=True)
        out["results"]["full_event_log"] = out["results"]["full_event_log"].assign(rep=rep+1)
        out["results"]["summary_df"]["rep"] = rep + 1
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
        simulation_time=5*24*60,
        random_number_set=42,
        n_triage=4,
        n_ed_beds=6,
        n_icu_beds=4,
        n_medsurg_beds=8,
        triage_mean=5.0,
        ed_stay_mean=60.0,
        icu_stay_mean=180.0,
        medsurg_stay_mean=120.0,
        p_icu=0.3,
        p_medsurg=0.5,
        p_icu_procedure=0.5,
        icu_proc_mean=30.0,
        n_reg=1,
        n_exam=3,
        reg_mean=8.0,
        reg_var=2.0,
        exam_mean=16.0,
        exam_var=3.0,
        day_shift_nurses=12,
        night_shift_nurses=8,
        shift_length=12,  # 12-hour shifts
        model="3-ward-flow"
    )
    df_summary = multiple_replications(scenario, rc_period=scenario.simulation_time, n_reps=3)
    print("Summary of Multiple Replications:")
    pd.set_option('display.max_columns', None)
    print(df_summary)

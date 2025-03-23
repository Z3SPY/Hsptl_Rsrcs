#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Three-Ward Hospital Simulation (Unified Patient Flow) – Refactored for Bottleneck Analysis

This refactored model captures a more detailed patient flow:
  1. Arrival and Triage for all patients.
  2. For non‑trauma patients, an added branch for Registration and Examination.
  3. All patients then wait for ED care.
  4. A decision phase routes patients to ICU, MedSurg, or discharge.
  
Each phase logs waiting and service events (with resource IDs) to facilitate detailed analysis of bottlenecks.
"""

import simpy
import numpy as np
import pandas as pd
import itertools
import datetime

# -----------------------------
# Custom Resource Class
# -----------------------------
class CustomResource:
    def __init__(self, env, capacity, id_attribute=None):
        self.env = env
        self.capacity = capacity  # For reference.
        self.id_attribute = id_attribute if id_attribute is not None else 0

    def __repr__(self):
        return f"CustomResource(id={self.id_attribute}, cap={self.capacity})"

# -----------------------------
# Distribution Classes
# -----------------------------
class Exponential:
    def __init__(self, mean, random_seed=None):
        self.mean = mean
        self.rng = np.random.default_rng(random_seed)
    def sample(self):
        return self.rng.exponential(self.mean)

class Bernoulli:
    def __init__(self, p, random_seed=None):
        self.p = p
        self.rng = np.random.default_rng(random_seed)
    def sample(self):
        return self.rng.random() < self.p

# For simplicity, assume basic implementations for Lognormal and Normal:
class Lognormal:
    def __init__(self, mean, sigma, random_seed=None):
        self.mean = mean
        self.sigma = sigma
        self.rng = np.random.default_rng(random_seed)
    def sample(self):
        return self.rng.lognormal(mean=np.log(self.mean) - 0.5*self.sigma**2, sigma=self.sigma)

class Normal:
    def __init__(self, mean, sigma, random_seed=None):
        self.mean = mean
        self.sigma = sigma
        self.rng = np.random.default_rng(random_seed)
    def sample(self):
        return self.rng.normal(self.mean, self.sigma)

# -----------------------------
# Utility Function
# -----------------------------
def trace(msg, TRACE=False):
    if TRACE:
        print(msg)

# -----------------------------
# Extended Scenario Class (Including Non-Trauma Parameters)
# -----------------------------
class Scenario:
    def __init__(self,
                 simulation_time=7*24*60,  # in minutes
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
                 # New parameters for non-trauma pathway:
                 n_reg=1,
                 n_exam=3,
                 reg_mean=8.0,
                 reg_var=2.0,
                 exam_mean=16.0,
                 exam_var=3.0,
                 model="3-ward-flow"):
        self.simulation_time = simulation_time
        self.random_number_set = random_number_set
        
        # Resource counts
        self.n_triage = n_triage
        self.n_ed_beds = n_ed_beds
        self.n_icu_beds = n_icu_beds
        self.n_medsurg_beds = n_medsurg_beds
        self.n_reg = n_reg
        self.n_exam = n_exam
        
        # Service time parameters
        self.triage_mean = triage_mean
        self.ed_stay_mean = ed_stay_mean
        self.icu_stay_mean = icu_stay_mean
        self.medsurg_stay_mean = medsurg_stay_mean
        self.icu_proc_mean = icu_proc_mean
        self.reg_mean = reg_mean
        self.reg_var = reg_var
        self.exam_mean = exam_mean
        self.exam_var = exam_var
        
        # Transition probabilities
        self.p_icu = p_icu
        self.p_medsurg = p_medsurg
        self.p_icu_procedure = p_icu_procedure
        
        self.model = model
        
        # Distributions for unified phases:
        self.triage_dist = Exponential(self.triage_mean, random_seed=self.random_number_set+1)
        self.ed_stay_dist = Exponential(self.ed_stay_mean, random_seed=self.random_number_set+2)
        self.icu_stay_dist = Exponential(self.icu_stay_mean, random_seed=self.random_number_set+3)
        self.medsurg_stay_dist = Exponential(self.medsurg_stay_mean, random_seed=self.random_number_set+4)
        self.icu_proc_dist = Exponential(self.icu_proc_mean, random_seed=self.random_number_set+5)
        
        self.icu_prob_dist = Bernoulli(self.p_icu, random_seed=self.random_number_set+6)
        self.medsurg_prob_dist = Bernoulli(self.p_medsurg, random_seed=self.random_number_set+7)
        self.icu_proc_prob_dist = Bernoulli(self.p_icu_procedure, random_seed=self.random_number_set+8)
        
        # New distributions for non-trauma phases:
        self.reg_dist = Lognormal(self.reg_mean, np.sqrt(self.reg_var), random_seed=self.random_number_set+9)
        self.exam_dist = Normal(self.exam_mean, np.sqrt(self.exam_var), random_seed=self.random_number_set+10)

    def set_random_no_set(self, random_number_set):
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
                      model=self.model)

# -----------------------------
# PatientFlow Class (Unified) – Enhanced Realism with Non-Trauma Pathway
# -----------------------------
class PatientFlow:
    """
    Models the unified patient flow with additional waiting phases:
      Arrival -> Triage -> (Optional: Registration -> Examination) -> ED or ICU/Medsurg Decision -> Discharge.
    Logs waiting and service events along with resource utilization.
    """
    def __init__(self, pid, env, scenario, event_log, start_datetime):
        self.pid = pid
        self.env = env
        self.scenario = scenario
        self.event_log = event_log
        self.start_datetime = start_datetime
        self.arrival = -np.inf
        self.total_time = -np.inf
        # Wait metrics (default NaN if not computed)
        self.triage_wait = np.nan
        self.registration_wait = np.nan
        self.examination_wait = np.nan
        self.icu_wait = np.nan
        self.medsurg_wait = np.nan

    def process(self, triage_store, ed_store, icu_store, medsurg_store):
        # 1. Arrival and Triage Phase
        self.arrival = self.env.now
        self.event_log.append({
            'patient': self.pid,
            'pathway': 'Unified',
            'event_type': 'arrival_departure',
            'event': 'arrival',
            'time': self.env.now,
            'resource_id': 0
        })
        # Record triage wait start time
        triage_wait_start = self.env.now
        self.event_log.append({
            'patient': self.pid,
            'pathway': 'Unified',
            'event_type': 'queue',
            'event': 'triage_wait_begins',
            'time': self.env.now
        })
        triage_resource = yield triage_store.get()
        self.triage_wait = self.env.now - triage_wait_start  # Compute triage wait time
        self.event_log.append({
            'patient': self.pid,
            'pathway': 'Unified',
            'event_type': 'resource_use',
            'event': 'triage_begins',
            'time': self.env.now,
            'resource_id': triage_resource.id_attribute
        })
        triage_duration = self.scenario.triage_dist.sample()
        yield self.env.timeout(triage_duration)
        self.event_log.append({
            'patient': self.pid,
            'pathway': 'Unified',
            'event_type': 'resource_use_end',
            'event': 'triage_complete',
            'time': self.env.now,
            'duration': triage_duration,
            'resource_id': triage_resource.id_attribute
        })
        triage_store.put(triage_resource)

        # 2. Registration & Examination Phase (Non-Trauma Only)
        if not self.scenario.icu_prob_dist.sample():
            # Registration Phase
            registration_wait_start = self.env.now
            self.event_log.append({
                'patient': self.pid,
                'pathway': 'Non-Trauma',
                'event_type': 'queue',
                'event': 'registration_wait_begins',
                'time': self.env.now
            })
            registration_resource = yield self.scenario.registration.get()
            self.registration_wait = self.env.now - registration_wait_start
            self.event_log.append({
                'patient': self.pid,
                'pathway': 'Non-Trauma',
                'event_type': 'resource_use',
                'event': 'registration_begins',
                'time': self.env.now,
                'resource_id': registration_resource.id_attribute
            })
            reg_duration = self.scenario.reg_dist.sample()
            yield self.env.timeout(reg_duration)
            self.event_log.append({
                'patient': self.pid,
                'pathway': 'Non-Trauma',
                'event_type': 'resource_use_end',
                'event': 'registration_complete',
                'time': self.env.now,
                'duration': reg_duration,
                'resource_id': registration_resource.id_attribute
            })
            self.scenario.registration.put(registration_resource)

            # Examination Phase
            exam_wait_start = self.env.now
            self.event_log.append({
                'patient': self.pid,
                'pathway': 'Non-Trauma',
                'event_type': 'queue',
                'event': 'examination_wait_begins',
                'time': self.env.now
            })
            exam_resource = yield self.scenario.exam.get()
            self.examination_wait = self.env.now - exam_wait_start
            self.event_log.append({
                'patient': self.pid,
                'pathway': 'Non-Trauma',
                'event_type': 'resource_use',
                'event': 'examination_begins',
                'time': self.env.now,
                'resource_id': exam_resource.id_attribute
            })
            exam_duration = self.scenario.exam_dist.sample()
            yield self.env.timeout(exam_duration)
            self.event_log.append({
                'patient': self.pid,
                'pathway': 'Non-Trauma',
                'event_type': 'resource_use_end',
                'event': 'examination_complete',
                'time': self.env.now,
                'duration': exam_duration,
                'resource_id': exam_resource.id_attribute
            })
            self.scenario.exam.put(exam_resource)
        # If trauma, additional events could be logged here.

        # 3. ED Stay Phase (Common to All)
        self.event_log.append({
            'patient': self.pid,
            'pathway': 'Unified',
            'event_type': 'queue',
            'event': 'ed_wait_begins',
            'time': self.env.now
        })
        ed_resource = yield ed_store.get()
        self.event_log.append({
            'patient': self.pid,
            'pathway': 'Unified',
            'event_type': 'resource_use',
            'event': 'ed_admit',
            'time': self.env.now,
            'resource_id': ed_resource.id_attribute
        })
        ed_duration = self.scenario.ed_stay_dist.sample()
        yield self.env.timeout(ed_duration)
        self.event_log.append({
            'patient': self.pid,
            'pathway': 'Unified',
            'event_type': 'resource_use_end',
            'event': 'ed_discharge',
            'time': self.env.now,
            'duration': ed_duration,
            'resource_id': ed_resource.id_attribute
        })
        ed_store.put(ed_resource)

        # 4. Decision Phase – ICU / MedSurg / Direct Discharge
        if self.scenario.icu_prob_dist.sample():
            # Record ICU wait start time
            icu_wait_start = self.env.now
            self.event_log.append({
                'patient': self.pid,
                'pathway': 'Unified',
                'event_type': 'queue',
                'event': 'icu_wait_begins',
                'time': self.env.now
            })
            icu_resource = yield icu_store.get()
            self.icu_wait = self.env.now - icu_wait_start
            self.event_log.append({
                'patient': self.pid,
                'pathway': 'Unified',
                'event_type': 'resource_use',
                'event': 'icu_admit',
                'time': self.env.now,
                'resource_id': icu_resource.id_attribute
            })
            if self.scenario.icu_proc_prob_dist.sample():
                self.event_log.append({
                    'patient': self.pid,
                    'pathway': 'Unified',
                    'event_type': 'resource_use',
                    'event': 'icu_procedure_start',
                    'time': self.env.now,
                    'resource_id': icu_resource.id_attribute
                })
                proc_duration = self.scenario.icu_proc_dist.sample()
                yield self.env.timeout(proc_duration)
                self.event_log.append({
                    'patient': self.pid,
                    'pathway': 'Unified',
                    'event_type': 'resource_use_end',
                    'event': 'icu_procedure_end',
                    'time': self.env.now,
                    'duration': proc_duration,
                    'resource_id': icu_resource.id_attribute
                })
            icu_duration = self.scenario.icu_stay_dist.sample()
            yield self.env.timeout(icu_duration)
            self.event_log.append({
                'patient': self.pid,
                'pathway': 'Unified',
                'event_type': 'resource_use_end',
                'event': 'icu_discharge',
                'time': self.env.now,
                'duration': icu_duration,
                'resource_id': icu_resource.id_attribute
            })
            icu_store.put(icu_resource)
        elif self.scenario.medsurg_prob_dist.sample():
            # Record Medsurg wait start time
            medsurg_wait_start = self.env.now
            self.event_log.append({
                'patient': self.pid,
                'pathway': 'Unified',
                'event_type': 'queue',
                'event': 'medsurg_wait_begins',
                'time': self.env.now
            })
            medsurg_resource = yield medsurg_store.get()
            self.medsurg_wait = self.env.now - medsurg_wait_start
            self.event_log.append({
                'patient': self.pid,
                'pathway': 'Unified',
                'event_type': 'resource_use',
                'event': 'medsurg_admit',
                'time': self.env.now,
                'resource_id': medsurg_resource.id_attribute
            })
            medsurg_duration = self.scenario.medsurg_stay_dist.sample()
            yield self.env.timeout(medsurg_duration)
            self.event_log.append({
                'patient': self.pid,
                'pathway': 'Unified',
                'event_type': 'resource_use_end',
                'event': 'medsurg_discharge',
                'time': self.env.now,
                'duration': medsurg_duration,
                'resource_id': medsurg_resource.id_attribute
            })
            medsurg_store.put(medsurg_resource)
        else:
            self.event_log.append({
                'patient': self.pid,
                'pathway': 'Unified',
                'event_type': 'transfer',
                'event': 'no_transfer',
                'time': self.env.now
            })

        # 5. Discharge
        self.event_log.append({
            'patient': self.pid,
            'pathway': 'Unified',
            'event_type': 'arrival_departure',
            'event': 'depart',
            'time': self.env.now,
            'resource_id': 0
        })
        self.total_time = self.env.now - self.arrival

    def execute(self, triage_store, ed_store, icu_store, medsurg_store):
        yield from self.process(triage_store, ed_store, icu_store, medsurg_store)

# -----------------------------
# WardFlowModel Class (Refactored)
# -----------------------------
class WardFlowModel:
    def __init__(self, scenario, start_datetime):
        self.env = simpy.Environment()
        self.scenario = scenario
        self.start_datetime = start_datetime

        self.utilisation_audit = []  # Initialize audit log list

        # Resources: using simpy.Store with CustomResource objects.
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
            
        self.event_log = []
        self.patient_count = 0
        self.patients = []
        
        # Additional resources for non-trauma pathway.
        self.scenario.registration = simpy.Store(self.env)
        for i in range(getattr(self.scenario, 'n_reg', 1)):
            self.scenario.registration.put(CustomResource(self.env, capacity=1, id_attribute=i+1))
        
        self.scenario.exam = simpy.Store(self.env)
        for i in range(getattr(self.scenario, 'n_exam', 3)):
            self.scenario.exam.put(CustomResource(self.env, capacity=1, id_attribute=i+1))

    def audit_utilisation(self, interval=1):
        while True:
            record = {
                'time': self.env.now,
                'triage_available': len(self.triage.items),
                'ed_beds_available': len(self.ed_beds.items),
                'icu_beds_available': len(self.icu_beds.items),
                'medsurg_beds_available': len(self.medsurg_beds.items),
                'registration_available': len(self.scenario.registration.items),
                'exam_available': len(self.scenario.exam.items)
            }
            self.utilisation_audit.append(record)
            yield self.env.timeout(interval)

    def arrivals_generator(self):
        for pid in itertools.count(1):
            if self.env.now >= self.scenario.simulation_time:
                break
            iat = np.random.exponential(5.0)
            yield self.env.timeout(iat)
            self.patient_count = pid
            patient = PatientFlow(pid, self.env, self.scenario, self.event_log, self.start_datetime)
            self.patients.append(patient)
            self.env.process(patient.execute(self.triage, self.ed_beds, self.icu_beds, self.medsurg_beds))

    def run(self, results_collection_period):
        # Start the utilisation audit process.
        self.env.process(self.audit_utilisation(interval=1))
        # Start generating arrivals.
        self.env.process(self.arrivals_generator())
        self.env.run(until=results_collection_period)
       

# -----------------------------
# SimulationSummary Class
# -----------------------------
class SimulationSummary:
    def __init__(self, model):
        self.model = model
        self.args = model.scenario
        self.results = None
        self.full_event_log = model.event_log
        self.patient_log = None

    def process_run_results(self):
        self.results = {}
        # Build a patient log with the new metrics:
        self.patient_log = []
        for p in self.model.patients:
            self.patient_log.append({
                'pid': p.pid,
                'arrival': p.arrival,
                'total_time': p.total_time,
                'triage_wait': getattr(p, 'triage_wait', np.nan),
                'registration_wait': getattr(p, 'registration_wait', np.nan),
                'examination_wait': getattr(p, 'examination_wait', np.nan),
                'icu_wait': getattr(p, 'icu_wait', np.nan),
                'medsurg_wait': getattr(p, 'medsurg_wait', np.nan)
            })
        
        patients = self.model.patients
        total_times = np.array([p.total_time for p in patients if p.total_time is not None])
        self.results['00_arrivals'] = len(patients)
        
        # Compute mean waits (if recorded)
        triage_waits = np.array([entry['triage_wait'] for entry in self.patient_log if not np.isnan(entry['triage_wait'])])
        registration_waits = np.array([entry['registration_wait'] for entry in self.patient_log if not np.isnan(entry['registration_wait'])])
        examination_waits = np.array([entry['examination_wait'] for entry in self.patient_log if not np.isnan(entry['examination_wait'])])
        icu_waits = np.array([entry['icu_wait'] for entry in self.patient_log if not np.isnan(entry['icu_wait'])])
        medsurg_waits = np.array([entry['medsurg_wait'] for entry in self.patient_log if not np.isnan(entry['medsurg_wait'])])
        
        self.results['01a_triage_wait'] = np.mean(triage_waits) if triage_waits.size > 0 else np.nan
        self.results['02a_registration_wait'] = np.mean(registration_waits) if registration_waits.size > 0 else np.nan
        self.results['03a_examination_wait'] = np.mean(examination_waits) if examination_waits.size > 0 else np.nan
        self.results['04a_ICU_wait'] = np.mean(icu_waits) if icu_waits.size > 0 else np.nan
        self.results['05a_Medsurg_wait'] = np.mean(medsurg_waits) if medsurg_waits.size > 0 else np.nan

        self.results['08_total_time'] = np.mean(total_times) if total_times.size > 0 else np.nan
        self.results['09_throughput'] = len([p for p in patients if p.total_time is not None])

        # Compute resource utilisation averages from the audit log:
        audit_df = pd.DataFrame(self.model.utilisation_audit)
        if not audit_df.empty:
            self.results['01b_triage_util'] = audit_df['triage_available'].mean() / self.args.n_triage
            self.results['02b_registration_util'] = audit_df['registration_available'].mean() / self.args.n_reg
            self.results['03b_examination_util'] = audit_df['exam_available'].mean() / self.args.n_exam
            self.results['04b_ICU_util'] = audit_df['icu_beds_available'].mean() / self.args.n_icu_beds
            self.results['05b_Medsurg_util'] = audit_df['medsurg_beds_available'].mean() / self.args.n_medsurg_beds
        else:
            self.results['01b_triage_util'] = np.nan
            self.results['02b_registration_util'] = np.nan
            self.results['03b_examination_util'] = np.nan
            self.results['04b_ICU_util'] = np.nan
            self.results['05b_Medsurg_util'] = np.nan

    def summary_frame(self):
        if self.results is None:
            self.process_run_results()
        df = pd.DataFrame({'1': self.results}).T
        df.index.name = 'rep'
        return df

    def detailed_logs(self):
        return {
            'full_event_log': pd.DataFrame(self.full_event_log),
            'patient_log': pd.DataFrame(self.patient_log),
            'utilisation_audit': pd.DataFrame(self.model.utilisation_audit),
            'summary_df': self.summary_frame()
        }

# -----------------------------
# Single Run and Multiple Replications Functions
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
                'patient_log': pd.DataFrame(summary_obj.patient_log),
                'utilisation_audit': pd.DataFrame(model.utilisation_audit),
            }
        }
        
    return summary_obj.summary_frame()

def multiple_replications(scenario, rc_period, n_reps, return_detailed_logs=False):
    outputs = []
    for rep in range(n_reps):
        out = single_run(scenario, rc_period, random_no_set=scenario.random_number_set + rep, return_detailed_logs=True)
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
# Main Execution Block
# -----------------------------
"""
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
        model="3-ward-flow"
    )
    df_summary = multiple_replications(scenario, rc_period=scenario.simulation_time, n_reps=5)
    print("Summary of Multiple Replications:")
    print(df_summary)
"""

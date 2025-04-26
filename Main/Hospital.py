import json
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
from scipy.stats import norm 
from modelclass import Scenario, CareUnit, NurseGroup, Patient, reactive_policy
from functions.helper import convert_mean_std_to_lognorm, load_empirical_los, load_transition_matrix, sanitize_transition_matrix, print_transition_matrix


USE_HARDCODED_PROBABILITIES = True  # Flag to control whether to use hardcoded probabilities
USE_QUEUES = True  # Flag to control whether to use queues (SimPy resource model)

# Define hardcoded probabilities for LOS and service transitions
LOS_PROBABILITIES = {
    "ED": [0.2, 0.3, 0.5],  # Probability distribution for LOS in ED
    "ICU": [0.1, 0.4, 0.5],  # Probability distribution for LOS in ICU
    "Ward": [0.3, 0.4, 0.3],  # Probability distribution for LOS in Ward
}

SERVICE_TRANSITIONS = {
    "ED": ["ICU", "Ward", "Discharge"],  # ED can go to ICU, Ward, or Discharge
    "ICU": ["Ward", "Discharge"],  # ICU can go to Ward or Discharge
    "Ward": ["Discharge"],  # Ward can only go to Discharge
}

# Simple hardcoded distribution for patient arrivals by severity
PATIENT_ARRIVAL_PROBABILITIES = [0.5, 0.3, 0.2]  # Severity 1, 2, 3 with respective probabilities


class HospitalFlowModel:

    # === 1. Initialization ===
    def __init__(self, scenario: Scenario, start_datetime=None, debug=False):
        self.scenario = scenario
        self.debug = debug
        self.start_datetime = start_datetime if start_datetime else datetime.datetime.now()
        self.env = simpy.Environment()

        # === PATIENT REGISTRY & EVENT LOGS ===
        self.n_total = 0  # Initialize n_total to track total patients
        self.pid_counter = 1
        self.patients = []
        self.completed_patients = []
        self.event_log = []
        self.utilisation_log = []
        self.raw_arrivals = 0
        self.active_boarding = []
        self.n_boarded = 0  # Initialize the boarded count
        self.total_boarding_time = 0  # Initialize total boarding time

        # === ARRIVAL TRACKING ===
        self.hourly_arrivals = [0] * 24
        self.daily_arrivals = defaultdict(int)

        # === LOS, WAIT TIMES, AND UNIT COUNTS ===
        self.unit_wait_totals = {"ED": 0.0, "ICU": 0.0, "Ward": 0.0, "Observation": 0.0}  # Add 'Observation'
        self.unit_counts = {"ED": 0, "ICU": 0, "Ward": 0, "Observation": 0}  # Add 'Observation'
        self.unit_los_totals = {"ED": 0.0, "ICU": 0.0, "Ward": 0.0, "Observation": 0.0}  # Add 'Observation'
        self.unit_avg_los = {"ED": [], "ICU": [], "Ward": [], "Observation": []}
        self.unit_avg_wait = {"ED": [], "ICU": [], "Ward": [], "Observation": []}


        # === CARE UNITS AND STAFFING ===
        self.transition_matrix = sanitize_transition_matrix(load_transition_matrix())
        self.ed = CareUnit(self.env, "ED", scenario.n_ed_beds)
        self.icu = CareUnit(self.env, "ICU", scenario.n_icu_beds)
        self.medsurg = CareUnit(self.env, "Ward", scenario.n_medsurg_beds)
        self.observation = CareUnit(self.env, "Observation", scenario.n_observation_beds)  # Add this line
        self.unit_capacities = {
            "ED": self.scenario.n_ed_beds,
            "ICU": self.scenario.n_icu_beds,
            "Ward": self.scenario.n_medsurg_beds,  # Make sure MedSurg is initialized correctly
            "Observation": self.scenario.n_observation_beds  # Ensure Observation is initialized here
        }


        self.nurse_day = NurseGroup(scenario.day_shift_nurses, name="DayNurses")
        self.nurse_night = NurseGroup(scenario.night_shift_nurses, name="NightNurses")
        self.nurses = simpy.Resource(self.env, capacity=scenario.day_shift_nurses)

        # === FLOW AND POLICY FLAGS ===
        self.exit_probs = scenario.exit_probs          # Optional: remove if unused

        self.overflow_allowed = False
        self.transfer_allowed = False
        self.discharge_acceleration = 0
        self.ambulance_diversion = False
        self.elective_deferral = 0
        self.icu_threshold = 0.5
        self.path_histogram = defaultdict(int)


        # === STATISTICS TRACKING ===
        self.lwbs_count = 0
        self.discharge_count = 0
        self.boarded_count = 0
        self.total_boarding_time = 0.0
        self.transferred_out = 0
        self.icu_admits = 0
        self.ward_admits = 0

        # === INITIAL UTILIZATION RECORD ===
        initial_audit = self.utilisation_audit(current_time=0.0)
        self.utilisation_log.append(initial_audit)

        # === BACKGROUND PROCESSES ===
        self.env.process(self.arrivals_generator())
        self.env.process(self.audit_util(interval=60))


    # === 2. Nurse & Severity Utilities ===
        

    @property
    def nurse_fatigue(self):
        """
        Compute composite fatigue for day and night shift nurses.
        """
        # Adjust for both day and night shifts
        total_base = self.nurse_day.num_nurses + self.nurse_night.num_nurses
        fatigue_day = self.nurse_day.fatigue_level
        fatigue_night = self.nurse_night.fatigue_level

        weighted_fatigue = (
            (fatigue_day * self.nurse_day.num_nurses) +
            (fatigue_night * self.nurse_night.num_nurses)
        ) / total_base if total_base > 0 else 1.0  # Avoid division by zero

        return weighted_fatigue


    def get_severity_multiplier(self, severity):
        return {1: 0.6, 2: 0.8, 3: 1.0, 4: 1.3, 5: 1.6}.get(severity, 1.0)
    

    def request_nurse(self):
        """Request a nurse resource, and optionally observe fatigue."""
        self.total_nurse_requests += 1

        if self.nurses.count >= self.nurses.capacity:
            self.overflow_events += 1

        # Log the fatigue level but DO NOT assign
        if self.debug:
            print(f"[DEBUG] Requesting nurse | Fatigue: {self.nurse_fatigue:.2f}")

        return self.nurses.request()



    # === 3. Patient Flow Routing ===

    def request_bed(self, unit_name):
        """
        Request a bed for a specific unit.
        """
        # Check if the unit exists in the unit capacities and count
        if unit_name in self.unit_capacities:
            # Check if the unit is available
            if unit_name == "ED":
                return self.ed.resource.request()
            elif unit_name == "ICU":
                return self.icu.resource.request()
            elif unit_name == "Ward":
                return self.medsurg.resource.request()
            elif unit_name == "Observation":
                return self.observation.resource.request()

        print(f"[WARN] Unknown unit capacity for {unit_name}.")
        return None




    def sample_next_unit(self, current_unit: str, severity: int):
        # This function handles transitions
        transitions = self.transition_matrix.get(severity, {}).get(current_unit, {})
        if not transitions:
            return "Discharge"  # Default to Discharge if no transitions found

        units, probs = zip(*transitions.items())
        r = random.random()
        cumulative = 0.0
        for unit, p in zip(units, probs):
            cumulative += p
            if r < cumulative:
                return unit
        return units[-1]



    # Updating try_transfer_or_board function for Observation
    def try_transfer_or_board(self, patient, target_unit):
        """
        Attempt to transfer patient to target unit.
        """
        if target_unit == "Discharge":
            self.log_event(patient, "discharge", unit=target_unit)
            return "discharge_bed"  # Treat discharge as an instant move

        bed = self.request_bed(target_unit)
        if bed is None:
            self.transferred_out += 1
            self.log_event(patient, f"transfer_out", unit=target_unit)
            return None

        # If overflow is allowed
        if target_unit == "ICU" and getattr(self, "overflow_to_ward", False):
            bed = self.request_bed("Ward")
            if bed:
                self.log_event(patient, "overflow_icu_to_ward")
                return bed

        if target_unit == "Ward" and getattr(self, "overflow_to_obs", False):
            bed = self.request_bed("Observation")
            if bed:
                self.log_event(patient, "overflow_ward_to_obs")
                return bed

        if target_unit == "Observation" and getattr(self, "overflow_to_obs", False):
            bed = self.request_bed("Observation")
            if bed:
                self.log_event(patient, "overflow_obs_to_medical")
                return bed

        return None










    def _handle_boarding(self, patient, target_unit: str):
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




    # === 4. Action System (DRL Integration) ===
    def apply_action(self, action):
        """
        Apply a DRL or heuristic action to the environment.
        Supports both:
        - Legacy 8-element vector
        - Modern Gym Dict with `strategic` and `tactical` keys
        """

        if isinstance(action, dict) and "strategic" in action and "tactical" in action:
            # === Gym Dict Action ===
            strat = action["strategic"]
            tact = action["tactical"]

            ms_adj      = int(np.round(strat[0]))  # MedSurg capacity adjustment
            day_nurse   = int(np.round(self.scenario.day_shift_nurses * strat[1]))
            night_nurse = int(np.round(self.scenario.night_shift_nurses * strat[2]))

            self.overflow_ed_to_ward       = bool(tact.get("overflow_flag", 0))
            self.overflow_icu_to_ward      = bool(tact.get("transfer_flag", 0))
            self.triage_threshold          = float(tact.get("triage_threshold", [0.0])[0])
            self.ambulance_diversion_rate  = float(tact.get("diversion_rate", [0.0])[0])

            self.day_shift_nurses   = max(1, day_nurse)
            self.night_shift_nurses = max(1, night_nurse)

            # Adjust MedSurg capacity
            self._adjust_capacity("Ward", ms_adj)

        else:
            # === Legacy Vector Action ===
            delta_nurse = action[0] - 3
            self.overflow_ed_to_ward = bool(action[1])
            self.overflow_icu_to_ward = bool(action[2])
            self.ambulance_diversion_rate = float(action[3])
            self.flex_ward_beds = action[4]
            self.triage_policy = action[5]
            self.admit_cap_per_hour = [5, 10, 15, 20][action[6]]
            self.icu_severity_threshold = [1, 2, 3][action[7]]

            self.day_shift_nurses = max(1, self.scenario.day_shift_nurses + delta_nurse)
            self.night_shift_nurses = max(1, self.scenario.night_shift_nurses + delta_nurse)

            new_ward_cap = self.scenario.n_medsurg_beds + self.flex_ward_beds
            self._set_ward_capacity(new_ward_cap)

        if self.debug:
            print("\n[APPLY ACTION]")
            print(f"Day/Night Nurses: {self.day_shift_nurses}/{self.night_shift_nurses}")
            print(f"Overflow Flags: ED→Ward={self.overflow_ed_to_ward}, ICU→Ward={self.overflow_icu_to_ward}")
            print(f"Triage Threshold: {self.triage_threshold:.2f}")
            print(f"Ambulance Diversion Rate: {self.ambulance_diversion_rate:.2f}")
            print(f"MedSurg Capacity: {self.medsurg.capacity} (ICU static at {self.icu.capacity})")


    
    def _adjust_capacity(self, unit_name, delta):
        if unit_name == "ICU":
            return  # ICU stays fixed
        elif unit_name == "Ward":
            target = self.medsurg
            base = self.scenario.n_medsurg_beds
        else:
            return

        new_cap = max(1, base + delta)
        if new_cap != target.capacity:
            print(f"[DEBUG] Resizing {unit_name} capacity from {target.capacity} to {new_cap}")
            target.capacity = new_cap
            target.resource = simpy.Resource(self.env, capacity=new_cap)


    def _set_ward_capacity(self, new_cap):
        new_cap = max(1, new_cap)
        if new_cap != self.medsurg.capacity:
            print(f"[DEBUG] Setting Ward capacity to {new_cap}")
            self.medsurg.capacity = new_cap
            self.medsurg.resource = simpy.Resource(self.env, capacity=new_cap)



    def step(self, action):
        # Reset shift-specific counters
        self.discharges_this_shift = 0
        self.lwbs_this_shift = 0
        self.overflow_events = 0
        self.total_nurse_requests = 0

        self.apply_action(action)
        next_shift = self.env.now + self.scenario.shift_length
        self.run(until=next_shift)

        obs = self.get_obs()
        reward_vector = self.calculate_multiobj_reward()

        done = self.env.now >= self.scenario.simulation_time


        self.nurse_day.reset()  # Reset fatigue for day shift nurses
        self.nurse_night.reset()  # Reset fatigue for night shift nurses

        info = self.get_patient_flow_summary()

        return obs, reward_vector, done, False, info

    # Reward Compute
    def calculate_multiobj_reward(self):
        throughput = float(self.discharge_count)
        wait_penalty = float(self.lwbs_count + self.boarded_count)
        fatigue_penalty = self.nurse_fatigue

        queue_penalty = (
            len(self.ed.queue) +
            len(self.icu.queue) +
            len(self.medsurg.queue)
        )

        utils = [u['ed_in_use'] for u in self.utilisation_log]
        avg_util = float(sum(utils) / max(len(utils), 1))

        return np.array([
            throughput,
            -wait_penalty,
            -fatigue_penalty,
            avg_util - 0.5 * queue_penalty
        ], dtype=np.float32)


    
    def compute_scalar_reward(self):
        """
        Scalar reward for PPO/DQN. Weighted combo of multi-objectives.
        """
        return (
            + 1.0 * self.discharge_count
            - 2.0 * self.lwbs_count
            - 1.0 * self.nurse_fatigue
            - 0.5 * len(self.active_boarding)
            - 0.3 * (len(self.ed.queue) + len(self.icu.queue) + len(self.medsurg.queue))
        )


    # Reward Compute
    def compute_reward(self):
        wait_penalty = 0.2 * self.avg_wait_time
        fatigue_penalty = 1.0 * self.nurse_fatigue
        overflow_penalty = 2 * self.overflow_events
        discharge_reward = 5 * self.discharges_this_shift
        lwbs_penalty = 10 * self.lwbs_this_shift

        return discharge_reward - lwbs_penalty - wait_penalty - fatigue_penalty - overflow_penalty
    

    # === 5. Arrivals and Patient Generation ===

    #Arrivals Generator
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

                severity = self.scenario.sample_acuity()

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
                pat.unit_path = ["ED"]

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



    # === 6. Length of Stay and Treatment Modeling ===
            
    # Samples LOS based on Acuity
    def sample_los(self, unit: str, severity: int) -> float:
        """
        Draw a log‑normal LOS for the given unit and severity.
        Scales by severity multiplier, nurse fatigue, and discharge acceleration (if non-ED).
        Applies bounds and debug logging.
        """
        try:
            if unit in self.scenario.los_params and severity in self.scenario.los_params[unit]:
                mu, sigma = self.scenario.los_params[unit][severity]
            else:
                mu, sigma = self.fallback_lognorm_params.get(severity, (3.8, 1.2))  # example fallback
            base_los = np.random.lognormal(mu, sigma)
        except (KeyError, IndexError):
            print(f"[WARN] Invalid severity={severity} for unit={unit} — using fallback LOS.")
            base_los = np.random.lognormal(2.0, 0.5)  # Fallback parameters

        severity_mult = self.get_severity_multiplier(severity)
        los = base_los * severity_mult * self.nurse_fatigue

        # Apply discharge acceleration for non-ED units
        if unit != "ED":
            accel_factor = max(0.5, 1.0 - 0.1 * self.discharge_acceleration)
            los *= accel_factor
        else:
            accel_factor = 1.0

        # Optional: floor LOS by unit type (e.g., minimum stay times)
        min_los_by_unit = {
            "Ward": 60,
            "ICU": 120,
            "Psych": 180,
            "ED": 30,
            "Observation": 45  # Define minimum LOS for Observation
        }
        los = max(los, min_los_by_unit.get(unit, 30))

        if self.debug:
            print(f"[LOS] Unit={unit}, Sev={severity}, base={base_los:.1f}, "
                f"× severity={severity_mult:.2f}, × fatigue={self.nurse_fatigue:.2f}, "
                f"× accel={accel_factor:.2f} → LOS={los:.1f} min")

        return los



    # Nurse Action Correlator
    def modify_los_with_fatigue(self, los):
        """Apply fatigue scaling and return bounded LOS."""
        fatigue_factor = 1.0 + min(1.0, self.nurse_fatigue * 0.05)  # scales LOS by up to 2x
        los *= fatigue_factor
        los = max(los, 30)  # keep minimum treatment time
        return los


    # === 7. Patient Pathways ===
    
    

    def handle_next_unit(self, patient, next_unit):
        if next_unit == "Discharge":
            yield from self.complete_discharge(patient, source=patient.current_unit)
        elif next_unit == "ICU":
            yield from self.process_icu_flow(patient)
        elif next_unit == "Ward":
            yield from self.process_ward_flow(patient)
        elif next_unit == "Observation":
            yield from self.process_observation_flow(patient)  # Handle Observation transition
        else:
            self.transferred_out += 1
            self.log_event(patient, "transfer_out", unit=patient.current_unit)
            yield self.env.timeout(0)




    # Handle Process Pathway
    def process_unit_flow(self, patient, unit_name, next_unit_options):
        """
        Generalized flow through ED, ICU, Ward, and Observation.
        - ED uses its own request/free pattern (no boarding to other units).
        - ICU/Ward/Observation can board if full.
        """
        patient.current_unit = unit_name
        patient.unit_path.append(unit_name)
        self.log_event(patient, f"{unit_name.lower()}_request")

        req_time = self.env.now

        # ==== 1) Occupy or board ==== 
        if unit_name == "ED":
            req = self.ed.resource.request()
            result = yield req | self.env.timeout(120)  # 2hr max wait
            if req not in result:
                # left without being seen
                self.lwbs_count += 1
                self.log_event(patient, "lwbs")
                return
            bed_req = req  # got an ED bed
        elif unit_name == "Observation":
            # Observation has a similar structure as ICU/Ward
            resource = self.observation  # Assuming the observation unit is properly initialized
            if resource.bed_available():
                bed_req = yield resource.occupy_bed()
            else:
                self.boarded_count += 1
                patient.boarding_start_time = self.env.now
                self.log_event(patient, "boarding", unit=unit_name)
                bed_req = yield resource.occupy_bed()
                patient.boarding_end_time = self.env.now
                self.total_boarding_time += (self.env.now - patient.boarding_start_time)
                self.log_event(patient, f"end_boarding_{unit_name}")
        else:  # ICU or Ward
            resource = self.icu if unit_name == "ICU" else self.medsurg
            if resource.bed_available():
                bed_req = yield resource.occupy_bed()
            else:
                # board on the same unit (not ED → Ward)
                self.boarded_count += 1
                patient.boarding_start_time = self.env.now
                self.log_event(patient, "boarding", unit=unit_name)
                bed_req = yield resource.occupy_bed()
                patient.boarding_end_time = self.env.now
                self.total_boarding_time += (self.env.now - patient.boarding_start_time)
                self.log_event(patient, f"end_boarding_{unit_name}")

        # ==== 2) Nurse + service ==== 
        yield self.request_nurse()
        
        print(f"Unit: {unit_name} | Current Wait Time: {self.unit_wait_totals[unit_name]} | Patient Count: {self.unit_counts[unit_name]}")
        self.unit_wait_totals[unit_name] += (self.env.now - req_time)
        self.unit_counts[unit_name] += 1

        self.log_event(patient, f"{unit_name.lower()}_admit", wait=self.env.now - req_time)

        # Length of Stay (LOS) based on the unit and patient severity
        los = self.modify_los_with_fatigue(self.sample_los(unit_name, patient.severity))
        yield self.env.timeout(los)  # Wait for length of stay

        # ==== 3) Free the bed ====
        if unit_name == "ED":
            self.ed.resource.release(bed_req)
        elif unit_name == "ICU":
            self.icu.free_bed(bed_req)
        elif unit_name == "Ward":
            self.medsurg.free_bed(bed_req)
        elif unit_name == "Observation":
            self.observation.free_bed(bed_req)

        self.log_event(patient, f"{unit_name.lower()}_discharge", los=los, time=self.env.now - patient.arrival_time)
        self.unit_los_totals[unit_name] += (self.env.now - req_time)

        # ==== 4) Next step or final discharge ====
        nxt = self.sample_next_unit(unit_name, patient.severity)
        if nxt not in next_unit_options:
            # forced out (no suitable next unit)
            self.transferred_out += 1
            self.log_event(patient, "transfer_out", from_unit=unit_name, to_unit=nxt)
            return

        if nxt == "Discharge":
            yield from self.complete_discharge(patient, source=unit_name)
        elif nxt == "ICU":
            yield from self.process_unit_flow(patient, "ICU", ["Ward", "Discharge"])
        elif nxt == "Ward":
            yield from self.process_unit_flow(patient, "Ward", ["Discharge"])
        elif nxt == "Observation":
            yield from self.process_unit_flow(patient, "Observation", ["Discharge"])



    #ED
    def process_ed_flow(self, patient):
        """
        Full patient flow through ED → ICU/Ward → Discharge.
        """
        patient.current_unit = "ED"
        self.log_event(patient, "ed_request")

        req_time = self.env.now
        next_unit = self.sample_next_unit("ED", patient.severity)

        # Update the wait time for ED
        print(f"Unit: {"ED"} | Current Wait Time: {self.unit_wait_totals["ED"]} | Patient Count: {self.unit_counts["ED"]}")
        self.unit_wait_totals["ED"] += (self.env.now - req_time)
        self.unit_counts["ED"] += 1

        if next_unit == "Discharge":
            self.log_event(patient, "ed_discharge")
            self.discharge_count += 1
            return

        # Handle transition to Observation from ED
        if next_unit == "Observation":
            yield from self.process_observation_flow(patient)

        # Try transferring to the next unit (ICU/Ward/Observation/Discharge)
        bed = self.try_transfer_or_board(patient, next_unit)

        if bed is None:
            # Handle overflow logic and boarding
            boarding_start = self.env.now
            max_boarding_time = 12 * 60  # 12 hours max boarding time
            self.log_event(patient, "boarding")

            while self.env.now - boarding_start < max_boarding_time:
                yield self.env.timeout(60)  # Retry every hour
                bed = self.try_transfer_or_board(patient, next_unit)
                if bed:
                    break

            if bed is None:
                # Forced discharge after waiting too long
                self.log_event(patient, "forced_discharge")
                self.discharge_count += 1
                return

        # Admit the patient to the next unit if a bed was found
        yield bed
        yield self.request_nurse()

        self.unit_counts[next_unit] += 1
        self.log_event(patient, f"{next_unit.lower()}_admit")

        unit_los = self.modify_los_with_fatigue(self.sample_los(next_unit, patient.severity))
        yield self.env.timeout(unit_los)  # Wait for length of stay

        # Free up the bed after treatment is completed
        yield self.free_bed(next_unit, bed)
        self.log_event(patient, f"{next_unit.lower()}_discharge")

        self.unit_los_totals[next_unit] += unit_los
        self.discharge_count += 1





    def process_observation_flow(self, patient):
        patient.current_unit = "Observation"
        self.log_event(patient, "observation_request")

        req_time = self.env.now
        next_unit = self.sample_next_unit("Observation", patient.severity)

        # Update wait time for Observation
        print(f"Unit: {"Observation"} | Current Wait Time: {self.unit_wait_totals["Observation"]} | Patient Count: {self.unit_counts["Observation"]}")
        self.unit_wait_totals["Observation"] += (self.env.now - req_time)
        self.unit_counts["Observation"] += 1

        if next_unit == "Discharge":
            self.log_event(patient, "observation_discharge")
            self.discharge_count += 1
            return

        bed = self.try_transfer_or_board(patient, next_unit)
        if bed is None:
            self.log_event(patient, "forced_discharge")
            self.discharge_count += 1
            return

        # Admit the patient to the next unit if a bed was found
        yield bed
        yield self.request_nurse()

        self.unit_counts[next_unit] += 1
        self.log_event(patient, f"{next_unit.lower()}_admit")

        unit_los = self.modify_los_with_fatigue(self.sample_los(next_unit, patient.severity))
        yield self.env.timeout(unit_los)  # Wait for length of stay

        # Free up the bed after treatment is completed
        yield self.free_bed(next_unit, bed)
        self.log_event(patient, f"{next_unit.lower()}_discharge")

        self.unit_los_totals[next_unit] += unit_los
        self.discharge_count += 1














    #ICU
    def process_icu_flow(self, patient):
        next_unit_options = ["Ward", "Discharge"]
        yield from self.process_unit_flow(patient, "ICU", next_unit_options)

    #WARD
    def process_ward_flow(self, patient):
        next_unit_options = ["Discharge"]
        yield from self.process_unit_flow(patient, "Ward", next_unit_options)


    def complete_discharge(self, patient: Patient, source="ward"):
        """
        Mark the patient as discharged, update logs and counters.
        """
        patient.status = "discharged"
        patient.current_unit = "Exit"
        patient.discharge_time = self.env.now
        patient.terminated = True
        self.discharge_count += 1

        # **Pathway Classification (for case severity)**
        if hasattr(patient, "unit_path"):
            if "ICU" in patient.unit_path:
                patient.case_type = "Severe"
            elif "Ward" in patient.unit_path:
                patient.case_type = "Moderate"
            else:
                patient.case_type = "Mild"

        # Log the discharge
        self.log_event(patient, event="discharge", unit=source)

        # Track completed patients
        self.completed_patients.append(patient)
        yield self.env.timeout(0)  # Asynchronous operation if necessary


   

    # === 8. Logging and Event History ===
        
    # Event Logger
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
        log.update(kwargs)  # Accept extra keyword arguments

        if hasattr(patient, "event_log"):
            patient.event_log.append(log)

        if self.debug:
            print(log)

        self.event_log.append(log)


        # Additional logs for patient transitions
        if event == "discharge":
            current_unit = getattr(patient, "current_unit", None)
            if current_unit and current_unit in self.unit_counts:
                self.unit_counts[current_unit] -= 1






    # === 9. Observations and Simulation Info ===
    
    # Get Patient Observations
    def get_obs(self):
        now = self.env.now

        # === 1. Unit Occupancy ===
        ed_in_use   = self.ed.beds_in_use()
        icu_in_use  = self.icu.beds_in_use()
        ward_in_use = self.medsurg.beds_in_use()

        # === 2. Queue Lengths ===
        ed_queue_len   = len(self.ed.queue)
        icu_queue_len  = len(self.icu.queue)
        ward_queue_len = len(self.medsurg.queue)

        # === 3. Boarding Load ===
        boarding_queue_len = len(self.active_boarding)

        # === 4. Arrivals & Fatigue ===
        arrivals_last_hour = sum(
            1 for p in self.patients if now - p.arrival_time <= 60
        )
        avg_arrival_rate = arrivals_last_hour / 60.0

        fatigue = np.clip(self.nurse_fatigue, 0.0, 10.0)
        hour_of_day = int((now // 60) % 24)

        # === 5. Severity Histogram (Sev 1-5) ===
        severity_counts = [0, 0, 0, 0, 0]
        for p in self.patients:
            if hasattr(p, "severity") and 1 <= p.severity <= 5:
                severity_counts[p.severity - 1] += 1

        # === 6. Construct Observation Vector ===
        obs_vector = np.array([
            ed_in_use,
            icu_in_use,
            ward_in_use,
            ed_queue_len,
            icu_queue_len,
            ward_queue_len,
            boarding_queue_len,
            int(self.overflow_allowed),
            int(self.ambulance_diversion),
            self.scenario.day_shift_nurses,
            self.medsurg.capacity,
            avg_arrival_rate,
            fatigue,
            hour_of_day,
            *severity_counts
        ], dtype=np.float32)

        # === 7. Debug Print ===
        if self.debug:
            print(f"[OBS DEBUG] ED: {ed_in_use} | ICU: {icu_in_use} | Ward: {ward_in_use}")
            print(f"[OBS DEBUG] Queues - ED: {ed_queue_len}, ICU: {icu_queue_len}, Ward: {ward_queue_len}")
            print(f"[OBS DEBUG] Active Boarding: {boarding_queue_len}")
            print(f"[OBS DEBUG] Fatigue: {fatigue:.2f}, Hour: {hour_of_day}")
            print(f"[OBS DEBUG] Severity Mix: {severity_counts}")

        return obs_vector



    #Get Patient Flow Summaries
    def get_patient_flow_summary(self):
        # Initialize unit_avg_los and unit_avg_wait
        summary = {
            'avg_boarding_time': self.total_boarding_time / self.n_boarded if self.n_boarded else 0,
            'avg_ed_wait': self.unit_wait_totals["ED"] / self.unit_counts["ED"] if self.unit_counts["ED"] else 0,
            'avg_icu_wait': self.unit_wait_totals["ICU"] / self.unit_counts["ICU"] if self.unit_counts["ICU"] else 0,
            'avg_ward_wait': self.unit_wait_totals["Ward"] / self.unit_counts["Ward"] if self.unit_counts["Ward"] else 0,
            'n_boarded': self.n_boarded,
            'n_discharged': self.discharge_count,
            'n_lwbs': self.lwbs_count,
            'n_total': self.n_total,
            'n_transferred': self.transferred_out,
            "unit_avg_los": {},  # Initialize unit_avg_los
            "unit_avg_wait": {}  # Initialize unit_avg_wait
        }

        durations = []
        boarding_times = []
        ed_waits = []
        icu_waits = []
        ward_waits = []

        for p in self.patients:
            if getattr(p, 'discharge_time', None) and p.arrival_time is not None:
                durations.append(p.discharge_time - p.arrival_time)
            if hasattr(p, 'boarding_start_time') and hasattr(p, 'boarding_end_time'):
                boarding_times.append(p.boarding_end_time - p.boarding_start_time)
            if hasattr(p, 'wait_time'):
                ed_waits.append(p.wait_time)
            if hasattr(p, 'ward_wait_time'):
                ward_waits.append(p.ward_wait_time)
            if hasattr(p, 'icu_wait_time'):
                icu_waits.append(p.icu_wait_time)

        # Aggregate totals
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

        # Overall queue wait
        all_waits = ed_waits + icu_waits + ward_waits
        if all_waits:
            summary["avg_wait_overall"] = float(np.mean(all_waits))

        # Add per-unit LOS and wait
        for unit in self.unit_counts:
            count = self.unit_counts[unit]
            if count > 0:
                summary["unit_avg_los"][unit] = self.unit_los_totals[unit] / count
                summary["unit_avg_wait"][unit] = self.unit_wait_totals[unit] / count

        # Print arrivals
        print("=== Arrival Counts ===")
        print(f"  raw_arrivals   : {self.raw_arrivals}")
        print(f"  admitted (n_total) : {len(self.patients)}")
        print(f"  discharged     : {self.discharge_count}")
        print(f"  left w/o svc   : {self.lwbs_count}")
        print(f"  transferred    : {self.transferred_out}")

        return summary







    # === 10. Audit & State Tracking ===  
    
    # Hospital Utilisation Audit
    def utilisation_audit(self, current_time=None):
         return {
            'time': current_time or self.env.now,
            'ed_in_use': self.ed.beds_in_use(),
            'icu_available': self.icu.available_beds(),
            'medsurg_available': self.medsurg.available_beds(),
            'nurses_available': self.nurses.capacity - len(self.nurses.queue) - self.nurses.count,
            'discharges': self.discharge_count,
            'lwbs': self.lwbs_count,
            'boarded': self.boarded_count,
            'transfers': self.transferred_out,
        }

    # A Background Process
    def audit_util(self, interval=60):
        while True:
            record = self.utilisation_audit()
            self.utilisation_log.append(record)
            yield self.env.timeout(interval)


    # Hospital Utilisation Audit
    def snapshot_state(self):
        """
        Returns a snapshot of the current simulation state to be used for front-end visualization 
        or by planners. Includes key metrics and logs.
        """
        return {
            "time": self.env.now,
            "patients": [vars(p) for p in self.patients],
            "ed_in_use": self.ed.beds_in_use(),
            "icu_occupancy": self.icu.beds_in_use(),
            "ward_occupancy": self.medsurg.beds_in_use(),
            "nurse_fatigue": self.nurse_fatigue,
            "discharge_count": self.discharge_count,
            "lwbs": self.lwbs_count,
            "boarded": self.boarded_count,
            "transfers": self.transferred_out,
            "event_log": self.event_log[-100:]
        }
    
    
    # === 11. Execution Control ===

    # Hospital Reset
    def reset(self):
        """
        Reinitializes the entire model simulation state.
        Must be called at the start of every episode.
        """
        self.__init__(self.scenario, start_datetime=self.start_datetime, debug=self.debug)
        return self.get_obs()
    
    #Hospital Runner
    def run(self, until):
        horizon = until if until is not None else self.scenario.simulation_time
        self.env.run(until=horizon)

        # Reset fatigue for each shift (end of each run)
        self.nurse_day.reset()
        self.nurse_night.reset()



# ------------------------------------------------------------------
# Debug runner – heuristic policy vs. no‑control baseline
# ------------------------------------------------------------------
if __name__ == "__main__":
    from pprint import pprint
    import datetime, time

    print("=== Hospital Simulation Debug Runner (30‑day test) ===")

    scenario = Scenario(
        simulation_time    = 30 * 24 * 60,
        n_ed_beds          = 1,
        n_icu_beds         = 1,
        n_medsurg_beds     = 1,
        n_observation_beds = 1,
        day_shift_nurses   = 2,
        night_shift_nurses = 5,
    )



    model = HospitalFlowModel(scenario, start_datetime=datetime.datetime.now(), debug=True)

    # 3) Run the heuristic policy, one shift at a time
    shift_len = scenario.shift_length     # 8 h ⇒ 480 min
    while model.env.now < scenario.simulation_time:
        # a) Observe
        obs = model.get_obs()

        # b) Choose action (reactive heuristic)
        action = {
            "strategic": np.array([0, 1.0, 1.0]),  # No change in MedSurg beds, normal staffing
            "tactical": {
                "overflow_flag": 1,
                "transfer_flag": 0,
                "triage_threshold": np.array([0.3]),
                "diversion_rate": np.array([0.1])
            }
        }

        # c) Apply action
        print("[DEBUG] Action type:", type(action))
        print("[DEBUG] Action keys:", action.keys() if isinstance(action, dict) else "Legacy vector")

        obs, reward, done, _, info = model.step(action)

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



    #print_transition_matrix(model.transition_matrix)

    # Pause so you can read the terminal
    time.sleep(2)
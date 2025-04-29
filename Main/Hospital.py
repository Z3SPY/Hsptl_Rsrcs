# hospitalflowmodel.py
from collections import defaultdict
import json
import simpy
import numpy as np
import random

from modelclass import Scenario, CareUnit, NurseGroup, Patient

class HospitalFlowModel:
    """
    Hospital patient flow environment (ED -> ICU -> Ward -> Discharge).
    Control actions adjust ward capacity, staffing, and diversion rate.
    """
    def __init__(self, scenario: Scenario, start_datetime=None, debug=False):
        self.scenario = scenario
        self.debug = debug
        self.env = simpy.Environment()

        # Initialize resources: ED, ICU, Ward
        self.ed = CareUnit(self.env, "ED", scenario.n_ed_beds)
        self.icu = CareUnit(self.env, "ICU", scenario.n_icu_beds)
        self.medsurg = CareUnit(self.env, "Ward", scenario.n_medsurg_beds)

        # Nurses (day shift resource)
        self.nurses = simpy.Resource(self.env, capacity=scenario.day_shift_nurses)
        self.day_shift_nurses = scenario.day_shift_nurses
        self.night_shift_nurses = scenario.night_shift_nurses
        self.nurse_day = NurseGroup(self.day_shift_nurses, name="Day")
        self.nurse_night = NurseGroup(self.night_shift_nurses, name="Night")

        # Statistics and tracking
        self.boarded_count = 0
        self.total_boarding_time = 0.0
        self.lwbs_count = 0
        self.discharge_count = 0
        self.transferred_out = 0
        self.raw_arrivals = 0

        self.unit_wait_totals = {"ED": 0.0, "ICU": 0.0, "Ward": 0.0}
        self.unit_los_totals = {"ED": 0.0, "ICU": 0.0, "Ward": 0.0}
        self.unit_counts = {"ED": 0, "ICU": 0, "Ward": 0}

        self.hourly_arrivals = defaultdict(int)
        self.daily_arrivals = defaultdict(int)

        self.patients = []
        self.event_log = []
        self.pid_counter = 1

        # Control parameters
        self.ambulance_diversion_rate = 0.0

        # Start arrival process
        self.env.process(self.arrivals_generator())

    def step(self, action):
        """
        Advance the simulation by one shift length with the given action.
        Returns: obs, reward_vector, done, info.
        """
        self.apply_action(action)
        # Run until end of shift
        next_shift = self.env.now + self.scenario.shift_length
        self.env.run(until=next_shift)

        obs = self.get_obs()
        reward_vec = self.calculate_multiobj_reward()
        done = self.env.now >= self.scenario.simulation_time
        info = self.get_patient_flow_summary()

        return obs, reward_vec, done, {}, info

    def apply_action(self, action):
        strat = action.get("strategic", None)
        tact = action.get("tactical", {})

        if strat is not None:
            # Adjust MedSurg (Ward) capacity
            ms_adj = int(np.round(strat[0]))
            new_ward_cap = max(1, self.scenario.n_medsurg_beds + ms_adj)

            if new_ward_cap != self.medsurg.capacity:
                self.medsurg.capacity = new_ward_cap
                self.medsurg.resource = simpy.Resource(self.env, capacity=new_ward_cap)

            # Adjust nurse staffing levels
            day_pct = strat[1]
            night_pct = strat[2]
            day_nurse = int(np.round(self.scenario.day_shift_nurses * day_pct))
            night_nurse = int(np.round(self.scenario.night_shift_nurses * night_pct))
            self.day_shift_nurses = max(1, day_nurse)
            self.night_shift_nurses = max(1, night_nurse)

            self.nurses = simpy.Resource(self.env, capacity=self.day_shift_nurses)

        # Tactical: Diversion
        self.ambulance_diversion_rate = float(tact.get("diversion_rate", [0.0])[0])



    def arrivals_generator(self):
        """
        Generate patient arrivals (non-homogeneous Poisson).
        """
        while self.env.now < self.scenario.simulation_time:
            hr = int((self.env.now // 60) % 24)
            lam = self.scenario.arrival_profile[hr]
            eff_lam = lam * (1.0 - self.ambulance_diversion_rate)
            n_arrivals = np.random.poisson(eff_lam)

            for _ in range(n_arrivals):
                delay = np.random.uniform(0, 60 / max(n_arrivals, 1))
                yield self.env.timeout(delay)
                severity = self.scenario.sample_acuity()
                # Create patient
                patient = Patient(pid=self.pid_counter,
                                  severity=severity,
                                  arrival_time=self.env.now)
                patient.unit_path = ["ED"]
                self.patients.append(patient)
                self.log_event(patient, "arrival")
                self.raw_arrivals += 1
                self.hourly_arrivals[int(self.env.now//60 % 24)] += 1
                self.daily_arrivals[int(self.env.now//1440)] += 1

                # Process ED flow for patient
                self.env.process(self.process_unit_flow(patient, "ED", ["ICU", "Ward", "Discharge"]))
                self.pid_counter += 1

            # Advance to next hour
            yield self.env.timeout(60 - (self.env.now % 60))

    def process_unit_flow(self, patient, unit_name, next_unit_options):
        """
        Generic flow through a unit:
        - Request bed (or board if ICU/Ward full)
        - Request nurse and treat (LOS)
        - Release bed and log
        - Transition to next unit or discharge
        """
        patient.current_unit = unit_name
        patient.unit_path.append(unit_name)
        self.log_event(patient, f"{unit_name.lower()}_request")
        req_time = self.env.now

        # 1) Occupy or board a bed
        if unit_name == "ED":
            req = self.ed.resource.request()
            result = yield req | self.env.timeout(120)  # max wait 2h
            if req not in result:
                # Left without service
                self.lwbs_count += 1
                self.log_event(patient, "lwbs")
                return
            bed_req = req
        else:
            # ICU or Ward
            resource = self.icu if unit_name == "ICU" else self.medsurg
            if resource.bed_available():
                bed_req = yield resource.occupy_bed()
            else:
                # Board patient
                self.boarded_count += 1
                patient.boarding_start_time = self.env.now
                self.log_event(patient, "boarding", unit=unit_name)
                bed_req = yield resource.occupy_bed()
                patient.boarding_end_time = self.env.now
                self.total_boarding_time += (self.env.now - patient.boarding_start_time)
                self.log_event(patient, f"end_boarding_{unit_name}")

        # 2) Nurse and treatment (LOS)
        yield self.request_nurse()
        wait_duration = self.env.now - req_time
        self.unit_wait_totals[unit_name] += wait_duration
        self.unit_counts[unit_name] += 1
        self.log_event(patient, f"{unit_name.lower()}_admit", wait=wait_duration)

        # Sample length of stay
        if unit_name == "ED":
            los = random.uniform(5, 15)  # short triage time
        elif unit_name == "ICU":
            los = random.expovariate(1/360.0)  # avg 360 min
        else:  # Ward
            los = random.expovariate(1/720.0)  # avg 720 min
        yield self.env.timeout(los)

        # 3) Release bed
        if unit_name == "ED":
            self.ed.resource.release(bed_req)
        elif unit_name == "ICU":
            self.icu.free_bed(bed_req)
        else:  # Ward
            self.medsurg.free_bed(bed_req)

        self.log_event(patient, f"{unit_name.lower()}_discharge", los=los,
                       time=self.env.now - patient.arrival_time)
        self.unit_los_totals[unit_name] += (self.env.now - req_time)

        # 4) Next step or discharge
        nxt = self._sample_next_unit(unit_name, patient.severity)
        if nxt not in next_unit_options:
            self.transferred_out += 1
            self.log_event(patient, "transfer_out", from_unit=unit_name, to_unit=nxt)
            return

        if nxt == "Discharge":
            self._complete_discharge(patient, source=unit_name)
        else:
            if nxt == "ICU":
                yield self.env.process(self.process_unit_flow(patient, "ICU", ["Ward", "Discharge"]))
            elif nxt == "Ward":
                yield self.env.process(self.process_unit_flow(patient, "Ward", ["Discharge"]))

    def request_nurse(self):
        """
        Request a nurse resource.
        """
        if self.nurses.count >= self.nurses.capacity:
            # No free nurse now; count as overflow if desired
            pass
        return self.nurses.request()

    def _sample_next_unit(self, current_unit, severity):
        """
        Simple transition logic based on current unit and severity.
        """
        if current_unit == "ED":
            if severity <= 2:
                r = random.random()
                if r < 0.6:
                    return "ICU"
                elif r < 0.9:
                    return "Ward"
                else:
                    return "Discharge"
            elif severity <= 4:
                r = random.random()
                if r < 0.3:
                    return "ICU"
                elif r < 0.8:
                    return "Ward"
                else:
                    return "Discharge"
            else:
                r = random.random()
                if r < 0.1:
                    return "ICU"
                elif r < 0.7:
                    return "Ward"
                else:
                    return "Discharge"
        elif current_unit == "ICU":
            return "Ward" if random.random() < 0.7 else "Discharge"
        elif current_unit == "Ward":
            return "Discharge"
        else:
            return "Discharge"

    def _complete_discharge(self, patient, source="Ward"):
        patient.status = "discharged"
        patient.current_unit = "Exit"
        patient.discharge_time = self.env.now
        self.discharge_count += 1
        self.log_event(patient, "discharge", unit=source)

    def log_event(self, patient, event, **kwargs):
        """
        Log a patient event (time, id, event, severity, unit, and extras).
        """
        log = {
            "time": self.env.now,
            "pid": patient.pid,
            "event": event,
            "severity": patient.severity,
            "unit": patient.current_unit
        }
        log.update(kwargs)
        patient.event_log.append(log)
        self.event_log.append(log)

    def calculate_multiobj_reward(self):
        """
        Multi-objective reward:
        [throughput, - (LWBS + boarded), - fatigue (ignored), - queue_length]
        """
        throughput = float(self.discharge_count)
        wait_penalty = float(self.lwbs_count + self.boarded_count)
        queue_penalty = len(self.ed.queue) + len(self.icu.queue) + len(self.medsurg.queue)
        return np.array([
            throughput,
            -wait_penalty,
            0.0,             # nurse fatigue (unused here)
            -queue_penalty
        ], dtype=np.float32)

    def get_obs(self):
        """
        Observation: [ED_in_use, ICU_in_use, Ward_in_use,
                      ED_queue, ICU_queue, Ward_queue,
                      boarded_count,
                      diversion_rate,
                      day_nurses, ward_capacity,
                      avg_arrival_rate, hour_of_day,
                      severity1_count, ..., severity5_count]
        """
        now = self.env.now
        ed_in_use = self.ed.beds_in_use()
        icu_in_use = self.icu.beds_in_use()
        ward_in_use = self.medsurg.beds_in_use()

        ed_queue = len(self.ed.queue)
        icu_queue = len(self.icu.queue)
        ward_queue = len(self.medsurg.queue)
        boarded_queue = self.boarded_count

        arrivals_last_hour = sum(1 for p in self.patients if now - p.arrival_time <= 60)
        avg_arrival_rate = arrivals_last_hour / 60.0

        hour_of_day = int((now // 60) % 24)
        day_nurses = float(self.day_shift_nurses)
        ward_cap = float(self.medsurg.capacity)

        severity_counts = [0]*5
        for p in self.patients:
            if 1 <= p.severity <= 5:
                severity_counts[p.severity - 1] += 1

        obs = np.array([
            ed_in_use,
            icu_in_use,
            ward_in_use,
            ed_queue,
            icu_queue,
            ward_queue,
            boarded_queue,
            self.ambulance_diversion_rate,
            day_nurses,
            ward_cap,
            avg_arrival_rate,
            hour_of_day,
            *severity_counts
        ], dtype=np.float32)
        return obs

    def get_patient_flow_summary(self):
        """
        Compute average waits and counts for reporting.
        """
        summary = {
            'avg_boarding_time': (self.total_boarding_time / self.boarded_count
                                  if self.boarded_count else 0.0),
            'avg_ed_wait': 0.0,
            'avg_icu_wait': 0.0,
            'avg_ward_wait': 0.0,
            'n_boarded': self.boarded_count,
            'n_discharged': self.discharge_count,
            'n_lwbs': self.lwbs_count,
            'n_total': len(self.patients),
            'n_transferred': self.transferred_out
        }

        ed_waits, icu_waits, ward_waits = [], [], []
        for p in self.patients:
            if p.current_unit == "Exit" and p.discharge_time is not None:
                if hasattr(p, 'ed_wait_time'):
                    ed_waits.append(p.ed_wait_time)
                if hasattr(p, 'icu_wait_time'):
                    icu_waits.append(p.icu_wait_time)
                if hasattr(p, 'ward_wait_time'):
                    ward_waits.append(p.ward_wait_time)

        if ed_waits:
            summary['avg_ed_wait'] = float(np.mean(ed_waits))
        if icu_waits:
            summary['avg_icu_wait'] = float(np.mean(icu_waits))
        if ward_waits:
            summary['avg_ward_wait'] = float(np.mean(ward_waits))

        all_waits = ed_waits + icu_waits + ward_waits
        summary['avg_wait_overall'] = float(np.mean(all_waits)) if all_waits else 0.0

        return summary

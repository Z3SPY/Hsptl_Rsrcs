# modelclass.py
import json
import simpy
import numpy as np
import random
from collections import defaultdict

class Scenario:
    def __init__(
        self,
        simulation_time: float = 60*24*7,
        shift_length: float = 8*60,
        n_ed_beds: int = 15,
        n_icu_beds: int = 4,
        n_medsurg_beds: int = 10,
        day_shift_nurses: int = 10,
        night_shift_nurses: int = 5,
        arrival_profile: list = None,
        severity_probs: list = None,
    ):
        # Simulation parameters
        self.simulation_time = simulation_time
        self.shift_length = shift_length
        self.n_ed_beds = n_ed_beds
        self.n_icu_beds = n_icu_beds
        self.n_medsurg_beds = n_medsurg_beds
        self.day_shift_nurses = day_shift_nurses
        self.night_shift_nurses = night_shift_nurses

        # Arrival and acuity distributions
        if arrival_profile is None:
            # Default: uniform arrival rate each hour
            self.arrival_profile = [n_ed_beds / 2.0 for _ in range(24)]
        else:
            self.arrival_profile = arrival_profile
        if severity_probs is None:
            # Default: equal probability of severities 1-5
            self.severity_probs = [0.2]*5
        else:
            self.severity_probs = severity_probs

    def sample_acuity(self) -> int:
        """
        Sample a patient severity level (1–5, where 1 is most severe).
        """
        return np.random.choice([1, 2, 3, 4, 5], p=self.severity_probs)

class NurseGroup:
    def __init__(self, num_nurses, shift_hours=8, name=""):
        self.num_nurses = num_nurses
        self.shift_hours = shift_hours
        self.fatigue_level = 0.0
        self.patients_attended = 0
        self.name = name

    def attend_patient(self):
        """Simulate a nurse attending a patient (increases fatigue)."""
        self.patients_attended += 1
        self.fatigue_level += 0.1
        if self.fatigue_level > 10:
            self.fatigue_level = 10

    def work(self):
        """Simulate a full shift (increases fatigue)."""
        self.fatigue_level += 0.2
        if self.fatigue_level > 10:
            self.fatigue_level = 10

    def reset(self):
        """Reset fatigue and counters after a shift."""
        self.fatigue_level = 0.0
        self.patients_attended = 0

    def is_available(self):
        return self.num_nurses > 0

class CareUnit:
    """
    Hospital unit (ED, ICU, or Ward) with finite bed capacity and queue.
    """
    def __init__(self, env, unit_name, capacity):
        self.env = env
        self.unit_name = unit_name
        self.capacity = capacity
        self.resource = simpy.Resource(self.env, capacity=self.capacity)

    def bed_available(self):
        return self.resource.count < self.capacity

    def beds_in_use(self):
        # Number of occupied beds
        return self.resource.count

    def available_beds(self):
        return self.capacity - self.resource.count

    def occupy_bed(self):
        if self.bed_available():
            return self.resource.request()
        return None

    def free_bed(self, req):
        req.release()

    @property
    def queue(self):
        return self.resource.queue

class Patient:
    def __init__(self, pid, severity, arrival_time):
        self.pid = pid
        self.severity = severity
        self.arrival_time = arrival_time
        self.status = "waiting"
        self.current_unit = None
        self.start_treatment = None
        self.end_treatment = None
        self.discharge_time = None
        self.unit_path = []
        self.event_log = []

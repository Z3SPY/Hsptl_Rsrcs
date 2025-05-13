import numpy as np

class EventLogger:
    def __init__(self):
        self.full_event_log = []

    def reset(self):
        self.full_event_log.clear()

    def log(self, patient_id, actor, action, event, timestamp):
        self.full_event_log.append((patient_id, actor, action, event, timestamp))


class FatigueManager:
    def __init__(self, step_size, base_rate=0.02, busy_rate=0.1, recovery_rate=0.2):
        self.step_size = step_size
        self.base_rate = base_rate
        self.busy_rate = busy_rate
        self.recovery_rate = recovery_rate

    def update(self, resources_by_unit, active_ids, resting_counts):
        for unit, resources in resources_by_unit.items():
            for res in resources:
                res.fatigue += self.base_rate * self.step_size
                if res.id_attribute in active_ids.get(unit, set()):
                    res.fatigue += self.busy_rate * self.step_size
                else:
                    res.fatigue -= self.recovery_rate * self.step_size
                res.fatigue = np.clip(res.fatigue, 0.0, 100.0)

    def reset(self):
        pass  # Optional: track fatigue state if needed

# hospital_env_helpers.py
# Helper modules for detailed logging, fatigue tracking, resource changes, and CSV exports

import csv
import os
import numpy as np
from datetime import datetime
from model_classes import CustomResource  # ensure CustomResource import for ResourceChangeManager

class EventLogger:
    """
    Records detailed event logs for patient pathways and resource usage.
    """
    def __init__(self, enable_console=False):
        self.full_event_log = []
        self.enable_console = enable_console

    def log(self, patient, pathway, event_type, event, time, resource_id=None, **kwargs):
        entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'patient': patient,
            'pathway': pathway,
            'event_type': event_type,
            'event': event,
            'time': time,
            'resource_id': resource_id
        }
        entry.update(kwargs)
        self.full_event_log.append(entry)
        if self.enable_console:
            print(f"LOG: {entry}")

    def export_csv(self, path='event_log.csv'):
        """
        Exports full_event_log to a CSV file.
        """
        if not self.full_event_log:
            return
        keys = list(self.full_event_log[0].keys())
        dirpath = os.path.dirname(path)
        if dirpath:
            os.makedirs(dirpath, exist_ok=True)
        with open(path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(self.full_event_log)

class FatigueManager:
    """
    Handles fatigue accumulation and recovery for resources.
    """
    def __init__(self, step_size, base_rate=0.02, busy_rate=0.1, recovery_rate=0.2):
        self.step_size = step_size
        self.base_rate = base_rate
        self.busy_rate = busy_rate
        self.recovery_rate = recovery_rate

    def update(self, resources, active_ids, resting_count):
        """
        resources: dict unit_name -> list of resource objects
        active_ids: dict unit_name -> set of active resource ids
        resting_count: dict unit_name -> int count
        """
        for unit, res_list in resources.items():
            for res in res_list:
                # Base fatigue accrual
                res.fatigue = min(100.0, res.fatigue + self.base_rate * self.step_size)
                # Busy bonus
                if res.id_attribute in active_ids.get(unit, set()):
                    res.fatigue = min(100.0, res.fatigue + self.busy_rate * self.step_size)
            # Recovery for resting staff could be added here

class ResourceChangeManager:
    """
    Queues resource add/remove operations with a delay.
    """
    def __init__(self, delay):
        self.delay = delay
        self.pending = []  # list of (execute_time, action, unit)

    def schedule(self, now, action, unit):
        """Schedule an 'add' or 'remove' action for a future time."""
        execute_time = now + self.delay
        self.pending.append((execute_time, action, unit))

    def apply_due(self, now, model_args):
        """Execute due actions on the model's resource stores."""
        still = []
        for exec_t, action, unit in self.pending:
            if exec_t <= now:
                store = getattr(model_args, unit)
                if action == 'add':
                    new_res = CustomResource(store._env, capacity=1, id_attribute=None)
                    store.put(new_res)
                elif action == 'remove' and hasattr(store, 'items') and store.items:
                    store.items.pop(0)
            else:
                still.append((exec_t, action, unit))
        self.pending = still

class CSVShiftLogger:
    """
    Logs shift-level metrics to a CSV file.
    """
    def __init__(self, path='shift_metrics.csv', header=None):
        self.path = path
        default_header = [
            'shift_idx', 'timesteps', 'throughput', 'avg_wait',
            'avg_fatigue', 'queue_penalty', 'fatigue_penalty', 'reward'
        ]
        self.header = header or default_header
        dirpath = os.path.dirname(self.path)
        if dirpath:
            os.makedirs(dirpath, exist_ok=True)
        if not os.path.exists(self.path):
            with open(self.path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(self.header)
        self.shift_idx = 0

    def log(self, timesteps, metrics):
        """Append a row with the provided metrics dict."""
        row = [
            self.shift_idx,
            timesteps,
            metrics.get('throughput', 0),
            metrics.get('avg_wait', 0.0),
            metrics.get('avg_fatigue', 0.0),
            metrics.get('queue_penalty', 0.0),
            metrics.get('fatigue_penalty', 0.0),
            metrics.get('reward', 0.0)
        ]
        with open(self.path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row)
        self.shift_idx += 1

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


def convert_mean_std_to_lognorm(mean, std):
    if mean <= 0 or std <= 0:
        return None
    variance = std ** 2
    sigma2 = np.log(1 + variance / mean**2)
    mu = np.log(mean) - 0.5 * sigma2
    sigma = np.sqrt(sigma2)
    return mu, sigma

def load_empirical_los(filepath="DataAnalysis/empirical_params.json"):
    with open(filepath, "r") as f:
        data = json.load(f)

    caregroup_los = data["caregroup_los"]
    los_params = {}

    for acuity_str, unit_stats in caregroup_los.items():
        acuity = int(acuity_str)
        for unit_name, stats in unit_stats.items():
            mean, std = stats["mean"], stats["std"]
            lognorm = convert_mean_std_to_lognorm(mean, std)
            if lognorm:
                if unit_name not in los_params:
                    los_params[unit_name] = {}
                los_params[unit_name][acuity] = lognorm

    return los_params  # dict[unit][acuity] = (mu, sigma)



def load_transition_matrix(path="DataAnalysis/graph/P_within_stay_per_acuity.csv"):
    df = pd.read_csv(path)
    tmatrix = {}
    for _, row in df.iterrows():
        sev = int(row["severity"])
        from_u = row["from_unit"]
        to_u = row["to_unit"]
        prob = float(row["probability"])
        tmatrix.setdefault(sev, {}).setdefault(from_u, {})[to_u] = prob
    return tmatrix

ALLOWED_UNITS = {"ED", "ICU", "Ward", "MedSurg", "Discharge"}


def print_transition_matrix(matrix):
    for severity, from_units in matrix.items():
        print(f"\nSeverity {severity}:")
        for from_unit, to_units in from_units.items():
            print(f"  From {from_unit}:")
            for to_unit, prob in to_units.items():
                print(f"    → {to_unit}: {prob:.3f}")

# Example transition matrix adjustment for ED → Discharge
def sanitize_transition_matrix(raw_matrix):
    UNIT_MAP = {
        "Emergency":   "ED",         # Changed to 'ED' for consistency
        "ICU":         "ICU",
        "Medical Ward": "MedSurg",   # 'Medical Ward' mapped to 'MedSurg'
        "Ward":        "MedSurg",    # 'Ward' mapped to 'MedSurg'
        "Discharge":   "Discharge",
        "Observation": "Observation",  # Observation added here
    }

    clean_matrix = {}

    # Loop through each severity level's transition mappings
    for sev, mapping in raw_matrix.items():
        clean_matrix[sev] = {}

        # Loop through each "from" unit's transition probabilities
        for from_raw, to_dict in mapping.items():
            # Map the "from" unit to standardized name
            from_unit = UNIT_MAP.get(from_raw, None)

            # If the unit name is not found in the UNIT_MAP, skip it
            if from_unit is None:
                print(f"[WARN] Skipping invalid 'from' unit: {from_raw}")
                continue

            cleaned = {}

            # Loop through each possible transition to another unit
            for to_raw, prob in to_dict.items():
                to_unit = UNIT_MAP.get(to_raw, None)

                # Skip invalid or self-loop transitions
                if to_unit is None:
                    print(f"[WARN] Skipping invalid 'to' unit: {to_raw}")
                    continue
                if to_unit == from_unit:
                    print(f"[WARN] Skipping invalid or self-loop transition: {from_unit} -> {to_unit}")
                    continue

                # Add valid transitions to cleaned dictionary
                cleaned[to_unit] = prob

            # Normalize transition probabilities
            if cleaned:
                total = sum(cleaned.values())
                normalized = {k: v / total for k, v in cleaned.items()}
                clean_matrix[sev][from_unit] = normalized

    return clean_matrix




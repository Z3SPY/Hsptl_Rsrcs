# run_hospital_simulation.py

from modelclass import Scenario
from Hospital import HospitalFlowModel
import numpy as np

if __name__ == "__main__":
    # Set up the scenario
    scenario = Scenario(
        simulation_time=5*24*60,  # 5 days
        n_ed_beds=10,
        n_icu_beds=4,
        n_medsurg_beds=12,
        day_shift_nurses=8,
        night_shift_nurses=5
    )

    # Initialize the hospital simulation
    hospital_env = HospitalFlowModel(scenario)

    # Dummy policy (no strategic adjustments for now)
    action = {
        "strategic": np.array([0, 1.0, 1.0]),  # no bed or nurse changes
        "tactical": {"diversion_rate": np.array([0.0])}  # no ambulance diversion
    }

    # Simulate until done
    done = False
    while not done:
        obs, reward, done, _, info = hospital_env.step(action)
        print(f"[Time {hospital_env.env.now/60:.1f} hr] Reward: {reward} Info: {info}")
    
    print("Simulation finished.")

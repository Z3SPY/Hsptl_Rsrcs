# random_agent_test.py
from hospital_env import HospitalSimEnv
import numpy as np

# Base simulation config
default_sim_config = {
    "n_triage": 2,
    "n_reg": 2,
    "n_exam": 3,
    "n_trauma": 2,
    "n_cubicles_1": 3,
    "n_cubicles_2": 2,
    "random_number_set": 1,
    "prob_trauma": 0.12,
}

# Create environment
env = HospitalSimEnv(sim_config=default_sim_config, step_size=60, rc_period=20160, cost_mode='diverse')

for ep in range(5):
    obs = env.reset()
    done = False
    total_reward = 0.0
    shift_counter = 0

    action = env.action_space.sample()

    while not done:
        obs, reward, done, _ = env.step(action)
        total_reward += reward

        # After stepping, check if new shift is needed
        if env.is_shift_planning:
            shift_counter += 1
            print(f"\n--- SHIFT {shift_counter} START ---")
            
            # Either sample random OR use MSO suggestion
            if np.random.rand() < 0.5:
                action = env.action_space.sample()
                print(f"Random allocation: {action}")
            else:
                action = env.mso_shift_planner_stub()
                print(f"MSO suggested allocation: {action}")


    print(f"\nRandom Episode {ep} finished: Total Reward = {total_reward:.2f}")

# rappo_hospital_train.py

import os
import numpy as np
import pandas as pd
from hospital_env import HospitalSimEnv
from RAPPO import RAPPOAgent

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

# Output CSV Setup
csv_file = "rappo_hospital_log.csv"
columns = [
    "timesteps",
    "sim_time_minutes",
    "avg_reward",
    "avg_triage_wait",
    "avg_registration_wait",
    "avg_exam_wait",
    "avg_trauma_wait",
    "avg_non_trauma_treat_wait",
    "avg_trauma_treat_wait",
    "avg_fatigue",
    "avg_throughput"
]

if not os.path.exists(csv_file):
    pd.DataFrame(columns=columns).to_csv(csv_file, index=False)

# Environment Setup
env = HospitalSimEnv(sim_config=default_sim_config, step_size=15, rc_period=40320, cost_mode='diverse')

# RAPPO Agent Setup
agent = RAPPOAgent(env, risk_alpha=0.1, beta=-2.0, clip_eps=0.2, lr=3e-4, gamma=0.99, lam=0.95)

# Training Parameters
total_timesteps = 200_000
n_steps_per_iter = 5000
episodes = total_timesteps // n_steps_per_iter

timestep_counter = 0

# Training Loop
for ep in range(episodes):
    obs = env.reset()
    done = False

    rewards, values, log_probs, actions, states, dones = [], [], [], [], [], []
    step_count = 0

    while not done and step_count < n_steps_per_iter:
        action, log_prob = agent.get_action(obs)
        next_obs, reward, done, _ = env.step(action)

        _, value = agent.model(torch.FloatTensor(obs).unsqueeze(0).to(agent.device))
        value = value.item()

        states.append(obs)
        actions.append(action)
        rewards.append(reward)
        values.append(value)
        log_probs.append(log_prob)
        dones.append(done)

        obs = next_obs
        step_count += 1

    # Finish batch
    _, last_value = agent.model(torch.FloatTensor(obs).unsqueeze(0).to(agent.device))
    returns = agent.compute_gae(rewards, values, dones, last_value.item())

    batch = {
        'obs': states,
        'actions': actions,
        'log_probs': log_probs,
        'returns': returns,
        'advantages': list(np.array(returns) - np.array(values))
    }
    agent.update(batch)

    timestep_counter += n_steps_per_iter

    # After each episode, log performance
    from model_classes import SimulationSummary
    summary = SimulationSummary(env.model)
    summary.process_run_results_live()

    new_row = {
        "timesteps": timestep_counter,
        "sim_time_minutes": env.current_time,
        "avg_reward": np.mean(rewards),
        "avg_triage_wait": summary.results.get('01a_triage_wait', np.nan),
        "avg_registration_wait": summary.results.get('02a_registration_wait', np.nan),
        "avg_exam_wait": summary.results.get('03a_examination_wait', np.nan),
        "avg_trauma_wait": summary.results.get('05a_stabilisation_wait(trauma)', np.nan),
        "avg_non_trauma_treat_wait": summary.results.get('04a_treatment_wait(non_trauma)', np.nan),
        "avg_trauma_treat_wait": summary.results.get('07a_treatment_wait(trauma)', np.nan),
        "avg_fatigue": np.mean([v for v in summary.fatigue_by_unit.values() if not np.isnan(v)]) if hasattr(summary, 'fatigue_by_unit') else np.nan,
        "avg_throughput": summary.results.get('09_throughput', np.nan)
    }

    # Save to CSV
    df = pd.read_csv(csv_file)
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    df.to_csv(csv_file, index=False)

    print(f"\n[Episode {ep+1}] Steps: {timestep_counter}, Avg Reward: {np.mean(rewards):.2f}")

print("\n✅ RAPPO Hospital Training Finished!")

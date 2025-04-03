# File: train.py (FINAL REFACTORED VERSION)

from modelclass import Scenario
from env_hospital import HospitalEnv
from sim_planner_env import SimPlannerEnv  # ✅ ADD THIS
from Agent import HybridAgent, PlannerControlCallback
from mcts_mpc_planner import MCTSMPCPlanner
from planner_mso import HospitalMSOPlanner
from modelclass import SimulationSummary
import torch
import numpy as np

def run_training_mcts_vs_mso(
    use_mcts=True,
    use_curriculum=False,
    total_timesteps=15000,
    mso_frequency_hours=8,
    eval_episodes=1,
    debug_logs=False
):
    print("\n===============================")
    print("Initial Setup: Creating Scenario")
    print("===============================")

    scenario = Scenario(
        simulation_time=120 * 60,
        random_number_set=42,
        n_triage=2,
        n_ed_beds=4,
        n_icu_beds=4,
        n_medsurg_beds=2
    )

    # Define a fresh environment maker for planners using SimPlannerEnv
    def env_maker():
        s = Scenario(simulation_time=120*60, random_number_set=999,
                     n_triage=2, n_ed_beds=4, n_icu_beds=4, n_medsurg_beds=2)
        return SimPlannerEnv(s)  # ✅ SWITCHED to SimPlannerEnv

    # 1) Temporary env to get PPO policy
    temp_env = env_maker()

    # 2) Instantiate DRL agent
    agent = HybridAgent(temp_env)

    # 3) Train DRL policy first
    print("\n===============================")
    print("Phase 1: DRL Pretraining")
    print("===============================")
    agent.train(timesteps=5000)

    # 4) Build the planner (MCTS or LP-MSO)
    print("\n===============================")
    print("Planner Setup: {}".format("MCTS-MPC" if use_mcts else "Linear MSO"))
    print("===============================")

    value_model = agent.get_value_function()

    if use_mcts:
        planner = MCTSMPCPlanner(env_maker, value_model,
                                 depth=3, branching=3, discount=0.99)
    else:
        planner = HospitalMSOPlanner(env_maker, value_model,
                                     depth=3, branching=3, discount=0.99)

    # 5) Final env with planner attached
    train_env = HospitalEnv(
        scenario=scenario,
        use_mso=True,
        mso_planner=planner,
        mso_frequency_hours=mso_frequency_hours,
        debug_logs=debug_logs
    )

    # 6) Curriculum toggle
    if use_curriculum:
        print("\n===============================")
        print("Curriculum Active: 5K DRL + 10K DRL+Planner")
        print("===============================")

          # 🧠 Add the callback that turns off MSO after 5000 PPO steps
        planner_callback = PlannerControlCallback(
            env=train_env,
            disable_after=5000,
            verbose=1
        )

        agent.env = train_env
        agent.model.learn(total_timesteps=10000, callback=planner_callback)

    else:
        print("\n===============================")
        print("Phase 2: DRL + Planner Override")
        print("===============================")
        agent.env = train_env
        agent.train(timesteps=total_timesteps)

    # 7) Final Evaluation
    print("\n===============================")
    print("Evaluation Episode")
    print("===============================")

    for ep in range(eval_episodes):
        obs = train_env.reset()
        done = False
        total_reward = 0
        step_count = 0
        while not done:
            action = agent.act(obs)
            print(f"[DEBUG] Step {step_count}, action => {action}")  # Manually checking

            obs, reward, done, info = train_env.step(action)
            total_reward += reward
            step_count += 1


        print(f"Episode {ep+1} Reward:", total_reward)
        if 'episode_summary' in info:
            print("Summary:", info['episode_summary'])

    # 8) Simulation Summary
    summary = SimulationSummary(train_env.model)
    summary.process_run_results()
    print("\nFinal Simulation Results:")
    print(summary.summary_frame())

if __name__ == "__main__":
    run_training_mcts_vs_mso(
        use_mcts=True,
        use_curriculum=True,
        total_timesteps=15000,
        debug_logs=False
    )
# File: train.py
from modelclass import Scenario, SimulationSummary
from env_hospital import HospitalEnv
from planner_mso import HospitalMSOPlanner
from Agent import HybridAgent

# Define simulation parameters (tune as needed)
scenario = Scenario(
    simulation_time=288 * 7,  # 1 day in minutes
    random_number_set=42,
    n_triage=10,
    n_ed_beds=10,
    n_icu_beds=10,
    n_medsurg_beds=2,
    day_shift_nurses=1,
    night_shift_nurses=1,
    shift_length=12,
    # Other service means and probabilities...
)

# Initialize MSO planner if using hybrid approach
total_beds = scenario.n_icu_beds + scenario.n_medsurg_beds
p_icu = getattr(scenario, 'p_icu', 0.3)
p_medsurg = getattr(scenario, 'p_medsurg', 0.4)
mean_icu_stay = getattr(scenario, 'icu_stay_mean', 8*60)
mean_medsurg_stay = getattr(scenario, 'medsurg_stay_mean', 6*60)

mso_planner = HospitalMSOPlanner(total_beds, p_icu, p_medsurg, mean_icu_stay, mean_medsurg_stay, horizon_hours=8)

# Create environment: toggle MSO usage via use_mso flag
use_mso = False  
env = HospitalEnv(scenario, mso_planner if use_mso else None, mso_frequency_hours=8)

# Initialize the DRL agent
agent = HybridAgent(env, use_mso=use_mso)

# Train the agent
agent.train(timesteps=20000)

# Evaluate the trained policy on a new simulation run
obs = env.reset()
done = False
while not done:
    action = agent.act(obs)
    obs, reward, done, info = env.step(action)
    # Optionally, log info per step or per episode here

# After simulation, print episode summary if available
if 'episode_summary' in info:
    print("Episode Summary:", info['episode_summary'])
    
# Process simulation summary if needed (using SimulationSummary from modelclass)
summary = SimulationSummary(env.model)
summary.process_run_results()
print(summary.summary_frame())

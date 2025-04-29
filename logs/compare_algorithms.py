import os
import pandas as pd
import matplotlib.pyplot as plt

# === Config ===
AGENTS = {
    "RAPPO": "C:/Users/Jabat/Desktop/SimHospital/logs/ra_ppo/run1.csv",
    "PPO": "C:/Users/Jabat/Desktop/SimHospital/logs/ppo_logged/run1.csv",
    "DQN": "C:/Users/Jabat/Desktop/SimHospital/logs/dqn_logged/run1.csv"
}

OUTPUT_DIR = "comparison_results_single"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# === Load Single CSV per Agent ===
agent_data = {}
for agent_name, csv_path in AGENTS.items():
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        agent_data[agent_name] = df
    else:
        print(f"⚠️ CSV not found for {agent_name}: {csv_path}")

# === Plotting ===
def plot_metric(metric_name, ylabel, save_name):
    plt.figure(figsize=(10,6))
    for agent_name, df in agent_data.items():
        if metric_name in df.columns:
            steps = df['iteration'] if 'iteration' in df.columns else df.index
            plt.plot(steps, df[metric_name], label=agent_name)
        else:
            print(f"⚠️ {metric_name} not found in {agent_name} data.")
    plt.xlabel("Training Iterations")
    plt.ylabel(ylabel)
    plt.title(f"{ylabel} Comparison")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(OUTPUT_DIR, save_name))
    plt.close()

# === Plot Important Metrics ===
plot_metric("avg_return", "Average Return", "avg_return_comparison.png")
plot_metric("policy_loss", "Policy Loss", "policy_loss_comparison.png")
plot_metric("value_loss", "Value Loss", "value_loss_comparison.png")
plot_metric("entropy", "Policy Entropy", "entropy_comparison.png")
plot_metric("steps_per_s", "Training Speed (Steps/sec)", "steps_per_sec_comparison.png")

# RAPPO specific
if "RAPPO" in agent_data and "cvar" in agent_data["RAPPO"].columns:
    plot_metric("cvar", "CVaR (Risk-Sensitive Return)", "cvar_comparison.png")

print("\n✅ All comparison plots saved in 'comparison_results_single/' folder.")

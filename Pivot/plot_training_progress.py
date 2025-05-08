import pandas as pd
import matplotlib.pyplot as plt

# Load your training log CSV
df = pd.read_csv("ppo_hospital_training_log.csv")

# Plot average reward per shift
plt.figure(figsize=(10, 5))
plt.plot(df["timesteps"], df["avg_reward"], label="Avg Reward per Shift", marker='o')
plt.title("PPO Hospital – Learning Progress")
plt.xlabel("Timesteps")
plt.ylabel("Average Reward")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

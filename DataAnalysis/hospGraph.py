import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("your_csv_file.csv")

# Plot Reward
plt.plot(df['timesteps'], df['avg_reward'])
plt.title("Average Reward Over Time")
plt.xlabel("Timesteps")
plt.ylabel("Average Reward")
plt.grid(True)
plt.show()

# Plot Throughput
plt.plot(df['timesteps'], df['avg_throughput'])
plt.title("Patient Throughput Over Time")
plt.xlabel("Timesteps")
plt.ylabel("Throughput")
plt.grid(True)
plt.show()

# Plot Fatigue
plt.plot(df['timesteps'], df['avg_fatigue'])
plt.title("Average Fatigue Over Time")
plt.xlabel("Timesteps")
plt.ylabel("Average Fatigue")
plt.grid(True)
plt.show()

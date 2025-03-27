import numpy as np
import gym
from gym import spaces
import copy
import matplotlib.pyplot as plt

# ------------------------------
# 1. Environment: Traffic Signal Control
# ------------------------------
class TrafficEnv(gym.Env):
    """
    A simplified traffic-signal control environment.
    - State: (q_N, q_E, q_S, q_W, phase, time_in_phase)
      where q_* are queue lengths (capped at max_queue),
      phase: 0 for NS-green (North & South), 1 for EW-green (East & West),
      and time_in_phase is the time elapsed in the current phase.
    - Action: Discrete {0: continue current phase, 1: switch phase}.
    - Dynamics: At each time step, vehicles in lanes with a green light are cleared
      (with a random capacity), and new vehicles arrive following a Poisson distribution.
    - Reward: Reward for cleared vehicles minus a penalty proportional to total queue length.
    """
    def __init__(self, max_queue=5, time_horizon=20, lambda_arrivals=(2,2,2,2)):
        super(TrafficEnv, self).__init__()
        self.max_queue = max_queue            # Maximum vehicles per lane
        self.time_horizon = time_horizon      # Episode length (in time steps)
        self.lambda_arrivals = lambda_arrivals  # Arrival rates for N, E, S, W
        self.phases = 2                       # Two phases: 0 = NS green, 1 = EW green
        
        # Observation: queues for 4 lanes, current phase, and time in phase.
        self.observation_space = spaces.MultiDiscrete([
            max_queue + 1, max_queue + 1, max_queue + 1, max_queue + 1,
            self.phases,
            time_horizon + 1
        ])
        # Actions: 0 (continue current phase) or 1 (switch phase)
        self.action_space = spaces.Discrete(2)
        self.reset()
    
    def reset(self):
        self.time_step = 0
        self.queues = [0, 0, 0, 0]  # Order: North, East, South, West
        self.phase = 0            # Start with NS-green
        self.time_in_phase = 0
        return self._get_state()
    
    def step(self, action):
        # Action: 0 = continue, 1 = switch phase
        if action == 1:
            self.phase = 1 - self.phase  # Toggle between 0 and 1
            self.time_in_phase = 0
        else:
            self.time_in_phase += 1
        
        # Clearing vehicles: only lanes with a green light clear vehicles.
        cleared = 0
        if self.phase == 0:  # NS-green: lanes 0 (North) and 2 (South)
            capacity_N = np.random.randint(3, 7)  # Random capacity between 3 and 6
            capacity_S = np.random.randint(3, 7)
            cleared_N = min(self.queues[0], capacity_N)
            cleared_S = min(self.queues[2], capacity_S)
            self.queues[0] -= cleared_N
            self.queues[2] -= cleared_S
            cleared = cleared_N + cleared_S
        else:  # EW-green: lanes 1 (East) and 3 (West)
            capacity_E = np.random.randint(3, 7)
            capacity_W = np.random.randint(3, 7)
            cleared_E = min(self.queues[1], capacity_E)
            cleared_W = min(self.queues[3], capacity_W)
            self.queues[1] -= cleared_E
            self.queues[3] -= cleared_W
            cleared = cleared_E + cleared_W
        
        total_cleared = cleared
        
        # New arrivals: Poisson arrivals for each lane.
        for i in range(4):
            arrivals = np.random.poisson(self.lambda_arrivals[i])
            self.queues[i] = min(self.queues[i] + arrivals, self.max_queue)
        
        total_queue = sum(self.queues)
        
        # Reward: Encourage clearing vehicles and penalize long queues.
        reward = total_cleared - 0.2 * total_queue
        
        self.time_step += 1
        done = self.time_step >= self.time_horizon
        info = {"total_queue": total_queue, "total_cleared": total_cleared}
        return self._get_state(), reward, done, info
    
    def _get_state(self):
        # Returns a tuple: (q_N, q_E, q_S, q_W, phase, time_in_phase)
        q = [min(q, self.max_queue) for q in self.queues]
        return (q[0], q[1], q[2], q[3], self.phase, self.time_in_phase)

# ------------------------------
# 2. DRL+MSO Integration using a Q-table
# ------------------------------
def initialize_Q(max_queue, phases, time_horizon, num_actions):
    # Q-table dimensions: for each lane's queue (0..max_queue) for 4 lanes,
    # phase (0 or 1), time_in_phase (0..time_horizon) and action.
    return np.zeros((max_queue+1, max_queue+1, max_queue+1, max_queue+1, phases, time_horizon+1, num_actions))

def update_Q(Q_table, state, action, reward, next_state, alpha, gamma, env):
    q_N, q_E, q_S, q_W, phase, time_in_phase = state
    next_q_N, next_q_E, next_q_S, next_q_W, next_phase, next_time_in_phase = next_state
    if next_time_in_phase >= env.time_horizon:
        max_next = 0
    else:
        max_next = np.max(Q_table[next_q_N, next_q_E, next_q_S, next_q_W, next_phase, next_time_in_phase, :])
    Q_table[q_N, q_E, q_S, q_W, phase, time_in_phase, action] += alpha * (
        reward + gamma * max_next - Q_table[q_N, q_E, q_S, q_W, phase, time_in_phase, action]
    )

def epsilon_greedy_action(Q_table, state, epsilon, num_actions):
    q_N, q_E, q_S, q_W, phase, time_in_phase = state
    if np.random.rand() < epsilon:
        return np.random.choice(range(num_actions))
    else:
        return np.argmax(Q_table[q_N, q_E, q_S, q_W, phase, time_in_phase, :])

def mso_planning(env, state, Q_table, horizon=5, num_scenarios=5, gamma=0.99):
    """
    For each candidate action (0 and 1), simulate num_scenarios rollouts (using deep copies of the environment)
    over a planning horizon. Use the Q-table to approximate terminal values.
    Return the action with the highest expected return.
    """
    num_actions = env.action_space.n
    best_action = None
    best_value = -np.inf
    
    for action in range(num_actions):
        scenario_values = []
        for _ in range(num_scenarios):
            # Clone the environment and set its state to the current state.
            env_sim = copy.deepcopy(env)
            q_N, q_E, q_S, q_W, phase, time_in_phase = state
            env_sim.queues = [q_N, q_E, q_S, q_W]
            env_sim.phase = phase
            env_sim.time_in_phase = time_in_phase
            
            new_state, reward, done, _ = env_sim.step(action)
            total_reward = reward
            discount = gamma
            current_state = new_state
            
            # Roll out for the remaining steps of the horizon.
            for _ in range(horizon - 1):
                if done:
                    break
                rollout_action = epsilon_greedy_action(Q_table, current_state, epsilon=0.0, num_actions=num_actions)
                new_state, reward, done, _ = env_sim.step(rollout_action)
                total_reward += discount * reward
                discount *= gamma
                current_state = new_state
            
            # Use terminal Q-value estimate if episode not done.
            if not done:
                q_values = Q_table[current_state[0], current_state[1], current_state[2],
                                   current_state[3], current_state[4], current_state[5], :]
                terminal_value = np.max(q_values)
            else:
                terminal_value = 0
            total_reward += discount * terminal_value
            scenario_values.append(total_reward)
        expected_value = np.mean(scenario_values)
        if expected_value > best_value:
            best_value = expected_value
            best_action = action
    return best_action

# ------------------------------
# 3. Training Loop and KPI Collection
# ------------------------------
# Set environment parameters for a manageable state space.
env = TrafficEnv(max_queue=5, time_horizon=20, lambda_arrivals=(2, 2, 2, 2))
max_queue = env.max_queue
time_horizon = env.time_horizon
phases = env.phases
num_actions = env.action_space.n

# Initialize the Q-table.
Q_table = initialize_Q(max_queue, phases, time_horizon, num_actions)

# Hyperparameters
num_episodes = 500
alpha = 0.1
gamma = 0.99
epsilon = 1.0
epsilon_min = 0.1
epsilon_decay = 0.995
mso_horizon = 5
mso_scenarios = 5

# KPI storage lists
episode_rewards = []
episode_avg_queue = []   # average queue length per episode
episode_throughput = []  # total cleared vehicles per episode

for ep in range(1, num_episodes + 1):
    state = env.reset()
    done = False
    total_reward = 0
    total_queue_accum = 0
    total_cleared_accum = 0
    steps = 0
    while not done:
        # Use DRL+MSO integration:
        if np.random.rand() < epsilon:
            action = np.random.choice(range(num_actions))
        else:
            action = mso_planning(env, state, Q_table, horizon=mso_horizon,
                                  num_scenarios=mso_scenarios, gamma=gamma)
        next_state, reward, done, info = env.step(action)
        total_reward += reward
        total_queue_accum += info["total_queue"]
        total_cleared_accum += info["total_cleared"]
        
        update_Q(Q_table, state, action, reward, next_state, alpha, gamma, env)
        state = next_state
        steps += 1
    episode_rewards.append(total_reward)
    episode_avg_queue.append(total_queue_accum / steps)
    episode_throughput.append(total_cleared_accum)
    
    if epsilon > epsilon_min:
        epsilon *= epsilon_decay
    
    if ep % 50 == 0:
        print(f"Episode {ep}, Total Reward: {total_reward:.2f}, "
              f"Avg Queue: {episode_avg_queue[-1]:.2f}, Throughput: {total_cleared_accum}, "
              f"Epsilon: {epsilon:.2f}")

# ------------------------------
# 4. Plotting Learning Curves and KPIs
# ------------------------------
episodes = np.arange(1, len(episode_rewards) + 1)
plt.figure(figsize=(12, 10))

plt.subplot(3, 1, 1)
plt.plot(episodes, episode_rewards, label='Total Reward per Episode')
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('Learning Curve (Reward)')
plt.legend()
plt.grid(True)

plt.subplot(3, 1, 2)
plt.plot(episodes, episode_avg_queue, label='Average Queue Length', color='orange')
plt.xlabel('Episode')
plt.ylabel('Avg Queue Length')
plt.title('Average Queue Length per Episode')
plt.legend()
plt.grid(True)

plt.subplot(3, 1, 3)
plt.plot(episodes, episode_throughput, label='Total Throughput (Cleared Vehicles)', color='green')
plt.xlabel('Episode')
plt.ylabel('Throughput')
plt.title('Throughput per Episode')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

print("Training completed.")
print(f"Total episodes trained: {len(episode_rewards)}")
print(f"Final Episode Reward: {episode_rewards[-1]:.2f}")
print(f"Final Average Queue Length: {episode_avg_queue[-1]:.2f}")
print(f"Final Episode Throughput: {episode_throughput[-1]}")

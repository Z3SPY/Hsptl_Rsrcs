import numpy as np
import gym
from gym import spaces
import copy
import matplotlib.pyplot as plt

# ===============================
# 1. Refined Supply Chain/Inventory Environment
# ===============================
class RefinedInventoryEnv(gym.Env):
    """
    A multi-period inventory management environment with:
      - A one-period lead time (orders placed are received in the next period).
      - Backorders for unmet demand.
      - Random supply disruptions: occasionally, only a fraction of the pending order is delivered.
      
    State: (inventory, pending_order, backorder, current_period)
      - inventory: units on-hand.
      - pending_order: units ordered in the previous period that will be delivered now.
      - backorder: unmet demand carried over.
      - current_period: current period index.
    
    Action: Order quantity (0 ... max_order)
    
    Dynamics:
      1. At the beginning of each period, the pending order (if any) is received.
         - With probability p_supply_disruption, only a fraction (disruption_factor) is received.
      2. The agent then chooses an order quantity which will be delivered next period.
      3. Demand is realized stochastically (with a base range plus occasional spikes).
      4. Available supply is used first to clear backorders, then to meet current demand.
      5. Unmet demand becomes new backorders.
      
    Costs and Revenue:
      - Revenue per unit sold.
      - Ordering cost per unit ordered.
      - Holding cost per unit of inventory at period end.
      - Backorder cost per unit of unmet demand.
      
    Reward = revenue - ordering_cost - holding_cost - backorder_cost
    """
    def __init__(self,
                 max_inventory=20,
                 max_order=10,
                 max_backorder=20,
                 T=15,
                 demand_low=3,
                 demand_high=8,
                 demand_spike_prob=0.2,
                 demand_spike_additional=10,
                 cost_order=2.0,
                 cost_hold=1.0,
                 cost_backorder=6.0,
                 price=15.0,
                 supply_disruption_prob=0.2,
                 disruption_factor=0.5,
                 seed=None):
        super(RefinedInventoryEnv, self).__init__()
        
        self.max_inventory = max_inventory      # on-hand inventory capacity
        self.max_order = max_order              # max order quantity per period
        self.max_backorder = max_backorder      # maximum backorder to track
        self.T = T                              # number of periods in an episode
        
        # Demand parameters
        self.demand_low = demand_low
        self.demand_high = demand_high
        self.demand_spike_prob = demand_spike_prob
        self.demand_spike_additional = demand_spike_additional
        
        # Cost and revenue parameters
        self.cost_order = cost_order
        self.cost_hold = cost_hold
        self.cost_backorder = cost_backorder
        self.price = price
        
        # Supply disruption parameters
        self.supply_disruption_prob = supply_disruption_prob
        self.disruption_factor = disruption_factor
        
        # Observation: (inventory, pending_order, backorder, current_period)
        self.observation_space = spaces.MultiDiscrete([
            max_inventory + 1,
            max_order + 1,
            max_backorder + 1,
            T + 1
        ])
        
        # Action: order quantity (0 to max_order)
        self.action_space = spaces.Discrete(max_order + 1)
        
        self.rng = np.random.default_rng(seed)
        self.reset()
    
    def reset(self):
        self.current_period = 0
        self.inventory = self.max_inventory // 2  # start with moderate inventory
        self.pending_order = 0   # no order pending at start
        self.backorder = 0       # no backorders initially
        return self._get_state()
    
    def _get_state(self):
        # Ensure values are within observation limits
        inv = min(self.inventory, self.max_inventory)
        pend = min(self.pending_order, self.max_order)
        back = min(self.backorder, self.max_backorder)
        period = self.current_period
        return (inv, pend, back, period)
    
    def step(self, action):
        # 1. Receive pending order from previous period, subject to supply disruption.
        received = self.pending_order
        if self.pending_order > 0 and self.rng.random() < self.supply_disruption_prob:
            received = int(self.pending_order * self.disruption_factor)
        self.inventory = min(self.inventory + received, self.max_inventory)
        # Reset pending order; new order will be placed below.
        self.pending_order = 0
        
        # 2. Place new order (action) that will be received next period.
        order_qty = action
        self.pending_order = order_qty  # will be received next period
        
        # 3. Realize Demand:
        base_demand = self.rng.integers(self.demand_low, self.demand_high + 1)
        if self.rng.random() < self.demand_spike_prob:
            demand = base_demand + self.demand_spike_additional
        else:
            demand = base_demand
        
        # 4. Fulfill Backorders first, then current demand.
        total_demand = self.backorder + demand
        sales = min(self.inventory, total_demand)
        self.inventory -= sales
        # Backorders are the remaining unmet demand.
        self.backorder = total_demand - sales
        
        # 5. Compute Revenue and Costs.
        revenue = sales * self.price
        ordering_cost = order_qty * self.cost_order
        holding_cost = self.inventory * self.cost_hold
        backorder_cost = self.backorder * self.cost_backorder
        
        reward = revenue - ordering_cost - holding_cost - backorder_cost
        
        # 6. Advance time.
        self.current_period += 1
        done = (self.current_period >= self.T)
        
        # Collect info for KPI logging.
        info = {
            "demand": demand,
            "sales": sales,
            "ordering_cost": ordering_cost,
            "holding_cost": holding_cost,
            "backorder_cost": backorder_cost,
            "inventory": self.inventory,
            "backorder": self.backorder
        }
        return self._get_state(), reward, done, info

# ===============================
# 2. DRL+MSO Integration Components (Using a Q-table)
# ===============================
def initialize_Q(max_inventory, max_order, max_backorder, T, num_actions):
    # Q-table shape: (inventory, pending_order, backorder, period, action)
    # pending_order is from 0 to max_order (since that's the max that can be ordered)
    return np.zeros((max_inventory + 1, max_order + 1, max_backorder + 1, T + 1, num_actions))

def update_Q(Q_table, state, action, reward, next_state, alpha, gamma, env):
    inv, pend, back, period = state
    next_inv, next_pend, next_back, next_period = next_state
    if next_period >= env.T:
        max_next = 0
    else:
        max_next = np.max(Q_table[next_inv, next_pend, next_back, next_period, :])
    Q_table[inv, pend, back, period, action] += alpha * (reward + gamma * max_next - Q_table[inv, pend, back, period, action])

def epsilon_greedy_action(Q_table, state, epsilon, num_actions):
    inv, pend, back, period = state
    if np.random.rand() < epsilon:
        return np.random.choice(range(num_actions))
    else:
        return np.argmax(Q_table[inv, pend, back, period, :])

def mso_planning(env, state, Q_table, horizon=3, num_scenarios=5, gamma=0.99):
    """
    For each candidate action, simulate num_scenarios rollouts from the current state
    using a planning horizon. Use the terminal Q-value as the terminal estimate.
    Return the action with the highest expected return.
    """
    num_actions = env.action_space.n
    best_action = None
    best_value = -np.inf
    
    for action in range(num_actions):
        scenario_values = []
        for _ in range(num_scenarios):
            # Clone the environment and force it to the current state.
            env_sim = copy.deepcopy(env)
            inv, pend, back, period = state
            env_sim.inventory = inv
            env_sim.pending_order = pend
            env_sim.backorder = back
            env_sim.current_period = period
            
            new_state, reward, done, _ = env_sim.step(action)
            total_reward = reward
            discount = gamma
            current_state = new_state
            
            # Roll out for the remaining horizon steps.
            for _ in range(horizon - 1):
                if done:
                    break
                rollout_action = epsilon_greedy_action(Q_table, current_state, epsilon=0.0, num_actions=num_actions)
                new_state, reward, done, _ = env_sim.step(rollout_action)
                total_reward += discount * reward
                discount *= gamma
                current_state = new_state
            
            # Terminal value estimation if not done.
            if not done:
                inv_t, pend_t, back_t, period_t = current_state
                terminal_value = np.max(Q_table[inv_t, pend_t, back_t, period_t, :])
            else:
                terminal_value = 0
            total_reward += discount * terminal_value
            scenario_values.append(total_reward)
        expected_value = np.mean(scenario_values)
        if expected_value > best_value:
            best_value = expected_value
            best_action = action
    return best_action

# ===============================
# 3. Training Loop with KPI Logging and Plotting
# ===============================
# Environment parameters.
env = RefinedInventoryEnv(max_inventory=20,
                          max_order=10,
                          max_backorder=20,
                          T=15,
                          demand_low=3,
                          demand_high=8,
                          demand_spike_prob=0.2,
                          demand_spike_additional=10,
                          cost_order=2.0,
                          cost_hold=1.0,
                          cost_backorder=6.0,
                          price=15.0,
                          supply_disruption_prob=0.2,
                          disruption_factor=0.5,
                          seed=42)

max_inventory = env.max_inventory
max_order = env.max_order
max_backorder = env.max_backorder
T = env.T
num_actions = env.action_space.n

# Initialize Q-table.
Q_table = initialize_Q(max_inventory, max_order, max_backorder, T, num_actions)

# Hyperparameters.
num_episodes = 5000
alpha = 0.1
gamma = 0.99
epsilon = 1.0
epsilon_min = 0.1
epsilon_decay = 0.995
mso_horizon = 3
mso_scenarios = 5

# KPI storage.
episode_rewards = []
episode_avg_inventory = []
episode_total_backorder = []

for ep in range(1, num_episodes + 1):
    state = env.reset()  # (inventory, pending_order, backorder, period)
    done = False
    total_reward = 0
    inv_accum = 0
    backorder_accum = 0
    steps = 0
    
    while not done:
        # Choose action: random exploration if under epsilon, otherwise use MSO planning.
        if np.random.rand() < epsilon:
            action = np.random.choice(range(num_actions))
        else:
            action = mso_planning(env, state, Q_table, horizon=mso_horizon,
                                  num_scenarios=mso_scenarios, gamma=gamma)
        next_state, reward, done, info = env.step(action)
        total_reward += reward
        inv_accum += info["inventory"]
        backorder_accum += info["backorder"]
        
        update_Q(Q_table, state, action, reward, next_state, alpha, gamma, env)
        state = next_state
        steps += 1
    
    episode_rewards.append(total_reward)
    episode_avg_inventory.append(inv_accum / steps)
    episode_total_backorder.append(backorder_accum)
    
    if epsilon > epsilon_min:
        epsilon *= epsilon_decay
    
    if ep % 50 == 0:
        print(f"Episode {ep}: Total Reward = {total_reward:.2f}, "
              f"Avg Inventory = {episode_avg_inventory[-1]:.2f}, "
              f"Total Backorder = {backorder_accum}, "
              f"Epsilon = {epsilon:.2f}")

# ===============================
# 4. Plot Learning Curves and KPIs
# ===============================
episodes = np.arange(1, len(episode_rewards) + 1)
plt.figure(figsize=(12, 10))

plt.subplot(3, 1, 1)
plt.plot(episodes, episode_rewards, label='Total Reward')
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('Learning Curve: Episode Reward')
plt.legend()
plt.grid(True)

plt.subplot(3, 1, 2)
plt.plot(episodes, episode_avg_inventory, label='Avg Inventory', color='orange')
plt.xlabel('Episode')
plt.ylabel('Avg Inventory')
plt.title('Average Inventory per Episode')
plt.legend()
plt.grid(True)

plt.subplot(3, 1, 3)
plt.plot(episodes, episode_total_backorder, label='Total Backorder', color='red')
plt.xlabel('Episode')
plt.ylabel('Total Backorder')
plt.title('Backorder Accumulation per Episode')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

print("Training completed.")
print(f"Total Episodes: {len(episode_rewards)}")
print(f"Final Episode Reward: {episode_rewards[-1]:.2f}")
print(f"Final Avg Inventory: {episode_avg_inventory[-1]:.2f}")
print(f"Final Total Backorder: {episode_total_backorder[-1]}")

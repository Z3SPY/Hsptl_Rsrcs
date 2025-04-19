import numpy as np
import matplotlib.pyplot as plt
from EnvCPole import CartPoleEnvWrapper
from AgntCPole import Agent

def train(agent_type='ppo', horizon=3, hybrid_prob=0.3, max_episodes=1000, window=20, threshold=195):
    env = CartPoleEnvWrapper(horizon=horizon,
                             use_mpc=(agent_type in ['mpc', 'hybrid']),
                             hybrid=(agent_type == 'hybrid'))
    agent = Agent(env.observation_space.shape[0], env.action_space.n) if agent_type != 'mpc' else None
    reward_history = []

    for episode in range(1, max_episodes + 1):
        try:
            state, _ = env.reset()
        except:
            state = env.reset()
        log_probs, rewards = [], []
        total_reward = 0
        done = False

        while not done:
            if agent_type == 'mpc':
                action = env.get_mpc_action(state)
                log_prob = None
            elif agent_type == 'hybrid' and np.random.rand() < hybrid_prob:
                action = env.get_mpc_action(state)
                log_prob = None
            else:
                if not isinstance(state, np.ndarray):
                    raise TypeError("State must be a NumPy array.")
                action, log_prob = agent.select_action(state)

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            if log_prob is not None:
                log_probs.append(log_prob)
                rewards.append(reward)

            state = next_state
            total_reward += reward

        if agent and log_probs:
            agent.update(rewards, log_probs)

        reward_history.append(total_reward)

        if episode % window == 0:
            avg = np.mean(reward_history[-window:])
            print(f"Episode {episode} | Avg Reward: {avg:.2f}")
            if avg >= threshold:
                print("Environment solved.")
                break

    plt.plot(reward_history, label=agent_type.upper())
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title(f"Training Curve - {agent_type.upper()}")
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    train(agent_type="hybrid")  # Change to 'ppo', 'mpc', or 'hybrid'

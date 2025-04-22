#!/usr/bin/env python3
"""
Runner for DQN LunarLander with logging and plotting.
"""
from DQN import DQNAgent

def main():
    # Instantiate and train the DQN agent—logging and plotting happen inside
    agent = DQNAgent()
    agent.train()
    print("✅ DQN training complete. Metrics and plots are in logs/dqn_logged/")

if __name__ == '__main__':
    main()

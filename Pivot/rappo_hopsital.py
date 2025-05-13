import random
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from RAPPO import RA_PPO_Agent as RA_PPO_Agent_Continuous
from hospital_env import HospitalSimEnv, DictToBoxAction
from model_classes import Scenario

SEED = 42
TOTAL_EPISODES = 200
BATCH_EPISODES = 8
LOG_DIR = "logs/rappo_tb"
MODEL_PATH = "rappo_hospital_model_tb.pth"

default_sim_config = {
    "n_triage": 2, "n_reg": 2, "n_exam": 4,
    "n_trauma": 3, "n_cubicles_1": 3, "n_cubicles_2": 3,
    "n_ward": 50, "n_icu": 50,
    "prob_trauma": 0.10,
    "rc_period": 1440 * 3,
    "random_number_set": 1
}

def make_env(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    sim_config = default_sim_config.copy()
    sim_config['n_triage'] = random.randint(1, 3)
    sim_config['n_reg'] = random.randint(1, 3)
    sim_config['n_exam'] = random.randint(2, 6)
    sim_config['n_trauma'] = random.randint(1, 4)
    sim_config['n_cubicles_1'] = random.randint(1, 4)
    sim_config['n_cubicles_2'] = random.randint(1, 4)

    scenario = Scenario(**sim_config)
    env = HospitalSimEnv(scenario, inject_resources=True)
    env = DictToBoxAction(env)
    env.seed(seed)
    return env

def compute_gae(rewards, values, gamma=0.99, lam=0.95):
    advs = []
    gae = 0
    values = values + [0]  # Bootstrap
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * values[t + 1] - values[t]
        gae = delta + gamma * lam * gae
        advs.insert(0, gae)
    return advs

def main():
    writer = SummaryWriter(LOG_DIR)
    agent = RA_PPO_Agent_Continuous(
        make_env(SEED),
        risk_alpha=0.05,
        beta=-1.0,
        clip_eps=0.2,
        lr=5e-4,
        gamma=0.99,
        lam=0.95
    )

    all_returns = []
    episode_idx = 0
    total_steps = 0
    MAX_STEPS = 1_000_000  # match PPO

    while total_steps < MAX_STEPS:
        batch_obs, batch_act, batch_logp, batch_ret, batch_adv = [], [], [], [], []
        batch_lens = []

        for _ in range(BATCH_EPISODES):
            env = make_env(SEED + episode_idx)
            obs, done = env.reset(), False
            rewards, values, logps, actions, states = [], [], [], [], []

            while not done:
                action, old_logp = agent.get_action(obs)
                next_obs, r, done, _ = env.step(action)
                _, _, value = agent.model(torch.FloatTensor(obs).to(agent.device))

                states.append(obs)
                actions.append(action)
                rewards.append(r)
                values.append(value.item())
                logps.append(old_logp)
                obs = next_obs

            episode_len = len(rewards)
            total_ret = sum(rewards)
            total_steps += episode_len
            all_returns.append(total_ret)

            risk_thr = np.quantile(all_returns[-100:], agent.risk_alpha)
            penalty = agent.beta if total_ret < risk_thr else 0
            shaped_rewards = [r + penalty for r in rewards]
            advs = compute_gae(shaped_rewards, values, gamma=0.99, lam=0.95)

            batch_obs.extend(states)
            batch_act.extend(actions)
            batch_logp.extend(logps)
            batch_ret.extend(advs)  # same as GAE returns
            batch_adv.extend(advs)
            batch_lens.append(episode_len)

            writer.add_scalar("episode/return", total_ret, episode_idx)
            writer.add_scalar("episode/risk_threshold", risk_thr, episode_idx)
            episode_idx += 1

        obs_t = torch.FloatTensor(batch_obs).to(agent.device)
        act_t = torch.FloatTensor(batch_act).to(agent.device)
        logp_t = torch.FloatTensor(batch_logp).to(agent.device)
        ret_t = torch.FloatTensor(batch_ret).to(agent.device)
        adv_t = torch.FloatTensor(batch_adv).to(agent.device)

        mean, std, values_pred = agent.model(obs_t)
        dist = torch.distributions.Normal(mean, std)
        log_probs = dist.log_prob(act_t).sum(axis=-1)
        entropy = dist.entropy().sum(axis=-1).mean()
        ratio = torch.exp(log_probs - logp_t)
        surr1 = ratio * adv_t
        surr2 = torch.clamp(ratio, 1.0 - agent.clip_eps, 1.0 + agent.clip_eps) * adv_t

        policy_loss = -torch.min(surr1, surr2).mean()
        value_loss = ((ret_t - values_pred) ** 2).mean()
        loss = policy_loss + 0.5 * value_loss - 0.01 * entropy

        agent.optimizer.zero_grad()
        loss.backward()
        agent.optimizer.step()

        writer.add_scalar("batch/loss_policy", policy_loss.item(), episode_idx)
        writer.add_scalar("batch/loss_value", value_loss.item(), episode_idx)
        writer.add_scalar("batch/entropy", entropy.item(), episode_idx)
        print(f"[EP {episode_idx:03d}] Steps={total_steps:,} | "
              f"PLoss={policy_loss.item():.4f}, VLoss={value_loss.item():.4f}, Entropy={entropy.item():.4f}")
        
        avg_return = np.mean(all_returns[-BATCH_EPISODES:])
        print(f"   ↳ Avg Return (last batch): {avg_return:.2f}")


    torch.save(agent.model.state_dict(), MODEL_PATH)
    writer.close()
    print(f"[DONE] Model saved to {MODEL_PATH} | Steps: {total_steps:,} | TensorBoard: {LOG_DIR}")


if __name__ == "__main__":
    main()

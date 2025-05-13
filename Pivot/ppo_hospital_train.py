import os
import random
import datetime
import numpy as np
import matplotlib.pyplot as plt
import gym

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, VecMonitor

from hospital_env import HospitalSimEnv, DictToBoxAction
from model_classes import Scenario

# ─── Global Configuration ─────────────────────────────────────────
SEED = 42
NUM_ENVS = 4
TOTAL_TIMESTEPS = 300_000

MODEL_PATH = "ppo_hospital_final_learnable.zip"
NORMALIZER_PATH = "vecnormalize_stats_learnable.pkl"

# Default simulation config for training (ward/ICU large to avoid bed effects)
default_sim_config = {
    "n_triage": 2, "n_reg": 2, "n_exam": 4,
    "n_trauma": 3, "n_cubicles_1": 3, "n_cubicles_2": 3,
    "n_ward": 50, "n_icu": 50,
    "prob_trauma": 0.10,
    "rc_period": 1440 * 3,   # 3-day episodes
    "random_number_set": 1
}

# ─── PPO Hyperparameters ───────────────────────────────────────────
ppo_kwargs = dict(
    policy="MlpPolicy",
    learning_rate=5e-4,
    n_steps=256,
    batch_size=64,
    n_epochs=5,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.1,
    vf_coef=0.5,
    verbose=1,
    tensorboard_log=f"./ppo_hospital_tensorboard/run_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}/",
    seed=SEED,
    device="cpu"
)

# ─── Environment Builder ───────────────────────────────────────────
def build_vec_env(num_envs, seed):
    """
    Creates a VecEnv where each episode randomizes base staffing and
    appends base_counts into the observation.
    """
    def make_env(rank):
        def _init():
            # 1) Randomize base staffing
            sim_config = default_sim_config.copy()
            sim_config['n_triage']     = random.randint(1, 3)
            sim_config['n_reg']        = random.randint(1, 3)
            sim_config['n_exam']       = random.randint(2, 6)
            sim_config['n_trauma']     = random.randint(1, 4)
            sim_config['n_cubicles_1'] = random.randint(1, 4)
            sim_config['n_cubicles_2'] = random.randint(1, 4)

            # 2) Base environment
            scenario = Scenario(**sim_config)
            env = HospitalSimEnv(scenario, inject_resources=True)

            # 3) Wrap the action
            wrapped = DictToBoxAction(env)
            wrapped.seed(seed + rank)

            # 4) Inject base_counts into observation
            from functools import wraps
            orig_get_obs = env._get_observation
            @wraps(orig_get_obs)
            def get_obs_with_base(*args, **kwargs):
                obs = orig_get_obs(*args, **kwargs)
                base = np.array([env.base_counts[u] for u, _ in env.resource_units], dtype=np.float32)
                return np.concatenate([obs, base])
            env._get_observation = get_obs_with_base

            # 5) Update observation_space for appended base dims
            orig_space = wrapped.observation_space
            base_low  = np.array([1,1,2,1,1,1], dtype=np.float32)
            base_high = np.array([3,3,6,4,4,4], dtype=np.float32)
            new_low  = np.concatenate([orig_space.low,  base_low])
            new_high = np.concatenate([orig_space.high, base_high])
            wrapped.observation_space = gym.spaces.Box(low=new_low, high=new_high, dtype=np.float32)

            return wrapped
        return _init

    env_fns = [make_env(i) for i in range(num_envs)]
    vec_env = DummyVecEnv(env_fns)

    # Remove stale normalizer stats
    if os.path.exists(NORMALIZER_PATH):
        os.remove(NORMALIZER_PATH)

    # Normalize observations (not rewards) and monitor
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=False, training=True)
    vec_env = VecMonitor(vec_env)
    return vec_env

# ─── Single-Env Builder for Fixed Base (for heatmap) ───────────────
def build_fixed_base_env(base_config):
    scenario = Scenario(**base_config)
    env = HospitalSimEnv(scenario, inject_resources=True)
    wrapped = DictToBoxAction(env)
    # Inject base_counts into observation
    from functools import wraps
    orig_get_obs = env._get_observation
    @wraps(orig_get_obs)
    def get_obs_with_base(*args, **kwargs):
        obs = orig_get_obs(*args, **kwargs)
        base = np.array([env.base_counts[u] for u, _ in env.resource_units], dtype=np.float32)
        return np.concatenate([obs, base])
    env._get_observation = get_obs_with_base
    # Update observation_space
    orig_space = wrapped.observation_space
    base_low  = np.array([1,1,2,1,1,1], dtype=np.float32)
    base_high = np.array([3,3,6,4,4,4], dtype=np.float32)
    new_low  = np.concatenate([orig_space.low,  base_low])
    new_high = np.concatenate([orig_space.high, base_high])
    wrapped.observation_space = gym.spaces.Box(low=new_low, high=new_high, dtype=np.float32)
    return wrapped

# ─── Evaluation Function ──────────────────────────────────────────
def evaluate_policy(env, model, n_episodes=5):
    rewards = []
    for _ in range(n_episodes):
        obs = env.reset()
        done, ep_rew = False, 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _ = env.step(action)
            ep_rew += reward
        rewards.append(ep_rew)
    return np.mean(rewards)

# ─── Heatmap Generator ────────────────────────────────────────────
def generate_policy_heatmap(model):
    # Sweep triage (1–3) vs. exam (2–6)
    triage_vals = np.arange(1,4)
    exam_vals   = np.arange(2,7)
    heatmap = np.zeros((len(exam_vals), len(triage_vals)))

    for i, base_exam in enumerate(exam_vals):
        for j, base_triage in enumerate(triage_vals):
            cfg = default_sim_config.copy()
            cfg['n_triage'] = int(base_triage)
            cfg['n_exam']   = int(base_exam)
            env = build_fixed_base_env(cfg)
            obs = env.reset()
            action, _ = model.predict(obs, deterministic=True)
            heatmap[i,j] = np.mean(action[:6])

    # Plot
    fig, ax = plt.subplots(figsize=(6,5))
    im = ax.imshow(
        heatmap, origin='lower',
        extent=[triage_vals[0]-0.5, triage_vals[-1]+0.5,
                exam_vals[0]-0.5,   exam_vals[-1]+0.5],
        aspect='auto'
    )
    ax.set_xlabel("Base Triage Nurses")
    ax.set_ylabel("Base Exam Bays")
    ax.set_title("Policy: Mean Extra Staff Δ by Base Configuration")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Mean Δ (extra staff)")
    plt.tight_layout()
    plt.show()

# ─── Main────────────────────────────────────────────────────────
def main():
    # 1) Training
    train_env = build_vec_env(NUM_ENVS, SEED)
    model     = PPO(env=train_env, **ppo_kwargs)
    print("[INFO] Starting PPO training...")
    model.learn(total_timesteps=TOTAL_TIMESTEPS, reset_num_timesteps=True)
    print("[INFO] Training complete. Saving model and normalizer...")
    model.save(MODEL_PATH)
    train_env.save(NORMALIZER_PATH)
    train_env.close()

    # 2) Evaluation
    eval_base = DummyVecEnv([lambda: DictToBoxAction(
        HospitalSimEnv(Scenario(**default_sim_config), inject_resources=True))])
    eval_env  = VecNormalize.load(NORMALIZER_PATH, eval_base)
    eval_env  = VecMonitor(eval_env)
    avg_rew   = evaluate_policy(eval_env, model, n_episodes=5)
    print(f"[EVAL] Avg reward over 5 episodes: {avg_rew:.2f}")

    # 3) Policy Heatmap
    generate_policy_heatmap(model)

if __name__ == "__main__":
    main()

import os
import numpy as np
from stable_baselines3 import PPO

from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize, VecMonitor

from hospital_env import HospitalSimEnv

# ─── Config ─────────────────────────────────────────────────────────
NUM_ENVS         = 4
SEED             = 42
TOTAL_TIMESTEPS  = 100_000
TB_LOG_DIR       = "./ppo_hospital_tensorboard/"
MODEL_PATH       = "ppo_hospital_final.zip"
NORMALIZER_PATH  = "vecnormalize_stats.pkl"

default_sim_config = {
    "n_triage": 2,
    "n_reg": 2,
    "n_exam": 3,
    "n_trauma": 2,
    "n_cubicles_1": 3,
    "n_cubicles_2": 2,
    "random_number_set": 1,
    "n_icu": 5,
    "n_ward": 10,
    "prob_trauma": 0.12,
}

# ─── 1) Environment builder ──────────────────────────────────────────
def build_vec_env(num_envs, seed):
    def make_env(rank):
        def _init():
            env = HospitalSimEnv(default_sim_config)
            env.seed(seed + rank)
            return env
        return _init

    # 1) create a SubprocVecEnv of raw envs
    env_fns = [make_env(i) for i in range(num_envs)]
    vec_env = SubprocVecEnv(env_fns)

    # 2) normalize observations & rewards during training
    vec_env = VecNormalize(
        vec_env,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
        clip_reward=10.0,
        training=True,    # must be True for learning
        epsilon=1e-8,
    )

    # 3) wrap in VecMonitor so ep_rew_mean is logged
    vec_env = VecMonitor(vec_env)

    return vec_env
# ─── 2) PPO hyperparameters for one-shift horizon ───────────────────
ppo_kwargs = dict(
    policy         = "MlpPolicy",
    learning_rate  = 1e-4,
    n_steps        = 512,     # shorter rollout
    batch_size     = 128,
    n_epochs       = 10,
    gamma          = 0.90,    # match 8h shift
    gae_lambda     = 0.92,
    clip_range     = 0.2,
    ent_coef       = 0.01,
    vf_coef        = 0.2,     # lower critic weight
    verbose        = 1,
    tensorboard_log= TB_LOG_DIR,
    seed           = SEED,
)

def main():
    # Build the training env
    train_env = build_vec_env(NUM_ENVS, SEED)

    # Instantiate the PPO model
    model = PPO(env=train_env, **ppo_kwargs)

    # Train
    model.learn(total_timesteps=TOTAL_TIMESTEPS, reset_num_timesteps=False)

    # Save model and normalizer
    model.save(MODEL_PATH)
    train_env.save(NORMALIZER_PATH)
    train_env.close()

    print(f"[DONE] Model saved to {MODEL_PATH}")
    print(f"[DONE] Normalizer stats saved to {NORMALIZER_PATH}")

if __name__ == "__main__":
    main()

# best_model_callback.py

from stable_baselines3.common.callbacks import BaseCallback
import os
import numpy as np

class SaveBestModelCallback(BaseCallback):
    """
    Callback for saving the best model based on training reward.
    """
    def __init__(self, save_path, check_freq=5000, verbose=1):
        super(SaveBestModelCallback, self).__init__(verbose)
        self.save_path = save_path
        self.check_freq = check_freq
        self.best_mean_reward = -float('inf')

    def _init_callback(self):
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        # Only check every check_freq steps
        if self.n_calls % self.check_freq == 0:
            # Get training reward
            if len(self.model.ep_info_buffer) > 0:
                mean_reward = np.mean([ep_info['r'] for ep_info in self.model.ep_info_buffer])
                if self.verbose > 0:
                    print(f"[CHECK] Step {self.num_timesteps}: Mean reward {mean_reward:.2f} | Best reward {self.best_mean_reward:.2f}")

                # New best model
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    path = os.path.join(self.save_path, 'best_model')
                    self.model.save(path)
                    if self.verbose > 0:
                        print(f"[SAVE] New best model at step {self.num_timesteps} with mean reward {mean_reward:.2f}")
        return True

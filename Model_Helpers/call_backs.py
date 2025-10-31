from stable_baselines3.common.callbacks import BaseCallback
import numpy as np

# -------------------------------
# Callbacks
# -------------------------------
class AchievementCallback(BaseCallback):
    """Log achievement statistics during training"""

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.achievements_history = []

    def _on_step(self) -> bool:
        if self.locals.get('dones'):
            for done, info in zip(self.locals['dones'], self.locals['infos']):
                if done and 'achievements' in info:
                    episode_unlocks = sum(1 for unlocked in info['achievements'].values() if unlocked)
                    self.achievements_history.append(episode_unlocks)
                    self.logger.record('achievements/per_episode', episode_unlocks)
                    if len(self.achievements_history) >= 100:
                        self.logger.record('achievements/mean_last_100', np.mean(self.achievements_history[-100:]))
        return True


class TensorBoardPrintCallback(BaseCallback):
    """Print TensorBoard metrics to console during training"""

    def __init__(self, print_freq: int = 100, verbose=0):
        super().__init__(verbose)
        self.print_freq = print_freq
        self.episode_rewards = []
        self.episode_lengths = []

    def _on_step(self) -> bool:
        # Collect episode statistics
        if self.locals.get('dones'):
            for idx, done in enumerate(self.locals['dones']):
                if done:
                    info = self.locals['infos'][idx]
                    if 'episode' in info:
                        self.episode_rewards.append(info['episode']['r'])
                        self.episode_lengths.append(info['episode']['l'])
        return True
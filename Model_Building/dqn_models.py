import argparse
import os
import json
from datetime import datetime
from typing import Dict, Any, Optional
import numpy as np
import torch
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback
from Model_Helpers.environments import make_env as make_crafter_env
from Model_Helpers.environments import make_shaped_env as make_shaped_crafter_env
from Model_Helpers.environments import framed_make_env as make_framed_crafter_env

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

# -------------------------------
# DQN Model Class
# -------------------------------
class DQN_Model:
    """Create and train DQN models for Crafter"""

    def __init__(self, make_env, log_path: str, model_path: Optional[str] = None,
                 total_timesteps: int = 100_000, config: Optional[Dict[str, Any]] = None,
                 print_freq: int = 100, model_type: str = 'baseline'):
        self.make_env = make_env
        self.log_path = log_path
        self.model_path = model_path
        self.total_timesteps = total_timesteps
        self.config = config or self.get_default_config()
        self.print_freq = print_freq
        self.model_type = model_type

        os.makedirs(self.log_path, exist_ok=True)
        if self.model_path:
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        self.save_config()

        print(f"\n{'='*70}")
        print("To view training in TensorBoard, run:")
        print(f"tensorboard --logdir {self.log_path}")
        print(f"{'='*70}\n")

    def get_default_config(self) -> Dict[str, Any]:
        return {
            'learning_rate': 1e-4,
            'gamma': 0.99,
            'train_freq': 4,
            'buffer_size': 100_000,
            'batch_size': 32,
            'target_update_interval': 10_000,
            'exploration_fraction': 0.1,
            'exploration_initial_eps': 1.0,
            'exploration_final_eps': 0.05,
            'learning_starts': 10_000,
            'gradient_steps': 1,
            'tau': 1.0,
        }

    def save_config(self):
        config_path = os.path.join(self.log_path, 'config.json')
        config_data = {
            'model_type': self.model_type,
            'total_timesteps': self.total_timesteps,
            'dqn_config': self.config,
            'timestamp': datetime.now().isoformat(),
            'model_path': self.model_path
        }
        with open(config_path, 'w') as f:
            json.dump(config_data, f, indent=4)
        print(f"✓ Configuration saved to: {config_path}")

    def create_model(self, env, model_name: str = "dqn") -> DQN:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"\n✓ Creating DQN model on device: {device}")
        obs_space = env.observation_space
        print(f"Observation space: {obs_space}")
        print(f"Action space: {env.action_space}")

        # Use CNN policy for image observations
        policy = "CnnPolicy"
        model = DQN(
            policy=policy,
            env=env,
            learning_rate=self.config['learning_rate'],
            gamma=self.config['gamma'],
            train_freq=self.config['train_freq'],
            buffer_size=self.config['buffer_size'],
            batch_size=self.config['batch_size'],
            target_update_interval=self.config['target_update_interval'],
            exploration_fraction=self.config['exploration_fraction'],
            exploration_initial_eps=self.config['exploration_initial_eps'],
            exploration_final_eps=self.config['exploration_final_eps'],
            learning_starts=self.config['learning_starts'],
            gradient_steps=self.config['gradient_steps'],
            tau=self.config['tau'],
            tensorboard_log=self.log_path,
            verbose=1,
            device=device
        )
        return model

    def create_callbacks(self, model_prefix: str):
        achievement_callback = AchievementCallback()
        tensorboard_print_callback = TensorBoardPrintCallback(print_freq=self.print_freq)
        return [achievement_callback, tensorboard_print_callback]

    # -------------------------------
    # Training Methods
    # -------------------------------
    def train_model(self, make_env_fn):
        env = make_env_fn()  # single environment, no DummyVecEnv
        model = self.create_model(env, model_name=self.model_type)
        callbacks = self.create_callbacks(self.model_type)

        try:
            model.learn(
                total_timesteps=self.total_timesteps,
                callback=callbacks,
                progress_bar=True,
                log_interval=100
            )

            if self.model_path:
                model.save(self.model_path)
                print(f"\n✓ Training complete! Model saved to: {self.model_path}.zip")
        except KeyboardInterrupt:
            print("\n⚠ Training interrupted!")
            if self.model_path:
                interrupted_path = self.model_path + "_interrupted"
                model.save(interrupted_path)
                print(f"✓ Interrupted model saved to: {interrupted_path}.zip")

        return model, env

    def baseline_dqn(self):
        return self.train_model(make_crafter_env)

    def reward_shaped_dqn(self):
        return self.train_model(make_shaped_crafter_env)

    def framed_dqn(self):
        return self.train_model(make_framed_crafter_env)

# -------------------------------
# Main script
# -------------------------------
def main():
    parser = argparse.ArgumentParser(description="Train DQN models for Crafter environment")
    parser.add_argument('--model_type', type=str, choices=['baseline', 'reward_shaped', 'framed'],
                        required=True, help='Type of model to train')
    parser.add_argument('--log_path', type=str, default='./Training/Logs/DQN/',
                        help='Base directory for logs')
    parser.add_argument('--model_path', type=str, default=None,
                        help='Path to save the trained model')
    parser.add_argument('--total_timesteps', type=int, default=1_000_000,
                        help='Total training timesteps')
    parser.add_argument('--print_freq', type=int, default=1000,
                        help='Print training stats every N timesteps')
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    model_name = f"{args.model_type}_dqn_{timestamp}"
    if args.model_path is None:
        args.model_path = os.path.join(args.log_path, "models", model_name)
    log_path = os.path.join(args.log_path, args.model_type, timestamp)

    # Create DQN trainer
    dqn_model = DQN_Model(
        make_env=None,  # will pass dynamically
        log_path=log_path,
        model_path=args.model_path,
        total_timesteps=args.total_timesteps,
        model_type=args.model_type
    )

    # Train the selected model
    if args.model_type == 'baseline':
        model, env = dqn_model.baseline_dqn()
    elif args.model_type == 'reward_shaped':
        model, env = dqn_model.reward_shaped_dqn()
    elif args.model_type == 'framed':
        model, env = dqn_model.framed_dqn()

    env.close()
    print(f"\n{'='*70}")
    print(f"✓ Training complete!")
    print(f"✓ Model saved to: {args.model_path}.zip")
    print(f"✓ Logs saved to: {log_path}")
    print(f"{'='*70}\n")

if __name__ == "__main__":
    main()
import argparse
import os
import json
from datetime import datetime
from typing import Dict, Any, Optional
import numpy as np
import torch
import torch.nn as nn
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import psutil
from Model_Helpers.environments import make_env as make_crafter_env
from Model_Helpers.environments import make_shaped_env as make_shaped_crafter_env
from Model_Helpers.environments import make_preprocessed_shaped_env as make_preprocessed_shaped_crafter_env


# -------------------------------
# Custom CNN for Single-Channel Images
# -------------------------------
class SingleChannelCNN(BaseFeaturesExtractor):
    """
    Custom CNN architecture for single-channel (grayscale) images.

    This is similar to NatureCNN but adapted for single-channel inputs.
    Architecture:
    - Conv2d(1, 32, kernel_size=8, stride=4)
    - Conv2d(32, 64, kernel_size=4, stride=2)
    - Conv2d(64, 64, kernel_size=3, stride=1)
    - Flatten -> Linear(features_dim)
    """

    def __init__(self, observation_space, features_dim: int = 512):
        super().__init__(observation_space, features_dim)

        # Extract input dimensions
        n_input_channels = observation_space.shape[2]  # Should be 1 for grayscale

        print(f"[CNN] Building custom CNN for single-channel images")
        print(f"[CNN] Input channels: {n_input_channels}")
        print(f"[CNN] Input shape: {observation_space.shape}")

        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with torch.no_grad():
            # Transpose to (batch, channels, height, width) format
            sample = torch.as_tensor(observation_space.sample()[None]).float()
            # Reshape from (1, H, W, C) to (1, C, H, W)
            sample = sample.permute(0, 3, 1, 2)
            n_flatten = self.cnn(sample).shape[1]

        print(f"[CNN] Flattened features: {n_flatten}")

        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU()
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # Input shape: (batch, height, width, channels)
        # Need: (batch, channels, height, width)
        observations = observations.permute(0, 3, 1, 2)
        return self.linear(self.cnn(observations))


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
        self.model_type = model_type

        # Use default config for all DQN variants
        if config is None:
            self.config = self.get_default_config()
        else:
            self.config = config

        self.print_freq = print_freq

        os.makedirs(self.log_path, exist_ok=True)
        if self.model_path:
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        self.save_config()

        print(f"\n{'=' * 70}")
        print("To view training in TensorBoard, run:")
        print(f"tensorboard --logdir {self.log_path}")
        print(f"{'=' * 70}\n")

    def get_default_config(self) -> Dict[str, Any]:
        """Default config for all DQN models"""
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

    def _estimate_buffer_memory(self, obs_shape, buffer_size):
        """Estimate replay buffer memory usage"""
        bytes_per_obs = np.prod(obs_shape)
        # Buffer stores: observations, next_observations, actions, rewards, dones
        # The big memory users are obs and next_obs (both same size)
        total_mb = (bytes_per_obs * buffer_size * 2) / (1024 * 1024)  # *2 for obs + next_obs
        return f"{total_mb:.1f} MB"

    def create_model(self, env, model_name: str = "dqn"):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"\n✓ Creating {model_name} model on device: {device}")
        obs_space = env.observation_space
        print(f"Observation space: {obs_space}")
        print(f"Action space: {env.action_space}")

        # Print memory-critical config
        print(f"\n[MEMORY CONFIG]")
        print(f"  Buffer size: {self.config['buffer_size']:,}")
        print(f"  Observation shape: {obs_space.shape}")
        estimated_mem = self._estimate_buffer_memory(obs_space.shape, self.config['buffer_size'])
        print(f"  Estimated buffer memory: {estimated_mem}")

        # Check if we have enough memory
        available_mem_gb = psutil.virtual_memory().available / (1024 ** 3)
        print(f"  Available system memory: {available_mem_gb:.2f} GB")

        # Test the environment to make sure it works
        print("\n[DEBUG] Testing environment reset and step...")
        try:
            test_obs, test_info = env.reset()
            print(f"[DEBUG] Reset successful! Obs type: {type(test_obs)}")

            # Handle LazyFrames
            if hasattr(test_obs, '__array__'):
                test_obs_array = np.array(test_obs)
                print(f"[DEBUG] Obs shape: {test_obs_array.shape}, dtype: {test_obs_array.dtype}")
            else:
                print(f"[DEBUG] Obs shape: {test_obs.shape}, dtype: {test_obs.dtype}")

            test_obs2, test_reward, test_term, test_trunc, test_info2 = env.step(env.action_space.sample())
            print(f"[DEBUG] Step successful!")
            print(f"[DEBUG] Environment test passed!\n")
        except Exception as e:
            print(f"[ERROR] Environment test failed: {e}")
            raise

        # Use CNN policy for image observations
        policy = "CnnPolicy"

        # Configure policy kwargs based on observation type
        policy_kwargs = {}

        # Check if this is a single-channel (grayscale) image
        is_single_channel = (obs_space.shape[2] == 1 and obs_space.dtype == np.float32)

        if is_single_channel:
            print(f"[INFO] Detected single-channel normalized observations")
            print(f"[INFO] Using custom SingleChannelCNN features extractor")

            # Use custom CNN for single-channel images
            policy_kwargs['features_extractor_class'] = SingleChannelCNN
            policy_kwargs['features_extractor_kwargs'] = dict(features_dim=512)
        elif obs_space.dtype == np.float32:
            # Multi-channel normalized images
            print(f"[INFO] Detected multi-channel normalized observations (float32)")
            print(f"[INFO] Setting normalized_image=True for standard NatureCNN")
            policy_kwargs['features_extractor_kwargs'] = dict(normalized_image=True)

        # Standard DQN for all variants
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
            device=device,
            policy_kwargs=policy_kwargs if policy_kwargs else None
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
        print(f"\n[INFO] Creating environment for {self.model_type}...")
        env = make_env_fn()
        print(f"[INFO] Environment created successfully")

        print(f"[INFO] Creating model...")
        model = self.create_model(env, model_name=self.model_type)
        print(f"[INFO] Model created successfully")

        callbacks = self.create_callbacks(self.model_type)

        try:
            print(f"\n[INFO] Starting training for {self.total_timesteps} timesteps...")
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
        except Exception as e:
            print(f"\n❌ Training failed with error: {e}")
            import traceback
            traceback.print_exc()
            raise

        return model, env

    def baseline_dqn(self):
        """
        Baseline DQN with standard 64x64x3 RGB observations.
        Iteration 1 - No improvements
        """
        return self.train_model(make_crafter_env)

    def reward_shaped_dqn(self):
        """
        DQN with reward shaping.
        Iteration 2 - Improvement 1: Reward Shaping
        """
        return self.train_model(make_shaped_crafter_env)

    def preprocessed_shaped_dqn(self):
        """
        DQN with reward shaping AND image preprocessing.
        Iteration 3 - Improvement 2: Grayscale + Normalize

        This combines both improvements:
        - Improvement 1: Reward shaping
        - Improvement 2: Image preprocessing (64x64x3 → 64x64x1, normalized to [0,1])

        Benefits:
        - 3x memory reduction
        - Faster training
        - Better gradient flow
        """
        return self.train_model(make_preprocessed_shaped_crafter_env)


# -------------------------------
# Main script
# -------------------------------
def main():
    parser = argparse.ArgumentParser(description="Train DQN models for Crafter environment")
    parser.add_argument('--model_type', type=str,
                        choices=['baseline', 'reward_shaped', 'preprocessed_shaped'],
                        required=True, help='Type of model to train')
    parser.add_argument('--log_path', type=str, default='./Training/Logs/DQN/',
                        help='Base directory for logs')
    parser.add_argument('--model_path', type=str, default=None,
                        help='Path to save the trained model')
    parser.add_argument('--total_timesteps', type=int, default=1_000_000,
                        help='Total training timesteps')
    parser.add_argument('--print_freq', type=int, default=1000,
                        help='Print training stats every N timesteps')
    parser.add_argument('--buffer_size', type=int, default=None,
                        help='Override buffer size (useful for memory-constrained systems)')
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    model_name = f"{args.model_type}_dqn_{timestamp}"
    if args.model_path is None:
        args.model_path = os.path.join(args.log_path, "models", model_name)
    log_path = os.path.join(args.log_path, args.model_type, timestamp)

    print(f"\n{'=' * 70}")
    print(f"Training Configuration:")
    print(f"  Model Type: {args.model_type}")
    print(f"  Total Timesteps: {args.total_timesteps:,}")
    print(f"  Log Path: {log_path}")
    print(f"  Model Path: {args.model_path}")
    if args.buffer_size:
        print(f"  Custom Buffer Size: {args.buffer_size:,}")
    print(f"{'=' * 70}\n")

    # Create custom config if buffer size is specified
    config = None
    if args.buffer_size:
        dqn_model_temp = DQN_Model(
            make_env=None,
            log_path=log_path,
            model_path=args.model_path,
            total_timesteps=args.total_timesteps,
            model_type=args.model_type
        )
        config = dqn_model_temp.config.copy()
        config['buffer_size'] = args.buffer_size
        # Adjust related parameters proportionally
        base_buffer = 100_000
        ratio = args.buffer_size / base_buffer
        base_target_update = 10_000
        base_learning_starts = 10_000
        config['target_update_interval'] = max(1000, int(base_target_update * ratio))
        config['learning_starts'] = max(1000, int(base_learning_starts * ratio))

    # Create DQN trainer
    dqn_model = DQN_Model(
        make_env=None,
        log_path=log_path,
        model_path=args.model_path,
        total_timesteps=args.total_timesteps,
        model_type=args.model_type,
        config=config
    )

    # Train the selected model
    if args.model_type == 'baseline':
        model, env = dqn_model.baseline_dqn()
    elif args.model_type == 'reward_shaped':
        model, env = dqn_model.reward_shaped_dqn()
    elif args.model_type == 'preprocessed_shaped':
        model, env = dqn_model.preprocessed_shaped_dqn()

    env.close()
    print(f"\n{'=' * 70}")
    print(f"✓ Training complete!")
    print(f"✓ Model saved to: {args.model_path}.zip")
    print(f"✓ Logs saved to: {log_path}")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    main()
import argparse
import os
import json
from datetime import datetime
from typing import Dict, Any, Optional
import numpy as np
import torch
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import psutil
from Model_Helpers.environments import make_env as make_crafter_env
from Model_Helpers.environments import make_shaped_env as make_shaped_crafter_env
from Model_Helpers.environments import make_preprocessed_shaped_env as make_preprocessed_shaped_crafter_env
from Model_Helpers.environments import make_framestack_env as make_framestack_crafter_env
from Model_Helpers.single_cnn_channel import SingleChannelCNN
from Model_Helpers.call_backs import AchievementCallback, TensorBoardPrintCallback


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
        print(f"Configuration saved to: {config_path}")

    def _estimate_buffer_memory(self, obs_shape, buffer_size):
        """Estimate replay buffer memory usage"""
        bytes_per_obs = np.prod(obs_shape) * 4  # float32 = 4 bytes
        # Buffer stores obs + next_obs
        total_mb = (bytes_per_obs * buffer_size * 2) / (1024 * 1024)
        return f"{total_mb:.1f} MB"

    def create_model(self, env, model_name: str = "dqn"):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"\nCreating {model_name} model on device: {device}")
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
                print(f"[DEBUG] Obs range: [{test_obs_array.min():.3f}, {test_obs_array.max():.3f}]")
            else:
                print(f"[DEBUG] Obs shape: {test_obs.shape}, dtype: {test_obs.dtype}")
                print(f"[DEBUG] Obs range: [{test_obs.min():.3f}, {test_obs.max():.3f}]")

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

        # Detect observation characteristics
        obs_channels = obs_space.shape[2] if len(obs_space.shape) == 3 else obs_space.shape[0]
        is_normalized = (obs_space.dtype == np.float32)
        is_single_channel = (obs_channels == 1)
        is_stacked = (obs_channels == 4)  # Typical frame stacking

        print(f"[INFO] Observation characteristics:")
        print(f"  Channels: {obs_channels}")
        print(f"  Normalized: {is_normalized}")
        print(f"  Single channel: {is_single_channel}")
        print(f"  Frame stacked: {is_stacked}")

        if is_single_channel and is_normalized:
            # Single grayscale channel
            print(f"[INFO] Using custom SingleChannelCNN features extractor")
            policy_kwargs['features_extractor_class'] = SingleChannelCNN
            policy_kwargs['features_extractor_kwargs'] = dict(features_dim=512)
        elif is_stacked and is_normalized:
            # Frame stacked observations (e.g., 4 grayscale frames)
            print(f"[INFO] Using custom SingleChannelCNN for stacked frames ({obs_channels} channels)")
            # The SingleChannelCNN should handle multiple channels
            policy_kwargs['features_extractor_class'] = SingleChannelCNN
            policy_kwargs['features_extractor_kwargs'] = dict(features_dim=512)
        elif is_normalized:
            # Multi-channel normalized images (standard RGB)
            print(f"[INFO] Using standard NatureCNN with normalized_image=True")
            policy_kwargs['features_extractor_kwargs'] = dict(normalized_image=True)
        else:
            # Default: uint8 RGB images
            print(f"[INFO] Using standard NatureCNN (default settings)")

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
                print(f"\nTraining complete! Model saved to: {self.model_path}.zip")
        except KeyboardInterrupt:
            print("\nTraining interrupted!")
            if self.model_path:
                interrupted_path = self.model_path + "_interrupted"
                model.save(interrupted_path)
                print(f"Interrupted model saved to: {interrupted_path}.zip")
        except Exception as e:
            print(f"\nTraining failed with error: {e}")
            import traceback
            traceback.print_exc()
            raise

        return model, env

    def baseline_dqn(self):
        """Baseline DQN """
        return self.train_model(make_crafter_env)

    def reward_shaped_dqn(self):
        """
        Improvement 1: Belief-based Reward Shaping
        """
        return self.train_model(make_shaped_crafter_env)

    def preprocessed_shaped_dqn(self):
        """
        Improvement 2: Grayscale + Normalize
        """
        return self.train_model(make_preprocessed_shaped_crafter_env)

    def framestack_dqn(self):
        """
        Improvement 3: Frame stacking
        """
        return self.train_model(make_framestack_crafter_env)


# -------------------------------
# Main script
# -------------------------------
def main():
    parser = argparse.ArgumentParser(description="Train DQN models for Crafter environment")
    parser.add_argument('--model_type', type=str,
                        choices=['baseline', 'reward_shaped', 'preprocessed_shaped', 'framestack'],
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
    elif args.model_type == 'framestack':
        model, env = dqn_model.framestack_dqn()

    env.close()
    print(f"\n{'=' * 70}")
    print(f"Training complete!")
    print(f"Model saved to: {args.model_path}.zip")
    print(f"Logs saved to: {log_path}")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":

    main()

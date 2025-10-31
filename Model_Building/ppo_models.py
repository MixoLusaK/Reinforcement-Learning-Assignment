import os
import sys
import json
import argparse
from datetime import datetime
from pathlib import Path
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gymnasium as gym
import gym as old_gym
from gym.envs.registration import register
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor
from shimmy import GymV21CompatibilityV0
import warnings

warnings.filterwarnings('ignore', message='.*Gym has been unmaintained.*')

#register Crafter environment
register(
    id='CrafterPartial-v1',
    entry_point='crafter:Env',
)

try:
    import crafter
except ImportError:
    print("Error: crafter package not found. Install with: pip install crafter")
    sys.exit(1)


class IntrinsicCuriosityWrapper(gym.Wrapper):

    def __init__(self, env, feature_dim=128, curiosity_weight=0.01, device='auto'):
        super().__init__(env)
        self.curiosity_weight = curiosity_weight

        if device == 'auto':
            try:
                if torch.cuda.is_available():
                    test_tensor = torch.zeros(1).cuda()
                    self.device = 'cuda'
                else:
                    self.device = 'cpu'
            except (AssertionError, RuntimeError):
                self.device = 'cpu'
        else:
            self.device = device

        print(f"ICM using device: {self.device}")

        #observation shape
        obs_shape = env.observation_space.shape

        self.feature_net = nn.Sequential(
            nn.Conv2d(obs_shape[2], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(1024, feature_dim)
        ).to(self.device)

        self.forward_model = nn.Sequential(
            nn.Linear(feature_dim + env.action_space.n, 256),
            nn.ReLU(),
            nn.Linear(256, feature_dim)
        ).to(self.device)

        self.inverse_model = nn.Sequential(
            nn.Linear(feature_dim * 2, 256),
            nn.ReLU(),
            nn.Linear(256, env.action_space.n)
        ).to(self.device)

        self.optimizer = torch.optim.Adam(
            list(self.feature_net.parameters()) +
            list(self.forward_model.parameters()) +
            list(self.inverse_model.parameters()),
            lr=1e-4
        )

        self.prev_obs = None
        self.total_intrinsic_reward = 0
        self.episode_intrinsic_reward = 0

    def _preprocess_obs(self, obs):
        if obs.dtype == np.uint8:
            obs = obs.astype(np.float32) / 255.0

        obs_tensor = torch.FloatTensor(obs).to(self.device)
        obs_tensor = obs_tensor.permute(2, 0, 1).unsqueeze(0)
        return obs_tensor

    def _compute_intrinsic_reward(self, prev_obs, action, next_obs):
        with torch.no_grad():
            prev_obs_tensor = self._preprocess_obs(prev_obs)
            next_obs_tensor = self._preprocess_obs(next_obs)

            #extract features
            prev_features = self.feature_net(prev_obs_tensor)
            next_features = self.feature_net(next_obs_tensor)

            #create action one-hot
            action_tensor = torch.zeros(1, self.env.action_space.n).to(self.device)
            action_tensor[0, action] = 1.0

            #forward model prediction
            pred_next_features = self.forward_model(
                torch.cat([prev_features, action_tensor], dim=1)
            )

            #prediction error = intrinsic reward
            intrinsic_reward = torch.norm(next_features - pred_next_features, dim=1).item()

        #train ICM networks
        self._train_icm(prev_obs, action, next_obs)

        return intrinsic_reward * self.curiosity_weight

    def _train_icm(self, prev_obs, action, next_obs):
        prev_obs_tensor = self._preprocess_obs(prev_obs)
        next_obs_tensor = self._preprocess_obs(next_obs)
        prev_features = self.feature_net(prev_obs_tensor)
        next_features = self.feature_net(next_obs_tensor)
        action_tensor = torch.zeros(1, self.env.action_space.n).to(self.device)
        action_tensor[0, action] = 1.0
        pred_next_features = self.forward_model(
            torch.cat([prev_features, action_tensor], dim=1)
        )
        forward_loss = nn.MSELoss()(pred_next_features, next_features.detach())

        #inverse model prediction
        pred_action_logits = self.inverse_model(
            torch.cat([prev_features, next_features], dim=1)
        )
        action_target = torch.LongTensor([action]).to(self.device)
        inverse_loss = nn.CrossEntropyLoss()(pred_action_logits, action_target)

        #total loss
        loss = forward_loss + inverse_loss

        #update networks
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.prev_obs = obs
        self.episode_intrinsic_reward = 0
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        #compute intrinsic reward
        if self.prev_obs is not None:
            intrinsic_reward = self._compute_intrinsic_reward(
                self.prev_obs, action, obs
            )
            self.episode_intrinsic_reward += intrinsic_reward
            self.total_intrinsic_reward += intrinsic_reward

            #add intrinsic reward to extrinsic reward
            reward = reward + intrinsic_reward
            info['intrinsic_reward'] = intrinsic_reward
            info['episode_intrinsic_reward'] = self.episode_intrinsic_reward

        self.prev_obs = obs
        return obs, reward, terminated, truncated, info


class BeliefRewardWrapper(gym.Wrapper):
    def __init__(self, env, health_weight=0.1, exploration_weight=0.01):
        super().__init__(env)
        self.health_weight = health_weight
        self.exploration_weight = exploration_weight
        self.prev_health = None
        self.prev_hunger = None
        self.visited_positions = set()

    def reset(self, **kwargs):
        result = self.env.reset(**kwargs)
        if isinstance(result, tuple) and len(result) == 2:
            obs, info = result
        else:
            obs = result
            info = {}

        self.prev_health = info.get('health', 9)
        self.prev_hunger = info.get('hunger', 9)
        self.visited_positions.clear()

        #initialize with starting position if available
        if 'player_pos' in info:
            start_pos = tuple(info['player_pos'])  # Convert to tuple
            self.visited_positions.add(start_pos)

        return obs, info

    def step(self, action):
        result = self.env.step(action)
        if len(result) == 5:
            obs, reward, terminated, truncated, info = result
        elif len(result) == 4:
            obs, reward, done, info = result
            terminated = done
            truncated = False
        else:
            raise ValueError(f"Unexpected step result length: {len(result)}")

        current_health = info.get('health', 9)
        current_hunger = info.get('hunger', 9)

        if self.prev_health is not None:
            health_reward = (current_health - self.prev_health) * self.health_weight
            reward += health_reward

        #add exploration reward
        if 'player_pos' in info:
            player_pos = info['player_pos']

            if isinstance(player_pos, np.ndarray):
                player_pos_tuple = tuple(player_pos.tolist())
            else:
                player_pos_tuple = tuple(player_pos)

            if player_pos_tuple not in self.visited_positions:
                self.visited_positions.add(player_pos_tuple)
                reward += self.exploration_weight

        self.prev_health = current_health
        self.prev_hunger = current_hunger

        return obs, reward, terminated, truncated, info

class ICMLoggingCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.intrinsic_rewards = []

    def _on_step(self):
        if len(self.locals.get('infos', [])) > 0:
            for info in self.locals['infos']:
                if 'intrinsic_reward' in info:
                    self.intrinsic_rewards.append(info['intrinsic_reward'])

        if self.n_calls % 1000 == 0 and len(self.intrinsic_rewards) > 0:
            mean_intrinsic = np.mean(self.intrinsic_rewards[-1000:])
            self.logger.record('icm/mean_intrinsic_reward', mean_intrinsic)

        return True


def _patch_metadata(env):
    if not hasattr(env, "metadata") or env.metadata is None:
        env.metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}
    elif "render_modes" not in env.metadata:
        env.metadata["render_modes"] = ["rgb_array"]
    if "render_fps" not in env.metadata:
        env.metadata["render_fps"] = 30
    return env


class PPOTrainer:
    def __init__(self,
                 model_type='baseline',
                 use_icm=False,
                 curiosity_weight=0.01,
                 total_timesteps=1_000_000,
                 learning_rate=3e-4,
                 n_steps=2048,
                 batch_size=64,
                 n_epochs=10,
                 gamma=0.99,
                 gae_lambda=0.95,
                 clip_range=0.2,
                 ent_coef=0.01,
                 vf_coef=0.5,
                 max_grad_norm=0.5,
                 device='auto'):

        self.model_type = model_type
        self.use_icm = use_icm
        self.curiosity_weight = curiosity_weight
        self.total_timesteps = total_timesteps
        self.learning_rate = learning_rate
        self.n_steps = n_steps
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_range = clip_range
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.device = device

        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        model_suffix = "_icm" if use_icm else ""
        self.run_dir = Path(f"./Training/Logs/PPO/{model_type}{model_suffix}/{timestamp}")
        self.run_dir.mkdir(parents=True, exist_ok=True)

        # Save configuration
        self.save_config()

    def save_config(self):
        config = {
            'algorithm': 'PPO',
            'model_type': self.model_type,
            'use_icm': self.use_icm,
            'curiosity_weight': self.curiosity_weight if self.use_icm else None,
            'total_timesteps': self.total_timesteps,
            'learning_rate': self.learning_rate,
            'n_steps': self.n_steps,
            'batch_size': self.batch_size,
            'n_epochs': self.n_epochs,
            'gamma': self.gamma,
            'gae_lambda': self.gae_lambda,
            'clip_range': self.clip_range,
            'ent_coef': self.ent_coef,
            'vf_coef': self.vf_coef,
            'max_grad_norm': self.max_grad_norm,
            'device': self.device,
            'timestamp': datetime.now().isoformat()
        }

        config_path = self.run_dir / "config.json"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=4)
        print(f"Configuration saved to: {config_path}")

    def create_env(self, is_eval=False):
        if is_eval:
            log_dir = str(self.run_dir / "eval_logs")
        else:
            log_dir = str(self.run_dir / "train_logs")

        os.makedirs(log_dir, exist_ok=True)
        env = old_gym.make("CrafterPartial-v1")

        def ensure_metadata(env_obj):
            if not hasattr(env_obj, "metadata") or env_obj.metadata is None:
                env_obj.metadata = {"render_modes": ["rgb_array"], "render_fps": 30}
            elif "render_modes" not in env_obj.metadata:
                env_obj.metadata["render_modes"] = ["rgb_array"]
            if "render_fps" not in env_obj.metadata:
                env_obj.metadata["render_fps"] = 30

        ensure_metadata(env)
        current = env
        while hasattr(current, 'env'):
            current = current.env
            ensure_metadata(current)

        #crafter's Recorder
        env = crafter.Recorder(
            env,
            log_dir,
            save_stats=True,
            save_video=is_eval,  # Save videos only for eval
            save_episode=True
        )
        ensure_metadata(env)

        env = GymV21CompatibilityV0(env=env)
        ensure_metadata(env)

        if self.model_type == 'shaped':
            print("Adding reward shaping wrapper")
            env = BeliefRewardWrapper(env)
            ensure_metadata(env)

        env = Monitor(env)

        if self.use_icm and not is_eval:
            print(f"Adding Intrinsic Curiosity Module (weight={self.curiosity_weight})")
            icm_device = self.device
            if icm_device == 'auto':
                try:
                    if torch.cuda.is_available():
                        test_tensor = torch.zeros(1).cuda()
                        icm_device = 'cuda'
                    else:
                        icm_device = 'cpu'
                except (AssertionError, RuntimeError):
                    # CUDA reported available but doesn't actually work
                    icm_device = 'cpu'

            env = IntrinsicCuriosityWrapper(
                env,
                curiosity_weight=self.curiosity_weight,
                device=icm_device
            )

        #vectorize environment
        env = DummyVecEnv([lambda: env])

        #transpose images for PyTorch
        env = VecTransposeImage(env)

        return env

    def create_model(self, train_env):
        print(f"\nCreating PPO model on device: {self.device}")

        tensorboard_log = str(self.run_dir / "tensorboard")

        model = PPO(
            policy="CnnPolicy",
            env=train_env,
            learning_rate=self.learning_rate,
            n_steps=self.n_steps,
            batch_size=self.batch_size,
            n_epochs=self.n_epochs,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            clip_range=self.clip_range,
            ent_coef=self.ent_coef,
            vf_coef=self.vf_coef,
            max_grad_norm=self.max_grad_norm,
            verbose=1,
            tensorboard_log=tensorboard_log,
            device=self.device
        )

        return model

    def train_model(self):
        print("\n" + "=" * 70)
        print(f"Training PPO Model {'with ICM' if self.use_icm else ''}")
        print("=" * 70)

        train_env = self.create_env(is_eval=False)
        eval_env = self.create_env(is_eval=True)

        print(f"Observation space: {train_env.observation_space}")
        print(f"Action space: {train_env.action_space}")

        model = self.create_model(train_env)

        callbacks = []

        if self.use_icm:
            icm_callback = ICMLoggingCallback()
            callbacks.append(icm_callback)

        checkpoint_callback = CheckpointCallback(
            save_freq=50_000,
            save_path=str(self.run_dir / "checkpoints"),
            name_prefix="ppo_crafter",
            save_replay_buffer=False,
            save_vecnormalize=True
        )
        callbacks.append(checkpoint_callback)

        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=str(self.run_dir / "best_model"),
            log_path=str(self.run_dir / "eval_logs"),
            eval_freq=25_000,
            n_eval_episodes=5,
            deterministic=True,
            render=False
        )
        callbacks.append(eval_callback)

        print(f"\nStarting training for {self.total_timesteps:,} timesteps")
        if self.use_icm:
            print(f"ICM Curiosity Weight: {self.curiosity_weight}")
        print(f"Checkpoints will be saved to: {self.run_dir / 'checkpoints'}")
        print(f"Best model will be saved to: {self.run_dir / 'best_model'}")
        print(f"\n{'=' * 70}")
        print(f"To view training in TensorBoard, run:")
        print(f"tensorboard --logdir {self.run_dir / 'tensorboard'}")
        print(f"{'=' * 70}\n")

        model.learn(
            total_timesteps=self.total_timesteps,
            callback=callbacks,
            progress_bar=True
        )

        #save final model
        final_model_path = self.run_dir / "final_model"
        model.save(final_model_path)
        print(f"\nFinal model saved to: {final_model_path}")

        return model, train_env


def main():
    parser = argparse.ArgumentParser(description='Train PPO on Crafter with optional ICM')
    parser.add_argument('--model_type', type=str, default='baseline',
                        choices=['baseline', 'shaped', 'shaped_icm'],
                        help='Model type: baseline or shaped (with reward shaping)')
    parser.add_argument('--use_icm', action='store_true',
                        help='Use Intrinsic Curiosity Module for exploration')
    parser.add_argument('--curiosity_weight', type=float, default=0.01,
                        help='Weight for intrinsic curiosity rewards')

    parser.add_argument('--total_timesteps', type=int, default=1_000_000,
                        help='Total training timesteps')
    parser.add_argument('--learning_rate', type=float, default=3e-4,
                        help='Learning rate')
    parser.add_argument('--n_steps', type=int, default=2048,
                        help='Number of steps to run per update')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Minibatch size')
    parser.add_argument('--n_epochs', type=int, default=10,
                        help='Number of epochs for optimization')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='Discount factor')
    parser.add_argument('--gae_lambda', type=float, default=0.95,
                        help='GAE lambda parameter')
    parser.add_argument('--clip_range', type=float, default=0.2,
                        help='PPO clipping parameter')
    parser.add_argument('--ent_coef', type=float, default=0.01,
                        help='Entropy coefficient')
    parser.add_argument('--vf_coef', type=float, default=0.5,
                        help='Value function coefficient')
    parser.add_argument('--max_grad_norm', type=float, default=0.5,
                        help='Max gradient norm')
    parser.add_argument('--device', type=str, default='auto',
                        choices=['auto', 'cuda', 'cpu'],
                        help='Device to use for training')

    args = parser.parse_args()

    ppo_trainer = PPOTrainer(
        model_type=args.model_type,
        use_icm=args.use_icm,
        curiosity_weight=args.curiosity_weight,
        total_timesteps=args.total_timesteps,
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_range=args.clip_range,
        ent_coef=args.ent_coef,
        vf_coef=args.vf_coef,
        max_grad_norm=args.max_grad_norm,
        device=args.device
    )

    model, env = ppo_trainer.train_model()


if __name__ == "__main__":
    main()
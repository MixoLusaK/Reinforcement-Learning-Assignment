import os
import sys
import json
import argparse
from datetime import datetime
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecTransposeImage
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor

from Model_Helpers.environments import make_env, make_shaped_env


class PPOTrainer:
    """
    PPO training wrapper for Crafter environment.
    """
    
    def __init__(self, 
                 model_type='baseline',
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
                 frame_stack=4,
                 device='auto'):
        """
        Initialize PPO trainer.
        
        Args:
            model_type: 'baseline' or 'shaped' (with reward shaping)
            total_timesteps: Total training timesteps
            learning_rate: Learning rate
            n_steps: Number of steps to run per update
            batch_size: Minibatch size
            n_epochs: Number of epochs when optimizing surrogate loss
            gamma: Discount factor
            gae_lambda: Factor for trade-off of bias vs variance for GAE
            clip_range: Clipping parameter for PPO
            ent_coef: Entropy coefficient for exploration
            vf_coef: Value function coefficient
            max_grad_norm: Max gradient norm for clipping
            frame_stack: Number of frames to stack
            device: Device to run on ('auto', 'cuda', 'cpu')
        """
        self.model_type = model_type
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
        self.frame_stack = frame_stack
        self.device = device
        
        # Create timestamped directory for this run
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.run_dir = Path(f"./Training/Logs/PPO/{model_type}/{timestamp}")
        self.run_dir.mkdir(parents=True, exist_ok=True)
        
        # Save configuration
        self.save_config()
        
    def save_config(self):
        """Save training configuration to JSON."""
        config = {
            'algorithm': 'PPO',
            'model_type': self.model_type,
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
            'frame_stack': self.frame_stack,
            'device': self.device,
            'timestamp': datetime.now().isoformat()
        }
        
        config_path = self.run_dir / "config.json"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=4)
        print(f"✓ Configuration saved to: {config_path}")
        
    def create_env(self, is_eval=False):
        """
        Create wrapped Crafter environment.
        
        Args:
            is_eval: Whether this is an evaluation environment
            
        Returns:
            Wrapped vectorized environment
        """
        # Different log directories for train/eval
        if is_eval:
            log_dir = str(self.run_dir / "eval_logs")
        else:
            log_dir = str(self.run_dir / "train_logs")
        
        # Create environment based on model type
        if self.model_type == 'shaped':
            env = make_shaped_env(
                log_dir=log_dir,
                save_video=is_eval,  # Save videos only for eval
                save_episode=True
            )
        else:
            env = make_env(
                log_dir=log_dir,
                save_video=is_eval,
                save_episode=True
            )
        
        # Wrap in Monitor for statistics
        env = Monitor(env)
        
        # Vectorize environment
        env = DummyVecEnv([lambda: env])
        
        # Stack frames for temporal information
        if self.frame_stack > 1:
            env = VecFrameStack(env, n_stack=self.frame_stack)
        
        # Transpose images for PyTorch (channels first)
        env = VecTransposeImage(env)
        
        return env
    
    def create_model(self, train_env):
        """
        Create PPO model.
        
        Args:
            train_env: Training environment
            
        Returns:
            PPO model
        """
        print(f"\n✓ Creating PPO model on device: {self.device}")
        
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
        """
        Train PPO model with callbacks.
        
        Returns:
            Trained model and training environment
        """
        print("\n" + "="*70)
        print("Training PPO Model")
        print("="*70)
        
        # Create environments
        train_env = self.create_env(is_eval=False)
        eval_env = self.create_env(is_eval=True)
        
        print(f"Observation space: {train_env.observation_space}")
        print(f"Action space: {train_env.action_space}")
        
        # Create model
        model = self.create_model(train_env)
        
        # Setup callbacks
        callbacks = []
        
        # Checkpoint callback - save model periodically
        checkpoint_callback = CheckpointCallback(
            save_freq=50_000,  # Save every 50k steps
            save_path=str(self.run_dir / "checkpoints"),
            name_prefix="ppo_crafter",
            save_replay_buffer=False,
            save_vecnormalize=True
        )
        callbacks.append(checkpoint_callback)
        
        # Evaluation callback - evaluate periodically
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=str(self.run_dir / "best_model"),
            log_path=str(self.run_dir / "eval_logs"),
            eval_freq=25_000,  # Evaluate every 25k steps
            n_eval_episodes=5,
            deterministic=True,
            render=False
        )
        callbacks.append(eval_callback)
        
        # Print training info
        print(f"\n✓ Starting training for {self.total_timesteps:,} timesteps")
        print(f"✓ Checkpoints will be saved to: {self.run_dir / 'checkpoints'}")
        print(f"✓ Best model will be saved to: {self.run_dir / 'best_model'}")
        print(f"\n{'='*70}")
        print(f"To view training in TensorBoard, run:")
        print(f"tensorboard --logdir {self.run_dir / 'tensorboard'}")
        print(f"{'='*70}\n")
        
        # Train the model
        model.learn(
            total_timesteps=self.total_timesteps,
            callback=callbacks,
            progress_bar=True
        )
        
        # Save final model
        final_model_path = self.run_dir / "final_model"
        model.save(final_model_path)
        print(f"\n✓ Final model saved to: {final_model_path}")
        
        return model, train_env
    
    def load_and_evaluate(self, model_path, n_eval_episodes=10):
        """
        Load a trained model and evaluate it.
        
        Args:
            model_path: Path to saved model
            n_eval_episodes: Number of episodes to evaluate
            
        Returns:
            List of episode rewards
        """
        print(f"\n✓ Loading model from: {model_path}")
        
        # Create evaluation environment
        eval_env = self.create_env(is_eval=True)
        
        # Load model
        model = PPO.load(model_path, env=eval_env, device=self.device)
        
        print(f"✓ Evaluating for {n_eval_episodes} episodes...")
        
        episode_rewards = []
        for episode in range(n_eval_episodes):
            obs = eval_env.reset()
            done = False
            episode_reward = 0
            
            while not done:
                action, _states = model.predict(obs, deterministic=True)
                obs, reward, done, info = eval_env.step(action)
                episode_reward += reward[0]
            
            episode_rewards.append(episode_reward)
            print(f"Episode {episode + 1}: Reward = {episode_reward:.2f}")
        
        mean_reward = sum(episode_rewards) / len(episode_rewards)
        print(f"\n✓ Mean reward over {n_eval_episodes} episodes: {mean_reward:.2f}")
        
        return episode_rewards


def main():
    """Main training script."""
    parser = argparse.ArgumentParser(description='Train PPO on Crafter')
    
    # Model configuration
    parser.add_argument('--model_type', type=str, default='baseline',
                        choices=['baseline', 'shaped'],
                        help='Model type: baseline or shaped (with reward shaping)')
    
    # Training parameters
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
    parser.add_argument('--frame_stack', type=int, default=4,
                        help='Number of frames to stack')
    parser.add_argument('--device', type=str, default='auto',
                        choices=['auto', 'cuda', 'cpu'],
                        help='Device to use for training')
    
    # Evaluation
    parser.add_argument('--eval_only', action='store_true',
                        help='Only evaluate a trained model')
    parser.add_argument('--model_path', type=str, default=None,
                        help='Path to model for evaluation')
    parser.add_argument('--n_eval_episodes', type=int, default=10,
                        help='Number of evaluation episodes')
    
    args = parser.parse_args()
    
    # Create trainer
    ppo_trainer = PPOTrainer(
        model_type=args.model_type,
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
        frame_stack=args.frame_stack,
        device=args.device
    )
    
    if args.eval_only:
        if args.model_path is None:
            raise ValueError("--model_path must be provided for evaluation")
        ppo_trainer.load_and_evaluate(args.model_path, args.n_eval_episodes)
    else:
        model, env = ppo_trainer.train_model()


if __name__ == "__main__":
    main()
# Dependencies
import argparse
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv
import os
import sys
import numpy as np
import json
from collections import defaultdict
from datetime import datetime
import gymnasium as gym
from shimmy import GymV21CompatibilityV0
import gym as old_gym
from gym.envs.registration import register
import warnings

warnings.filterwarnings('ignore', message='.*Gym has been unmaintained.*')

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    import fix_numpy_compat
except ImportError:
    pass

register(
    id='CrafterPartial-v1',
    entry_point='crafter:Env',
)

try:
    import crafter
except ImportError:
    print("Error: crafter package not found. Install with: pip install crafter")
    sys.exit(1)


class GrayscaleNormalizeWrapper(gym.ObservationWrapper):
    """
    Combines grayscale conversion and normalization into one wrapper.
    Must match the preprocessing used during training.
    """

    def __init__(self, env):
        super().__init__(env)
        obs_shape = self.observation_space.shape
        new_shape = (obs_shape[0], obs_shape[1], 1)

        self.observation_space = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=new_shape,
            dtype=np.float32
        )

    def observation(self, obs):
        """Convert RGB observation to grayscale and normalize"""
        grayscale = np.dot(obs[..., :3], [0.299, 0.587, 0.114])
        normalized = grayscale.astype(np.float32) / 255.0
        return np.expand_dims(normalized, axis=-1)


class FrameStackWrapper(gym.Wrapper):
    """
    Stack frames for temporal information.
    Must match the frame stacking used during training.
    """

    def __init__(self, env, num_stack=4):
        super().__init__(env)
        self.num_stack = num_stack
        self.frames = None

        obs_shape = env.observation_space.shape
        new_shape = (obs_shape[0], obs_shape[1], obs_shape[2] * num_stack)

        self.observation_space = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=new_shape,
            dtype=np.float32
        )

    def reset(self, **kwargs):
        result = self.env.reset(**kwargs)
        if isinstance(result, tuple):
            obs, info = result
        else:
            obs = result
            info = {}

        # Initialize frame stack with the first observation
        self.frames = [obs for _ in range(self.num_stack)]
        return self._get_stacked_obs(), info

    def step(self, action):
        result = self.env.step(action)

        if len(result) == 5:
            obs, reward, terminated, truncated, info = result
            done = terminated or truncated
        else:
            obs, reward, done, info = result
            terminated = done
            truncated = False

        # Update frame stack
        self.frames.pop(0)
        self.frames.append(obs)

        if len(result) == 5:
            return self._get_stacked_obs(), reward, terminated, truncated, info
        else:
            return self._get_stacked_obs(), reward, done, info

    def _get_stacked_obs(self):
        """Stack frames along the channel dimension"""
        return np.concatenate(self.frames, axis=-1)


class DQN_Testing:
    """
    Class to test DQN models on Crafter environment using evaluate_policy.
    """

    def __init__(self, model_path, num_episodes=100, video_dir="./crafter_videos/",
                 results_dir="./results/", use_preprocessing=False,
                 use_frame_stacking=False, num_stack=4):
        self.model_path = model_path
        self.num_episodes = num_episodes
        self.video_dir = video_dir
        self.results_dir = results_dir
        self.use_preprocessing = use_preprocessing
        self.use_frame_stacking = use_frame_stacking
        self.num_stack = num_stack

        # Create directories
        os.makedirs(self.video_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)

        # All possible achievements in Crafter
        self.all_achievements = [
            "collect_wood", "collect_stone", "collect_coal", "collect_iron",
            "collect_diamond", "collect_sapling", "collect_drink", "place_table",
            "place_plant", "place_stone", "place_furnace", "make_wood_pickaxe",
            "make_stone_pickaxe", "make_iron_pickaxe", "make_wood_sword",
            "make_stone_sword", "make_iron_sword", "defeat_zombie", "defeat_skeleton",
            "eat_cow", "eat_plant", "wake_up"
        ]

    def _make_test_env(self, record_stats=True, record_video=False):
        """
        Create a testing environment compatible with the training setup.
        """
        # Create base Crafter environment (standard rewards, no reward shaping)
        env = old_gym.make("CrafterPartial-v1")

        # CRITICAL: Ensure all wrappers have proper metadata
        # The Recorder needs metadata.render_modes to be available at every level
        def ensure_metadata(env_obj):
            """Helper to ensure metadata exists on environment"""
            if not hasattr(env_obj, "metadata") or env_obj.metadata is None:
                env_obj.metadata = {"render_modes": ["rgb_array"], "render_fps": 30}
            elif "render_modes" not in env_obj.metadata:
                env_obj.metadata["render_modes"] = ["rgb_array"]
            if "render_fps" not in env_obj.metadata:
                env_obj.metadata["render_fps"] = 30

        ensure_metadata(env)

        # Ensure all nested environments have metadata
        current = env
        while hasattr(current, 'env'):
            current = current.env
            ensure_metadata(current)

        # Use Crafter's built-in Recorder for stats tracking
        if record_stats:
            env = crafter.Recorder(
                env,
                self.video_dir,
                save_stats=True,
                save_video=record_video,
                save_episode=False
            )
            ensure_metadata(env)

        # Apply GymV21 compatibility wrapper
        env = GymV21CompatibilityV0(env=env)
        ensure_metadata(env)

        # Apply preprocessing if required
        if self.use_preprocessing:
            env = GrayscaleNormalizeWrapper(env)
            ensure_metadata(env)

        # Apply frame stacking if required
        if self.use_frame_stacking:
            env = FrameStackWrapper(env, num_stack=self.num_stack)
            ensure_metadata(env)

        return env

    def test_with_evaluate_policy(self, model_type='baseline'):
        """
        Test using Stable Baselines3's evaluate_policy function.
        This is simpler but provides less detailed metrics.
        """
        print(f"Loading model from: {self.model_path}")
        model = DQN.load(self.model_path)

        model_name_display = {
            'baseline': 'DQN Baseline',
            'reward_shaped': 'DQN Reward Shaped',
            'preprocessed_shaped': 'DQN Preprocessed + Reward Shaped',
            'frame_stacking': 'DQN Frame Stacking + Preprocessed + Reward Shaped'
        }.get(model_type, f'DQN {model_type.title()}')

        print(f"\nTesting {model_name_display} Model using evaluate_policy...")
        print("=" * 70)

        # Create evaluation environment
        eval_env = self._make_test_env(record_stats=True)

        # Use evaluate_policy for quick evaluation
        mean_reward, std_reward = evaluate_policy(
            model,
            eval_env,
            n_eval_episodes=self.num_episodes,
            deterministic=True,
            return_episode_rewards=False
        )

        print(f"\nEvaluation Results:")
        print(f"  Mean Reward: {mean_reward:.2f} ± {std_reward:.2f}")
        print(f"  Episodes: {self.num_episodes}")

        # Save basic results
        results = {
            "model": model_name_display,
            "num_episodes": self.num_episodes,
            "timestamp": datetime.now().isoformat(),
            "evaluation_method": "evaluate_policy",
            "preprocessing": self.use_preprocessing,
            "frame_stacking": self.use_frame_stacking,
            "num_stack": self.num_stack if self.use_frame_stacking else None,
            "metrics": {
                "mean_reward": float(mean_reward),
                "std_reward": float(std_reward),
            }
        }

        results_path = os.path.join(self.results_dir, f"dqn_{model_type}_evaluate_policy_results.json")
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"Results saved to: {results_path}")
        print("=" * 70)

        eval_env.close()
        return results

    def test_with_detailed_tracking(self, model_type='baseline'):
        """
        Test with detailed achievement and action tracking.
        This is your original implementation with more metrics.
        """
        print(f"Loading model from: {self.model_path}")
        model = DQN.load(self.model_path)

        # Metrics storage
        episode_rewards = []
        episode_lengths = []
        achievement_unlocks = defaultdict(int)
        achievement_per_episode = []
        action_counts = defaultdict(int)

        model_name_display = {
            'baseline': 'DQN Baseline',
            'reward_shaped': 'DQN Reward Shaped',
            'preprocessed_shaped': 'DQN Preprocessed + Reward Shaped',
            'frame_stacking': 'DQN Frame Stacking + Preprocessed + Reward Shaped'
        }.get(model_type, f'DQN {model_type.title()}')

        print(f"\nTesting {model_name_display} Model with detailed tracking...")
        print("=" * 70)

        for episode in range(self.num_episodes):
            # Record video for first 10 episodes
            record_video = episode < 10

            # Create new environment with video recording enabled if needed
            env = self._make_test_env(record_stats=True, record_video=record_video)

            if record_video:
                print(f"Recording video for episode {episode + 1}")

            obs = env.reset()
            if isinstance(obs, tuple):
                obs, info = obs
            else:
                info = {}

            done = False
            episode_reward = 0
            step = 0
            episode_achievements = set()

            while not done:
                action, _ = model.predict(obs, deterministic=True)
                action_counts[int(action)] += 1

                step_result = env.step(action)

                if len(step_result) == 5:
                    obs, reward, terminated, truncated, info = step_result
                    done = terminated or truncated
                else:
                    obs, reward, done, info = step_result

                episode_reward += reward
                step += 1

                # Track achievements
                if 'achievements' in info:
                    for achievement, unlocked in info['achievements'].items():
                        if unlocked and achievement not in episode_achievements:
                            achievement_unlocks[achievement] += 1
                            episode_achievements.add(achievement)
                elif 'semantic' in info and isinstance(info['semantic'], dict):
                    for achievement, unlocked in info['semantic'].items():
                        if unlocked and achievement not in episode_achievements:
                            achievement_unlocks[achievement] += 1
                            episode_achievements.add(achievement)

            episode_rewards.append(episode_reward)
            episode_lengths.append(step)
            achievement_per_episode.append(len(episode_achievements))

            print(f"Episode {episode + 1:3d}/{self.num_episodes}: "
                  f"Reward={episode_reward:6.2f}, "
                  f"Steps={step:4d}, "
                  f"Achievements={len(episode_achievements):2d}")

            env.close()

        # Calculate metrics
        avg_reward = np.mean(episode_rewards)
        std_reward = np.std(episode_rewards)
        avg_survival_time = np.mean(episode_lengths)
        std_survival_time = np.std(episode_lengths)
        avg_achievements = np.mean(achievement_per_episode)

        # Achievement rates
        achievement_rates = {}
        for achievement in self.all_achievements:
            count = achievement_unlocks.get(achievement, 0)
            achievement_rates[achievement] = count / self.num_episodes

        # Geometric mean
        epsilon = 1e-10
        non_zero_rates = [rate + epsilon for rate in achievement_rates.values() if rate > 0]
        geometric_mean = np.exp(np.mean(np.log(non_zero_rates))) - epsilon if non_zero_rates else 0.0

        # Print summary
        print("\n" + "=" * 70)
        print(f"EVALUATION SUMMARY - {model_name_display.upper()}")
        print("=" * 70)
        print(f"\nPerformance Metrics:")
        print(f"  Average Cumulative Reward:     {avg_reward:8.2f} ± {std_reward:.2f}")
        print(f"  Average Survival Time:         {avg_survival_time:8.2f} ± {std_survival_time:.2f} steps")
        print(f"  Average Achievements/Episode:  {avg_achievements:8.2f}")
        print(f"  Geometric Mean of Achievements: {geometric_mean:7.4f}")

        print(f"\nAchievement Unlock Rates:")
        sorted_achievements = sorted(achievement_rates.items(), key=lambda x: x[1], reverse=True)
        for achievement, rate in sorted_achievements:
            count = achievement_unlocks.get(achievement, 0)
            bar = "█" * int(rate * 50)
            bar = bar.ljust(50)
            print(f"  {achievement:30s}: {rate * 100:5.1f}% [{bar}] ({count:3d}/{self.num_episodes})")

        # Save results
        results = {
            "model": model_name_display,
            "num_episodes": self.num_episodes,
            "timestamp": datetime.now().isoformat(),
            "evaluation_method": "detailed_tracking",
            "preprocessing": self.use_preprocessing,
            "frame_stacking": self.use_frame_stacking,
            "num_stack": self.num_stack if self.use_frame_stacking else None,
            "metrics": {
                "average_reward": float(avg_reward),
                "std_reward": float(std_reward),
                "average_survival_time": float(avg_survival_time),
                "std_survival_time": float(std_survival_time),
                "average_achievements_per_episode": float(avg_achievements),
                "geometric_mean_achievements": float(geometric_mean),
            },
            "achievement_unlock_rates": achievement_rates,
            "action_distribution": {k: int(v) for k, v in action_counts.items()},
            "episode_rewards": episode_rewards,
            "episode_lengths": episode_lengths,
            "achievements_per_episode": achievement_per_episode
        }

        results_path = os.path.join(self.results_dir, f"dqn_{model_type}_detailed_results.json")
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\n Results saved to: {results_path}")
        print(f" Videos saved to: {self.video_dir}")
        print("=" * 70)

        return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test DQN models on Crafter')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the trained model')
    parser.add_argument('--model_type', type=str,
                        choices=['baseline', 'reward_shaped', 'preprocessed_shaped', 'frame_stacking'],
                        default='baseline', help='Type of model to test')
    parser.add_argument('--num_episodes', type=int, default=200,
                        help='Number of episodes to test')
    parser.add_argument('--use_preprocessing', action='store_true',
                        help='Use grayscale+normalization preprocessing')
    parser.add_argument('--use_frame_stacking', action='store_true',
                        help='Use frame stacking')
    parser.add_argument('--num_stack', type=int, default=4,
                        help='Number of frames to stack (default: 4)')
    parser.add_argument('--evaluation_method', type=str,
                        choices=['evaluate_policy', 'detailed'],
                        default='detailed',
                        help='Evaluation method: evaluate_policy (fast) or detailed (full metrics)')
    parser.add_argument('--video_dir', type=str, default='./crafter_videos/',
                        help='Directory to save videos')
    parser.add_argument('--results_dir', type=str, default='./results/',
                        help='Directory to save results')

    args = parser.parse_args()

    # Auto-enable preprocessing for preprocessed_shaped model
    if args.model_type == 'preprocessed_shaped':
        args.use_preprocessing = True
        print(f"Note: Automatically enabled preprocessing for '{args.model_type}' model")

    # Auto-enable preprocessing + frame stacking for frame_stacking model
    if args.model_type == 'frame_stacking':
        args.use_preprocessing = True
        args.use_frame_stacking = True
        print(
            f"Note: Automatically enabled preprocessing + frame stacking ({args.num_stack} frames) for '{args.model_type}' model")

    tester = DQN_Testing(
        model_path=args.model_path,
        num_episodes=args.num_episodes,
        video_dir=args.video_dir,
        results_dir=args.results_dir,
        use_preprocessing=args.use_preprocessing,
        use_frame_stacking=args.use_frame_stacking,
        num_stack=args.num_stack
    )

    # Run test based on evaluation method
    if args.evaluation_method == 'evaluate_policy':
        results = tester.test_with_evaluate_policy(model_type=args.model_type)
    else:

        results = tester.test_with_detailed_tracking(model_type=args.model_type)

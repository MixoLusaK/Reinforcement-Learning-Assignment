# Dependencies
import argparse
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
import os
import sys
import numpy as np
import imageio
import json
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import gymnasium as gym
from shimmy import GymV21CompatibilityV0
import gym as old_gym
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    import fix_numpy_compat
except ImportError:
    pass

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


class DQN_Testing:
    """
    Class to test DQN models on Crafter environment.

    CRITICAL: Tests are conducted on the STANDARD environment (without reward shaping)
    to comply with restriction 5c: "for all evaluation you do, please include
    performance on the standard rewards and achievement unlock rates."
    """

    def __init__(self, model_path, num_episodes=100, video_dir="./crafter_videos/",
                 results_dir="./results/", plots_dir="./plots/",
                 use_preprocessing=False):
        self.model_path = model_path
        self.num_episodes = num_episodes
        self.video_dir = video_dir
        self.results_dir = results_dir
        self.plots_dir = plots_dir
        self.use_preprocessing = use_preprocessing

        # Create directories
        os.makedirs(self.video_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.plots_dir, exist_ok=True)

        # All possible achievements in Crafter
        self.all_achievements = [
            "collect_wood", "collect_stone", "collect_coal", "collect_iron",
            "collect_diamond", "collect_sapling", "collect_drink", "place_table",
            "place_plant", "place_stone", "place_furnace", "make_wood_pickaxe",
            "make_stone_pickaxe", "make_iron_pickaxe", "make_wood_sword",
            "make_stone_sword", "make_iron_sword", "defeat_zombie", "defeat_skeleton",
            "eat_cow", "eat_plant", "wake_up"
        ]

    def _make_test_env(self, episode_num=None, record_video=False):
        """
        Create a testing environment compatible with the training setup.

        Args:
            episode_num: Episode number for naming video files
            record_video: Whether to record video for this episode

        Returns:
            Wrapped Crafter environment
        """
        # Create base Crafter environment (standard rewards, no reward shaping)
        env = crafter.Env()

        # Use Crafter's built-in Recorder for video and stats
        if record_video and episode_num is not None:
            video_dir = os.path.join(self.video_dir, f"episode_{episode_num}")
            env = crafter.Recorder(
                env,
                video_dir,
                save_stats=True,
                save_video=True,
                save_episode=True
            )
        else:
            # Just record stats without video
            env = crafter.Recorder(
                env,
                self.results_dir,
                save_stats=True,
                save_video=False,
                save_episode=False
            )

        # Apply GymV21 compatibility wrapper (matches training setup)
        env = GymV21CompatibilityV0(env=env)

        # Apply preprocessing if required (matches training setup)
        if self.use_preprocessing:
            env = GrayscaleNormalizeWrapper(env)

        return env

    def test_dqn_model(self, model_type='baseline'):
        """
        Comprehensive evaluation of DQN model.

        Args:
            model_type: 'baseline', 'reward_shaped', or 'preprocessed_shaped'
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
            'preprocessed_shaped': 'DQN Preprocessed + Reward Shaped'
        }.get(model_type, f'DQN {model_type.title()}')

        print(f"\nTesting {model_name_display} Model over {self.num_episodes} episodes...")
        print("=" * 70)
        print("EVALUATION ON STANDARD REWARDS")
        if self.use_preprocessing:
            print("Using preprocessing: Grayscale + Normalization")
        if model_type in ['reward_shaped', 'preprocessed_shaped']:
            print("Model was trained WITH improvements, but evaluated WITHOUT reward shaping.")
        print("=" * 70)

        first_high_achievement_recorded = False

        for episode in range(self.num_episodes):
            # Determine if we should record video for this episode
            record_video = not first_high_achievement_recorded

            env = self._make_test_env(episode_num=episode + 1, record_video=record_video)

            obs, info = env.reset()
            done = False

            episode_reward = 0
            step = 0
            episode_achievements = set()

            while not done:
                # Predict action
                action, _ = model.predict(obs, deterministic=True)
                action_counts[int(action)] += 1

                # Step environment
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

                episode_reward += reward
                step += 1

                # Track achievements (info might have 'semantic' with achievements)
                if 'achievements' in info:
                    for achievement, unlocked in info['achievements'].items():
                        if unlocked and achievement not in episode_achievements:
                            achievement_unlocks[achievement] += 1
                            episode_achievements.add(achievement)
                elif 'semantic' in info and isinstance(info['semantic'], dict):
                    # Some Crafter versions use 'semantic' key
                    for achievement, unlocked in info['semantic'].items():
                        if unlocked and achievement not in episode_achievements:
                            achievement_unlocks[achievement] += 1
                            episode_achievements.add(achievement)

            episode_rewards.append(episode_reward)
            episode_lengths.append(step)
            achievement_per_episode.append(len(episode_achievements))

            # Mark that we recorded a high achievement episode
            num_achievements = len(episode_achievements)
            if not first_high_achievement_recorded and num_achievements >= 11 and record_video:
                print(f"\n{'=' * 70}")
                print(f"🎉 HIGH ACHIEVEMENT EPISODE DETECTED! ({num_achievements} achievements)")
                print(f"Video saved to: {os.path.join(self.video_dir, f'episode_{episode + 1}')}")
                print(f"{'=' * 70}\n")
                first_high_achievement_recorded = True

            status_marker = ' ⭐ RECORDED!' if (not first_high_achievement_recorded and num_achievements >= 11 and record_video) else ''
            print(f"Episode {episode + 1:3d}/{self.num_episodes}: "
                  f"Reward={episode_reward:6.2f}, "
                  f"Steps={step:4d}, "
                  f"Achievements={len(episode_achievements):2d}"
                  f"{status_marker}")

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
        print(f"\nPerformance Metrics (STANDARD REWARDS):")
        print(f"  Average Cumulative Reward:     {avg_reward:8.2f} ± {std_reward:.2f}")
        print(f"  Average Survival Time:         {avg_survival_time:8.2f} ± {std_survival_time:.2f} steps")
        print(f"  Average Achievements/Episode:  {avg_achievements:8.2f}")
        print(f"  Geometric Mean of Achievements: {geometric_mean:7.4f}")

        print(f"\nAchievement Unlock Rates (All {len(self.all_achievements)} Achievements):")
        sorted_achievements = sorted(achievement_rates.items(), key=lambda x: x[1], reverse=True)
        for achievement, rate in sorted_achievements:
            count = achievement_unlocks.get(achievement, 0)
            bar = "█" * int(rate * 50)
            bar = bar.ljust(50)
            print(f"  {achievement:30s}: {rate * 100:5.1f}% [{bar}] ({count:3d}/{self.num_episodes})")

        # Create visualizations
        print("\nGenerating visualizations...")
        self.create_visualizations(
            episode_rewards, episode_lengths, achievement_per_episode,
            achievement_rates, action_counts, model_name=model_type
        )

        # Save results
        results = {
            "model": model_name_display,
            "num_episodes": self.num_episodes,
            "timestamp": datetime.now().isoformat(),
            "evaluation_type": "STANDARD_REWARDS",
            "preprocessing": self.use_preprocessing,
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

        results_path = os.path.join(self.results_dir, f"dqn_{model_type}_results.json")
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"✓ Results saved to: {results_path}")
        print("=" * 70)

        return results

    def create_visualizations(self, episode_rewards, episode_lengths,
                              achievements_per_episode, achievement_rates,
                              action_counts, model_name="model"):
        """Create comprehensive visualization plots"""

        sns.set_style("whitegrid")

        # Episode metrics plot
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        axes[0, 0].plot(episode_rewards, linewidth=1, alpha=0.7, color='blue')
        axes[0, 0].axhline(np.mean(episode_rewards), color='red', linestyle='--',
                           label=f'Mean: {np.mean(episode_rewards):.2f}')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Cumulative Reward (Standard)')
        axes[0, 0].set_title(f'Episode Rewards - {model_name.title()} Model')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        axes[0, 1].plot(episode_lengths, linewidth=1, alpha=0.7, color='green')
        axes[0, 1].axhline(np.mean(episode_lengths), color='red', linestyle='--',
                           label=f'Mean: {np.mean(episode_lengths):.2f}')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Steps')
        axes[0, 1].set_title(f'Survival Time - {model_name.title()} Model')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        axes[1, 0].plot(achievements_per_episode, linewidth=1, alpha=0.7, color='purple')
        axes[1, 0].axhline(np.mean(achievements_per_episode), color='red', linestyle='--',
                           label=f'Mean: {np.mean(achievements_per_episode):.2f}')
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Achievements')
        axes[1, 0].set_title(f'Achievements per Episode - {model_name.title()}')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        axes[1, 1].hist(episode_rewards, bins=20, color='blue', alpha=0.7, edgecolor='black')
        axes[1, 1].axvline(np.mean(episode_rewards), color='red', linestyle='--',
                           label=f'Mean: {np.mean(episode_rewards):.2f}')
        axes[1, 1].set_xlabel('Cumulative Reward')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title(f'Reward Distribution - {model_name.title()}')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, f'episode_metrics_{model_name}.png'),
                    dpi=300, bbox_inches='tight')
        print(f"✓ Saved: episode_metrics_{model_name}.png")
        plt.close()

        # Achievement unlock rates
        fig, ax = plt.subplots(figsize=(12, max(6, len(achievement_rates) * 0.3)))
        sorted_achievements = sorted(achievement_rates.items(), key=lambda x: x[1], reverse=True)
        achievements, rates = zip(*sorted_achievements)

        colors = ['skyblue' if rate > 0 else 'lightgray' for rate in rates]
        bars = ax.barh(range(len(achievements)), [r * 100 for r in rates],
                       color=colors, edgecolor='black')
        ax.set_yticks(range(len(achievements)))
        ax.set_yticklabels(achievements)
        ax.set_xlabel('Unlock Rate (%)')
        ax.set_title(f'Achievement Unlock Rates - {model_name.title()}')
        ax.grid(axis='x', alpha=0.3)

        for i, (bar, rate) in enumerate(zip(bars, rates)):
            if rate > 0:
                ax.text(rate * 100 + 1, i, f'{rate * 100:.1f}%', va='center', fontweight='bold')

        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, f'achievement_rates_{model_name}.png'),
                    dpi=300, bbox_inches='tight')
        print(f"✓ Saved: achievement_rates_{model_name}.png")
        plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test DQN models on Crafter')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the trained model')
    parser.add_argument('--model_type', type=str,
                        choices=['baseline', 'reward_shaped', 'preprocessed_shaped'],
                        default='baseline', help='Type of model to test')
    parser.add_argument('--num_episodes', type=int, default=200,
                        help='Number of episodes to test')
    parser.add_argument('--use_preprocessing', action='store_true',
                        help='Use grayscale+normalization preprocessing (required for preprocessed_shaped model)')
    parser.add_argument('--video_dir', type=str, default='./crafter_videos/',
                        help='Directory to save videos')
    parser.add_argument('--results_dir', type=str, default='./results/',
                        help='Directory to save results')
    parser.add_argument('--plots_dir', type=str, default='./plots/',
                        help='Directory to save plots')

    args = parser.parse_args()

    # Auto-detect preprocessing for 'preprocessed_shaped' model
    if args.model_type == 'preprocessed_shaped':
        args.use_preprocessing = True
        print(f"Note: Automatically enabled preprocessing for '{args.model_type}' model")

    tester = DQN_Testing(
        model_path=args.model_path,
        num_episodes=args.num_episodes,
        video_dir=args.video_dir,
        results_dir=args.results_dir,
        plots_dir=args.plots_dir,
        use_preprocessing=args.use_preprocessing
    )

    # Run test
    results = tester.test_dqn_model(model_type=args.model_type)
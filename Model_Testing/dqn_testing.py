# Dependencies
import argparse
from stable_baselines3 import DQN
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
import crafter as crafter_pkg
import gym as old_gym
from gym.envs.registration import register
from shimmy import GymV21CompatibilityV0

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    import fix_numpy_compat
except ImportError:
    pass  # Patch not needed or not available


# Register Crafter environment
register(
    id='CrafterPartial-v1',
    entry_point='crafter:Env',
)


class RenderableWrapper(gym.Wrapper):
    """Wrapper to ensure rendering works properly for testing"""

    def __init__(self, env):
        super().__init__(env)
        # Ensure metadata exists
        self.metadata = {'render_modes': ['rgb_array', 'human'], 'render_fps': 30}

    def reset(self, **kwargs):
        """Reset without passing unsupported kwargs to old Gym env"""
        # Remove seed and options that old Gym doesn't support
        kwargs.pop('seed', None)
        kwargs.pop('options', None)
        obs = self.env.reset(**kwargs)
        # Return just obs (old Gym format)
        return obs

    def render(self, mode='rgb_array'):
        """Render with fallback to unwrapped environment"""
        try:
            # Try to get the base Crafter environment
            base_env = self.unwrapped
            if hasattr(base_env, 'render'):
                return base_env.render(mode=mode)
            return None
        except Exception as e:
            print(f"Render warning: {e}")
            return None


class DQN_Testing:
    """
    Class to test DQN models on Crafter environment.

    CRITICAL: Tests are conducted on the STANDARD environment (without reward shaping)
    to comply with restriction 5c: "for all evaluation you do, please include
    performance on the standard rewards and achievement unlock rates."
    """

    def __init__(self, model_path, num_episodes=100, video_dir="./crafter_videos/",
                 results_dir="./results/", plots_dir="./plots/"):
        self.model_path = model_path
        self.num_episodes = num_episodes
        self.video_dir = video_dir
        self.results_dir = results_dir
        self.plots_dir = plots_dir

        # Create directories if they don't exist
        os.makedirs(self.video_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.plots_dir, exist_ok=True)

        # Define all possible achievements in Crafter for complete reporting
        self.all_achievements = [
            "collect_wood", "collect_stone", "collect_coal", "collect_iron",
            "collect_diamond", "collect_sapling", "collect_drink", "place_table",
            "place_plant", "place_stone", "place_furnace", "make_wood_pickaxe",
            "make_stone_pickaxe", "make_iron_pickaxe", "make_wood_sword",
            "make_stone_sword", "make_iron_sword", "defeat_zombie", "defeat_skeleton",
            "eat_cow", "eat_plant", "wake_up"
        ]

    def _make_test_env(self):
        """
        Create a testing-specific environment with rendering support.
        Uses standard Crafter (no reward shaping) for fair evaluation.
        """
        # Create base Crafter environment
        env = old_gym.make("CrafterPartial-v1")

        # Wrap with our renderable wrapper for proper rendering
        env = RenderableWrapper(env)

        # Convert to Gymnasium API
        env = GymV21CompatibilityV0(env=env)

        return env

    def test_dqn_baseline(self):
        """
        Comprehensive evaluation of DQN baseline model.

        CRITICAL: Evaluates on STANDARD rewards (no shaping) as required by restriction 5c.
        """
        print(f"Loading model from: {self.model_path}")
        model = DQN.load(self.model_path)

        # Metrics storage
        episode_rewards = []
        episode_lengths = []
        achievement_unlocks = defaultdict(int)
        achievement_per_episode = []
        action_counts = defaultdict(int)

        print(f"\nTesting DQN Baseline Model over {self.num_episodes} episodes...")
        print("=" * 70)
        print("EVALUATION ON STANDARD REWARDS (No Reward Shaping)")
        print("=" * 70)

        for episode in range(self.num_episodes):
            # CRITICAL: Use standard environment (no reward shaping) for fair evaluation
            # FIX: Use special testing environment with rendering support
            env = self._make_test_env()

            # Handle both Gym and Gymnasium reset formats
            reset_result = env.reset()
            if isinstance(reset_result, tuple) and len(reset_result) == 2:
                obs, info = reset_result
            else:
                obs = reset_result
                info = {}
            terminated = False
            truncated = False
            episode_reward = 0
            step = 0
            episode_achievements = set()

            # Record video for first, middle, and last episodes
            record_video = (episode == 0 or episode == self.num_episodes // 2 or episode == self.num_episodes - 1)
            episode_frames = []

            while not (terminated or truncated):
                # Render frame for video
                if record_video:
                    try:
                        frame = env.render()
                        if frame is not None:
                            episode_frames.append(frame)
                    except Exception as e:
                        # Only print warning once per episode
                        if len(episode_frames) == 0:
                            print(f"  Note: Rendering disabled for this episode ({e})")
                        record_video = False  # Disable recording for this episode

                # Predict action
                action, _ = model.predict(obs, deterministic=True)
                action_counts[int(action)] += 1

                # Gymnasium API: returns (obs, reward, terminated, truncated, info)
                step_result = env.step(action)
                if len(step_result) == 5:
                    obs, reward, terminated, truncated, info = step_result
                elif len(step_result) == 4:
                    obs, reward, done, info = step_result
                    terminated = done
                    truncated = False
                else:
                    raise ValueError(f"Unexpected step result: {step_result}")

                episode_reward += reward  # This is the STANDARD reward
                step += 1

                # Track achievement unlocks
                if 'achievements' in info:
                    for achievement, unlocked in info['achievements'].items():
                        if unlocked and achievement not in episode_achievements:
                            achievement_unlocks[achievement] += 1
                            episode_achievements.add(achievement)

            episode_rewards.append(episode_reward)
            episode_lengths.append(step)
            achievement_per_episode.append(len(episode_achievements))

            # Save video with Windows-compatible settings
            if record_video and episode_frames:
                video_name = f"dqn_baseline_episode_{episode + 1}.mp4"
                video_path = os.path.join(self.video_dir, video_name)

                try:
                    # Windows Media Player compatible settings
                    imageio.mimsave(
                        video_path,
                        episode_frames,
                        fps=30,
                        codec='libx264',
                        quality=8,
                        pixelformat='yuv420p',
                    )
                    print(f"✓ Video saved: {video_name}")
                except Exception as e:
                    print(f"✗ Error saving MP4: {e}")
                    # Fallback to GIF if MP4 fails
                    try:
                        gif_path = os.path.join(self.video_dir, f"dqn_baseline_episode_{episode + 1}.gif")
                        imageio.mimsave(gif_path, episode_frames, fps=30)
                        print(f"✓ Video saved as GIF: dqn_baseline_episode_{episode + 1}.gif")
                    except Exception as e2:
                        print(f"✗ Error saving GIF: {e2}")

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

        # Calculate achievement unlock rates - include ALL achievements
        achievement_rates = {}
        for achievement in self.all_achievements:
            count = achievement_unlocks.get(achievement, 0)
            achievement_rates[achievement] = count / self.num_episodes

        # Calculate geometric mean of achievement unlock rates
        epsilon = 1e-10
        non_zero_rates = [rate + epsilon for rate in achievement_rates.values() if rate > 0]
        if non_zero_rates:
            geometric_mean = np.exp(np.mean(np.log(non_zero_rates))) - epsilon
        else:
            geometric_mean = 0.0

        # Print summary
        print("\n" + "=" * 70)
        print("EVALUATION SUMMARY - DQN BASELINE")
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

        print(f"\nAction Distribution (Total actions: {sum(action_counts.values())}):")
        total_actions = sum(action_counts.values())
        for action_id in sorted(action_counts.keys()):
            count = action_counts[action_id]
            percentage = (count / total_actions) * 100
            print(f"  Action {action_id:2d}: {count:7d} ({percentage:5.2f}%)")

        # Create visualizations
        print("\nGenerating visualizations...")
        self.create_visualizations(
            episode_rewards,
            episode_lengths,
            achievement_per_episode,
            achievement_rates,
            action_counts,
            model_name="baseline"
        )

        # Save results to JSON
        results = {
            "model": "DQN_Baseline",
            "num_episodes": self.num_episodes,
            "timestamp": datetime.now().isoformat(),
            "evaluation_type": "STANDARD_REWARDS",  # Explicitly state this
            "metrics": {
                "average_reward": float(avg_reward),
                "std_reward": float(std_reward),
                "average_survival_time": float(avg_survival_time),
                "std_survival_time": float(std_survival_time),
                "average_achievements_per_episode": float(avg_achievements),
                "geometric_mean_achievements": float(geometric_mean),
                "total_unique_achievements": len([a for a in achievement_rates.values() if a > 0]),
                "total_possible_achievements": len(self.all_achievements)
            },
            "achievement_unlock_rates": achievement_rates,
            "action_distribution": {k: int(v) for k, v in action_counts.items()},
            "episode_rewards": episode_rewards,
            "episode_lengths": episode_lengths,
            "achievements_per_episode": achievement_per_episode
        }

        results_path = os.path.join(self.results_dir, "dqn_baseline_results.json")
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"✓ Results saved to: {results_path}")
        print("=" * 70)

        return results

    def test_dqn_improved(self):
        """
        Comprehensive evaluation of DQN improved model.

        CRITICAL: Even though the model was trained WITH reward shaping,
        we evaluate on STANDARD rewards (no shaping) as required by restriction 5c.
        This ensures fair comparison with the baseline model.
        """
        print(f"Loading model from: {self.model_path}")
        model = DQN.load(self.model_path)

        # Metrics storage
        episode_rewards = []
        episode_lengths = []
        achievement_unlocks = defaultdict(int)
        achievement_per_episode = []
        action_counts = defaultdict(int)

        print(f"\nTesting DQN Improved Model over {self.num_episodes} episodes...")
        print("=" * 70)
        print("EVALUATION ON STANDARD REWARDS (No Reward Shaping)")
        print("Model was trained WITH belief reward shaping, but evaluated WITHOUT it.")
        print("=" * 70)

        for episode in range(self.num_episodes):
            # CRITICAL: Use standard environment (no reward shaping) for fair evaluation
            # FIX: Use special testing environment with rendering support
            # Even though this model was trained with BeliefRewardWrapper, we evaluate without it
            env = self._make_test_env()

            # Handle both Gym and Gymnasium reset formats
            reset_result = env.reset()
            if isinstance(reset_result, tuple):
                obs, info = reset_result
            else:
                obs = reset_result
                info = {}
            terminated = False
            truncated = False
            episode_reward = 0
            step = 0
            episode_achievements = set()

            # Record video for first, middle, and last episodes
            record_video = (episode == 0 or episode == self.num_episodes // 2 or episode == self.num_episodes - 1)
            episode_frames = []

            while not (terminated or truncated):
                # Render frame for video
                if record_video:
                    try:
                        frame = env.render()
                        if frame is not None:
                            episode_frames.append(frame)
                    except Exception as e:
                        # Only print warning once per episode
                        if len(episode_frames) == 0:
                            print(f"  Note: Rendering disabled for this episode ({e})")
                        record_video = False  # Disable recording for this episode

                # Predict action
                action, _ = model.predict(obs, deterministic=True)
                action_counts[int(action)] += 1

                # Gymnasium API
                step_result = env.step(action)
                if len(step_result) == 5:
                    obs, reward, terminated, truncated, info = step_result
                elif len(step_result) == 4:
                    obs, reward, done, info = step_result
                    terminated = done
                    truncated = False
                else:
                    raise ValueError(f"Unexpected step result: {step_result}")

                episode_reward += reward  # This is the STANDARD reward (no shaping)
                step += 1

                # Track achievement unlocks
                if 'achievements' in info:
                    for achievement, unlocked in info['achievements'].items():
                        if unlocked and achievement not in episode_achievements:
                            achievement_unlocks[achievement] += 1
                            episode_achievements.add(achievement)

            episode_rewards.append(episode_reward)
            episode_lengths.append(step)
            achievement_per_episode.append(len(episode_achievements))

            # Save video for recorded episodes
            if record_video and episode_frames:
                video_name = f"dqn_improved_episode_{episode + 1}.mp4"
                video_path = os.path.join(self.video_dir, video_name)

                try:
                    imageio.mimsave(
                        video_path,
                        episode_frames,
                        fps=30,
                        codec='libx264',
                        quality=8,
                        pixelformat='yuv420p',
                    )
                    print(f"✓ Video saved: {video_name}")
                except Exception as e:
                    print(f"✗ Error saving MP4: {e}")
                    try:
                        gif_path = os.path.join(self.video_dir, f"dqn_improved_episode_{episode + 1}.gif")
                        imageio.mimsave(gif_path, episode_frames, fps=30)
                        print(f"✓ Video saved as GIF: dqn_improved_episode_{episode + 1}.gif")
                    except Exception as e2:
                        print(f"✗ Error saving GIF: {e2}")

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

        # Calculate achievement unlock rates
        achievement_rates = {}
        for achievement in self.all_achievements:
            count = achievement_unlocks.get(achievement, 0)
            achievement_rates[achievement] = count / self.num_episodes

        # Calculate geometric mean
        epsilon = 1e-10
        non_zero_rates = [rate + epsilon for rate in achievement_rates.values() if rate > 0]
        if non_zero_rates:
            geometric_mean = np.exp(np.mean(np.log(non_zero_rates))) - epsilon
        else:
            geometric_mean = 0.0

        # Print summary
        print("\n" + "=" * 70)
        print("EVALUATION SUMMARY - DQN IMPROVED")
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

        print(f"\nAction Distribution (Total actions: {sum(action_counts.values())}):")
        total_actions = sum(action_counts.values())
        for action_id in sorted(action_counts.keys()):
            count = action_counts[action_id]
            percentage = (count / total_actions) * 100
            print(f"  Action {action_id:2d}: {count:7d} ({percentage:5.2f}%)")

        # Create visualizations
        print("\nGenerating visualizations...")
        self.create_visualizations(
            episode_rewards,
            episode_lengths,
            achievement_per_episode,
            achievement_rates,
            action_counts,
            model_name="improved"
        )

        # Save results to JSON
        results = {
            "model": "DQN_Improved",
            "num_episodes": self.num_episodes,
            "timestamp": datetime.now().isoformat(),
            "evaluation_type": "STANDARD_REWARDS",  # Explicitly state this
            "training_method": "Belief Reward Shaping",
            "metrics": {
                "average_reward": float(avg_reward),
                "std_reward": float(std_reward),
                "average_survival_time": float(avg_survival_time),
                "std_survival_time": float(std_survival_time),
                "average_achievements_per_episode": float(avg_achievements),
                "geometric_mean_achievements": float(geometric_mean),
                "total_unique_achievements": len([a for a in achievement_rates.values() if a > 0]),
                "total_possible_achievements": len(self.all_achievements)
            },
            "achievement_unlock_rates": achievement_rates,
            "action_distribution": {k: int(v) for k, v in action_counts.items()},
            "episode_rewards": episode_rewards,
            "episode_lengths": episode_lengths,
            "achievements_per_episode": achievement_per_episode
        }

        results_path = os.path.join(self.results_dir, "dqn_improved_results.json")
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"✓ Results saved to: {results_path}")
        print("=" * 70)

        return results

    def compare_models(self, baseline_results, improved_results):
        """
        Create comparison visualizations between baseline and improved models.
        """
        print("\nGenerating comparison visualizations...")

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # 1. Average rewards comparison
        models = ['Baseline', 'Improved']
        avg_rewards = [
            baseline_results['metrics']['average_reward'],
            improved_results['metrics']['average_reward']
        ]
        std_rewards = [
            baseline_results['metrics']['std_reward'],
            improved_results['metrics']['std_reward']
        ]

        axes[0, 0].bar(models, avg_rewards, yerr=std_rewards, capsize=5,
                       color=['skyblue', 'lightcoral'], edgecolor='black')
        axes[0, 0].set_ylabel('Average Cumulative Reward')
        axes[0, 0].set_title('Average Reward Comparison')
        axes[0, 0].grid(axis='y', alpha=0.3)

        # 2. Average achievements comparison
        avg_achievements = [
            baseline_results['metrics']['average_achievements_per_episode'],
            improved_results['metrics']['average_achievements_per_episode']
        ]

        axes[0, 1].bar(models, avg_achievements, color=['skyblue', 'lightcoral'],
                       edgecolor='black')
        axes[0, 1].set_ylabel('Average Achievements per Episode')
        axes[0, 1].set_title('Achievement Comparison')
        axes[0, 1].grid(axis='y', alpha=0.3)

        # 3. Survival time comparison
        avg_survival = [
            baseline_results['metrics']['average_survival_time'],
            improved_results['metrics']['average_survival_time']
        ]
        std_survival = [
            baseline_results['metrics']['std_survival_time'],
            improved_results['metrics']['std_survival_time']
        ]

        axes[1, 0].bar(models, avg_survival, yerr=std_survival, capsize=5,
                       color=['skyblue', 'lightcoral'], edgecolor='black')
        axes[1, 0].set_ylabel('Average Survival Time (steps)')
        axes[1, 0].set_title('Survival Time Comparison')
        axes[1, 0].grid(axis='y', alpha=0.3)

        # 4. Geometric mean comparison
        geom_means = [
            baseline_results['metrics']['geometric_mean_achievements'],
            improved_results['metrics']['geometric_mean_achievements']
        ]

        axes[1, 1].bar(models, geom_means, color=['skyblue', 'lightcoral'],
                       edgecolor='black')
        axes[1, 1].set_ylabel('Geometric Mean of Achievement Rates')
        axes[1, 1].set_title('Overall Achievement Score')
        axes[1, 1].grid(axis='y', alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, 'model_comparison.png'),
                    dpi=300, bbox_inches='tight')
        print(f"✓ Saved: model_comparison.png")
        plt.close()

    def create_visualizations(self, episode_rewards, episode_lengths, achievements_per_episode,
                              achievement_rates, action_counts, model_name="model"):
        """Create comprehensive visualization plots"""

        sns.set_style("whitegrid")

        # 1. Rewards and Survival Time over Episodes
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Rewards over episodes
        axes[0, 0].plot(episode_rewards, linewidth=1, alpha=0.7, color='blue')
        axes[0, 0].axhline(np.mean(episode_rewards), color='red', linestyle='--',
                           label=f'Mean: {np.mean(episode_rewards):.2f}')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Cumulative Reward (Standard)')
        axes[0, 0].set_title(f'Episode Rewards - {model_name.title()} Model')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Survival time over episodes
        axes[0, 1].plot(episode_lengths, linewidth=1, alpha=0.7, color='green')
        axes[0, 1].axhline(np.mean(episode_lengths), color='red', linestyle='--',
                           label=f'Mean: {np.mean(episode_lengths):.2f}')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Steps')
        axes[0, 1].set_title(f'Survival Time per Episode - {model_name.title()} Model')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # Achievements per episode
        axes[1, 0].plot(achievements_per_episode, linewidth=1, alpha=0.7, color='purple')
        axes[1, 0].axhline(np.mean(achievements_per_episode), color='red', linestyle='--',
                           label=f'Mean: {np.mean(achievements_per_episode):.2f}')
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Number of Achievements')
        axes[1, 0].set_title(f'Achievements Unlocked per Episode - {model_name.title()} Model')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # Distribution of rewards
        axes[1, 1].hist(episode_rewards, bins=20, color='blue', alpha=0.7, edgecolor='black')
        axes[1, 1].axvline(np.mean(episode_rewards), color='red', linestyle='--',
                           label=f'Mean: {np.mean(episode_rewards):.2f}')
        axes[1, 1].set_xlabel('Cumulative Reward (Standard)')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title(f'Distribution of Episode Rewards - {model_name.title()} Model')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, f'episode_metrics_{model_name}.png'),
                    dpi=300, bbox_inches='tight')
        print(f"✓ Saved: episode_metrics_{model_name}.png")
        plt.close()

        # 2. Achievement unlock rates
        fig, ax = plt.subplots(figsize=(12, max(6, len(achievement_rates) * 0.3)))
        sorted_achievements = sorted(achievement_rates.items(), key=lambda x: x[1], reverse=True)
        achievements, rates = zip(*sorted_achievements)

        colors = ['skyblue' if rate > 0 else 'lightgray' for rate in rates]

        bars = ax.barh(range(len(achievements)), [r * 100 for r in rates],
                       color=colors, edgecolor='black')
        ax.set_yticks(range(len(achievements)))
        ax.set_yticklabels(achievements)
        ax.set_xlabel('Unlock Rate (%)')
        ax.set_title(f'Achievement Unlock Rates - {model_name.title()} Model (Standard Rewards)')
        ax.grid(axis='x', alpha=0.3)

        for i, (bar, rate) in enumerate(zip(bars, rates)):
            if rate > 0:
                ax.text(rate * 100 + 1, i, f'{rate * 100:.1f}%', va='center', fontweight='bold')
            else:
                ax.text(1, i, '0.0%', va='center', color='gray', fontstyle='italic')

        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, f'achievement_rates_{model_name}.png'),
                    dpi=300, bbox_inches='tight')
        print(f"✓ Saved: achievement_rates_{model_name}.png")
        plt.close()

        # 3. Action distribution
        if action_counts:
            fig, ax = plt.subplots(figsize=(10, 6))
            actions = sorted(action_counts.keys())
            counts = [action_counts[a] for a in actions]
            total = sum(counts)
            percentages = [c / total * 100 for c in counts]

            bars = ax.bar(actions, percentages, color='coral', edgecolor='black')
            ax.set_xlabel('Action ID')
            ax.set_ylabel('Percentage (%)')
            ax.set_title(f'Action Distribution - {model_name.title()} Model')
            ax.grid(axis='y', alpha=0.3)

            for bar, pct in zip(bars, percentages):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2., height,
                        f'{pct:.1f}%', ha='center', va='bottom', fontsize=8)

            plt.tight_layout()
            plt.savefig(os.path.join(self.plots_dir, f'action_distribution_{model_name}.png'),
                        dpi=300, bbox_inches='tight')
            print(f"✓ Saved: action_distribution_{model_name}.png")
            plt.close()


if __name__ == "__main__":
    # Example usage
    parser = argparse.ArgumentParser(description='Test DQN models on Crafter')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model')
    parser.add_argument('--model_type', type=str, choices=['baseline', 'improved'],
                        default='baseline', help='Type of model to test')
    parser.add_argument('--num_episodes', type=int, default=100,
                        help='Number of episodes to test')
    parser.add_argument('--video_dir', type=str, default='./crafter_videos/',
                        help='Directory to save videos')
    parser.add_argument('--results_dir', type=str, default='./results/',
                        help='Directory to save results')
    parser.add_argument('--plots_dir', type=str, default='./plots/',
                        help='Directory to save plots')

    args = parser.parse_args()

    tester = DQN_Testing(
        model_path=args.model_path,
        num_episodes=args.num_episodes,
        video_dir=args.video_dir,
        results_dir=args.results_dir,
        plots_dir=args.plots_dir
    )

    # Run appropriate test
    if args.model_type == 'baseline':
        results = tester.test_dqn_baseline()
    elif args.model_type == 'improved':
        results = tester.test_dqn_improved()
# Dependencies
import argparse
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
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

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    import fix_numpy_compat
except ImportError:
    pass


class CrafterGymnasiumWrapper(gym.Env):
    """Wrapper to make Crafter compatible with Gymnasium API"""

    def __init__(self):
        import crafter
        self._env = crafter.Env()

        # Define spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255,
            shape=self._env.observation_space.shape,
            dtype=np.uint8
        )
        self.action_space = gym.spaces.Discrete(self._env.action_space.n)

        # Metadata for rendering
        self.metadata = {'render_modes': ['rgb_array', 'human'], 'render_fps': 30}
        self.render_mode = 'rgb_array'

        self._last_obs = None

    def reset(self, seed=None, options=None):
        """Reset environment with Gymnasium API"""
        if seed is not None:
            np.random.seed(seed)

        obs = self._env.reset()
        self._last_obs = obs

        # Return (observation, info) tuple as required by Gymnasium
        info = {}
        return obs, info

    def step(self, action):
        """Step with Gymnasium API (5-tuple return)"""
        obs, reward, done, info = self._env.step(action)
        self._last_obs = obs

        # Convert to Gymnasium format: (obs, reward, terminated, truncated, info)
        terminated = done
        truncated = False  # Crafter doesn't use truncation

        return obs, reward, terminated, truncated, info

    def render(self):
        """Render the current state"""
        return self.get_frame()

    def get_frame(self):
        """Get RGB frame for video recording"""
        try:
            # Method 1: Try Crafter's internal image
            if hasattr(self._env, '_image') and self._env._image is not None:
                return self._env._image.copy()

            # Method 2: Use last observation
            if self._last_obs is not None:
                return self._last_obs.copy()

            # Method 3: Try _view attribute
            if hasattr(self._env, '_view') and self._env._view is not None:
                return self._env._view.copy()

            return None
        except Exception as e:
            print(f"Frame capture error: {e}")
            return None

    def close(self):
        """Close the environment"""
        if hasattr(self._env, 'close'):
            self._env.close()


class VecFrameCaptureWrapper:
    """
    Wrapper to capture frames from VecFrameStack environment for video recording.
    Extracts only the most recent frame from stacked observations.
    """

    def __init__(self, vec_env):
        self.vec_env = vec_env
        self._last_frame = None

    def __getattr__(self, name):
        """Delegate attribute access to wrapped environment"""
        return getattr(self.vec_env, name)

    def reset(self):
        """Reset and capture initial frame"""
        obs = self.vec_env.reset()
        self._capture_current_frame(obs)
        return obs

    def step(self, actions):
        """Step and capture frame"""
        obs, rewards, dones, infos = self.vec_env.step(actions)
        self._capture_current_frame(obs)
        return obs, rewards, dones, infos

    def _capture_current_frame(self, stacked_obs):
        """Extract the most recent frame from stacked observation"""
        # stacked_obs shape: (1, 12, 64, 64) for batch
        # We want the last 3 channels (most recent frame)
        try:
            obs = stacked_obs[0]  # Remove batch dimension: (12, 64, 64)

            # Last 3 channels are the most recent frame
            recent_frame = obs[-3:, :, :]  # Shape: (3, 64, 64)

            # Convert from (C, H, W) to (H, W, C) for video
            frame = np.transpose(recent_frame, (1, 2, 0))  # Shape: (64, 64, 3)

            self._last_frame = frame.astype(np.uint8)
        except Exception as e:
            print(f"Frame capture error: {e}")
            self._last_frame = None

    def get_current_frame(self):
        """Get the most recently captured frame"""
        return self._last_frame


class DQN_Testing:
    """
    Class to test DQN models on Crafter environment.

    CRITICAL: Tests are conducted on the STANDARD environment (without reward shaping)
    to comply with restriction 5c: "for all evaluation you do, please include
    performance on the standard rewards and achievement unlock rates."
    """

    def __init__(self, model_path, num_episodes=100, video_dir="./crafter_videos/",
                 results_dir="./results/", plots_dir="./plots/", use_frame_stacking=False,
                 n_stack=4):
        self.model_path = model_path
        self.num_episodes = num_episodes
        self.video_dir = video_dir
        self.results_dir = results_dir
        self.plots_dir = plots_dir
        self.use_frame_stacking = use_frame_stacking
        self.n_stack = n_stack

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

    def _make_test_env(self):
        """
        Create a testing environment.

        If use_frame_stacking=True, wraps the environment with frame stacking
        to match the training configuration.
        """
        # Create base environment
        base_env = CrafterGymnasiumWrapper()

        if self.use_frame_stacking:
            # Wrap in DummyVecEnv (required for VecFrameStack)
            vec_env = DummyVecEnv([lambda: base_env])

            # Apply frame stacking
            stacked_env = VecFrameStack(vec_env, n_stack=self.n_stack)

            # Wrap with frame capture wrapper for video recording
            env = VecFrameCaptureWrapper(stacked_env)

            return env
        else:
            return base_env

    def _save_video(self, frames, episode_num, model_name):
        """Save video with fallback options"""
        if not frames:
            return

        video_name = f"{model_name}_episode_{episode_num}"
        mp4_saved = False

        # Method 1: Try with imageio-ffmpeg (if available)
        try:
            video_path = os.path.join(self.video_dir, f"{video_name}.mp4")
            imageio.mimsave(
                video_path,
                frames,
                fps=30,
                codec='libx264',
                quality=8,
                pixelformat='yuv420p',
            )
            print(f"✓ Video saved: {video_name}.mp4 ({len(frames)} frames)")
            mp4_saved = True
        except Exception as e:
            pass

        # Method 2: Try with system ffmpeg via subprocess
        if not mp4_saved:
            try:
                import subprocess
                import tempfile

                # Save frames as temporary images
                with tempfile.TemporaryDirectory() as tmpdir:
                    # Save frames
                    for i, frame in enumerate(frames):
                        frame_path = os.path.join(tmpdir, f"frame_{i:06d}.png")
                        imageio.imwrite(frame_path, frame)

                    # Use ffmpeg to create video
                    video_path = os.path.join(self.video_dir, f"{video_name}.mp4")
                    cmd = [
                        'ffmpeg', '-y',  # Overwrite output
                        '-framerate', '30',
                        '-i', os.path.join(tmpdir, 'frame_%06d.png'),
                        '-c:v', 'libx264',
                        '-pix_fmt', 'yuv420p',
                        '-crf', '23',
                        video_path
                    ]

                    result = subprocess.run(cmd, capture_output=True, text=True)
                    if result.returncode == 0:
                        print(f"✓ Video saved: {video_name}.mp4 ({len(frames)} frames)")
                        mp4_saved = True

            except Exception as e:
                pass

        # Fallback to GIF if MP4 fails
        if not mp4_saved:
            try:
                gif_path = os.path.join(self.video_dir, f"{video_name}.gif")
                imageio.mimsave(gif_path, frames, fps=30, loop=0)
                print(f"✓ Video saved as GIF: {video_name}.gif ({len(frames)} frames)")
                print("  Note: Install ffmpeg for MP4 support: conda install -c conda-forge ffmpeg")
            except Exception as e:
                print(f"✗ Failed to save video: {e}")

    def test_dqn_model(self, model_type='baseline'):
        """
        Comprehensive evaluation of DQN model.

        Args:
            model_type: 'baseline', 'framed', or 'improved'
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
            'framed': 'DQN Frame Stacking',
            'improved': 'DQN Fully Improved'
        }.get(model_type, f'DQN {model_type.title()}')

        print(f"\nTesting {model_name_display} Model over {self.num_episodes} episodes...")
        print("=" * 70)
        print("EVALUATION ON STANDARD REWARDS")
        if self.use_frame_stacking:
            print(f"Using frame stacking: {self.n_stack} frames")
        if model_type in ['improved', 'framed']:
            print("Model was trained WITH improvements, but evaluated WITHOUT reward shaping.")
        print("=" * 70)

        first_high_achievement_recorded = False

        for episode in range(self.num_episodes):
            env = self._make_test_env()

            # Handle both regular and vectorized environments
            if self.use_frame_stacking:
                obs = env.reset()
                terminated = np.array([False])
                truncated = np.array([False])
            else:
                obs, info = env.reset()
                terminated = False
                truncated = False

            episode_reward = 0
            step = 0
            episode_achievements = set()

            # Always collect frames
            episode_frames = []

            # Capture initial frame
            if self.use_frame_stacking:
                frame = env.get_current_frame()
            else:
                frame = env.get_frame()

            if frame is not None:
                episode_frames.append(frame)

            while not (terminated.any() if isinstance(terminated, np.ndarray) else terminated):
                # Predict action
                action, _ = model.predict(obs, deterministic=True)

                if self.use_frame_stacking:
                    # For vectorized env, action is already an array
                    action_counts[int(action[0])] += 1
                else:
                    action_counts[int(action)] += 1

                # Step environment
                if self.use_frame_stacking:
                    obs, reward, done, infos = env.step(action)
                    reward = reward[0]
                    terminated = done
                    truncated = np.array([False])
                    info = infos[0]
                else:
                    obs, reward, terminated, truncated, info = env.step(action)

                episode_reward += reward
                step += 1

                # Capture frame
                if self.use_frame_stacking:
                    frame = env.get_current_frame()
                else:
                    frame = env.get_frame()

                if frame is not None:
                    episode_frames.append(frame)

                # Track achievements
                if 'achievements' in info:
                    for achievement, unlocked in info['achievements'].items():
                        if unlocked and achievement not in episode_achievements:
                            achievement_unlocks[achievement] += 1
                            episode_achievements.add(achievement)

            episode_rewards.append(episode_reward)
            episode_lengths.append(step)
            achievement_per_episode.append(len(episode_achievements))

            # Save video ONLY for the first episode with 11+ achievements
            num_achievements = len(episode_achievements)
            if not first_high_achievement_recorded and num_achievements >= 11 and episode_frames:
                print(f"\n{'=' * 70}")
                print(f"🎉 HIGH ACHIEVEMENT EPISODE DETECTED! ({num_achievements} achievements)")
                print(f"{'=' * 70}")
                self._save_video(episode_frames, episode + 1,
                                 f"dqn_{model_type}_best")
                first_high_achievement_recorded = True
                print(f"{'=' * 70}\n")

            print(f"Episode {episode + 1:3d}/{self.num_episodes}: "
                  f"Reward={episode_reward:6.2f}, "
                  f"Steps={step:4d}, "
                  f"Achievements={len(episode_achievements):2d}"
                  f"{' ⭐ RECORDED!' if (not first_high_achievement_recorded and num_achievements >= 11) else ''}")

            # Close environment
            if self.use_frame_stacking:
                env.close()
            else:
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
            "frame_stacking": self.use_frame_stacking,
            "n_stack": self.n_stack if self.use_frame_stacking else None,
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
    parser.add_argument('--model_type', type=str, choices=['baseline', 'improved', 'framed'],
                        default='baseline', help='Type of model to test')
    parser.add_argument('--num_episodes', type=int, default=100,
                        help='Number of episodes to test')
    parser.add_argument('--use_frame_stacking', action='store_true',
                        help='Use frame stacking (required for models trained with frame stacking)')
    parser.add_argument('--n_stack', type=int, default=4,
                        help='Number of frames to stack (if using frame stacking)')
    parser.add_argument('--video_dir', type=str, default='./crafter_videos/',
                        help='Directory to save videos')
    parser.add_argument('--results_dir', type=str, default='./results/',
                        help='Directory to save results')
    parser.add_argument('--plots_dir', type=str, default='./plots/',
                        help='Directory to save plots')

    args = parser.parse_args()

    # Auto-detect frame stacking for 'framed' and 'improved' models
    if args.model_type in ['framed', 'improved']:
        args.use_frame_stacking = True
        print(f"Note: Automatically enabled frame stacking for '{args.model_type}' model")

    tester = DQN_Testing(
        model_path=args.model_path,
        num_episodes=args.num_episodes,
        video_dir=args.video_dir,
        results_dir=args.results_dir,
        plots_dir=args.plots_dir,
        use_frame_stacking=args.use_frame_stacking,
        n_stack=args.n_stack
    )

    # Run test
    results = tester.test_dqn_model(model_type=args.model_type)
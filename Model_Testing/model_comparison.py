"""
Enhanced Model Comparison Script for DQN Models
Compares baseline, frame stacking, and fully improved models.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
from scipy import stats
from datetime import datetime


class ModelComparison:
    """Compare multiple DQN model variants using saved results."""

    def __init__(self, model_paths, model_names=None, output_dir='./comparison_results/'):
        """
        Initialize comparison with paths to result files.

        Args:
            model_paths: List of paths to model results JSONs (in order: baseline, frame_stack, improved)
            model_names: List of names for each model (optional)
            output_dir: Directory to save comparison outputs
        """
        self.model_paths = [Path(p) for p in model_paths]
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Load all models
        self.models = [self._load_results(path) for path in self.model_paths]

        # Set model names
        if model_names is None:
            self.model_names = [f"Model {i+1}" for i in range(len(self.models))]
        else:
            self.model_names = model_names

        # Define colors for each model
        self.colors = ['#3498db', '#f39c12', '#e74c3c', '#9b59b6', '#27ae60'][:len(self.models)]

        print(f"\nLoaded {len(self.models)} models for comparison:")
        for name, model in zip(self.model_names, self.models):
            print(f"  {name}: {model['num_episodes']} episodes")

    def _load_results(self, path):
        """Load results from JSON file."""
        with open(path, 'r') as f:
            return json.load(f)

    def generate_full_comparison(self):
        """Generate all comparison visualizations and reports."""
        print("\n" + "=" * 70)
        print("GENERATING COMPREHENSIVE MODEL COMPARISON")
        print("=" * 70)

        # 1. Statistical comparison
        print("\n1. Performing statistical analysis...")
        self.statistical_comparison()

        # 2. Performance metrics comparison
        print("2. Creating performance metrics comparison...")
        self.plot_performance_metrics()

        # 3. Achievement comparison
        print("3. Creating achievement comparison...")
        self.plot_achievement_comparison()

        # 4. Distribution comparison
        print("4. Creating distribution comparisons...")
        self.plot_distributions()

        # 5. Episode-by-episode comparison
        print("5. Creating episode traces...")
        self.plot_episode_traces()

        # 6. Action distribution comparison
        print("6. Creating action distribution comparison...")
        self.plot_action_comparison()

        # 7. Progressive improvement visualization
        print("7. Creating progressive improvement visualization...")
        self.plot_progressive_improvement()

        # 8. Generate text report
        print("8. Generating comparison report...")
        self.generate_report()

        print("\n" + "=" * 70)
        print(f"All comparison results saved to: {self.output_dir}")
        print("=" * 70)

    def statistical_comparison(self):
        """Perform statistical tests and print results."""
        print("\n" + "=" * 70)
        print("STATISTICAL ANALYSIS")
        print("=" * 70)

        # Extract data for all models
        all_rewards = [np.array(model['episode_rewards']) for model in self.models]
        all_achievements = [np.array(model['achievements_per_episode']) for model in self.models]
        all_survival = [np.array(model['episode_lengths']) for model in self.models]

        # Reward comparison
        print("\n📊 REWARD COMPARISON")
        print("-" * 70)
        print(f"{'Model':<20} {'Mean':>12} {'Std':>12} {'vs Baseline':>15}")
        print("-" * 70)

        baseline_reward_mean = np.mean(all_rewards[0])
        for i, (name, rewards) in enumerate(zip(self.model_names, all_rewards)):
            mean_val = np.mean(rewards)
            std_val = np.std(rewards)
            if i == 0:
                improvement = "-"
            else:
                improvement = f"{((mean_val - baseline_reward_mean) / abs(baseline_reward_mean) * 100):+.2f}%"
            print(f"{name:<20} {mean_val:>12.2f} {std_val:>12.2f} {improvement:>15}")

        # Pairwise t-tests for rewards
        print("\n  Pairwise Statistical Tests (Rewards):")
        for i in range(len(self.models) - 1):
            for j in range(i + 1, len(self.models)):
                t_stat, p_value = stats.ttest_ind(all_rewards[i], all_rewards[j])
                sig = "✓ Significant" if p_value < 0.05 else "✗ Not significant"
                print(f"    {self.model_names[i]} vs {self.model_names[j]}: "
                      f"p={p_value:.6f} ({sig})")

        # Achievement comparison
        print("\n🏆 ACHIEVEMENT COMPARISON")
        print("-" * 70)
        print(f"{'Model':<20} {'Mean':>12} {'Std':>12} {'vs Baseline':>15}")
        print("-" * 70)

        baseline_ach_mean = np.mean(all_achievements[0])
        for i, (name, achievements) in enumerate(zip(self.model_names, all_achievements)):
            mean_val = np.mean(achievements)
            std_val = np.std(achievements)
            if i == 0:
                improvement = "-"
            else:
                improvement = f"{((mean_val - baseline_ach_mean) / baseline_ach_mean * 100):+.2f}%"
            print(f"{name:<20} {mean_val:>12.2f} {std_val:>12.2f} {improvement:>15}")

        # Pairwise t-tests for achievements
        print("\n  Pairwise Statistical Tests (Achievements):")
        for i in range(len(self.models) - 1):
            for j in range(i + 1, len(self.models)):
                t_stat, p_value = stats.ttest_ind(all_achievements[i], all_achievements[j])
                sig = "✓ Significant" if p_value < 0.05 else "✗ Not significant"
                print(f"    {self.model_names[i]} vs {self.model_names[j]}: "
                      f"p={p_value:.6f} ({sig})")

        # Survival time comparison
        print("\n⏱️  SURVIVAL TIME COMPARISON")
        print("-" * 70)
        print(f"{'Model':<20} {'Mean (steps)':>15} {'Std':>12} {'vs Baseline':>15}")
        print("-" * 70)

        baseline_surv_mean = np.mean(all_survival[0])
        for i, (name, survival) in enumerate(zip(self.model_names, all_survival)):
            mean_val = np.mean(survival)
            std_val = np.std(survival)
            if i == 0:
                improvement = "-"
            else:
                improvement = f"{((mean_val - baseline_surv_mean) / baseline_surv_mean * 100):+.2f}%"
            print(f"{name:<20} {mean_val:>15.2f} {std_val:>12.2f} {improvement:>15}")

        # Effect sizes
        print("\n📏 EFFECT SIZES (Cohen's d vs Baseline)")
        print("-" * 70)
        for i in range(1, len(self.models)):
            reward_effect = self._cohens_d(all_rewards[0], all_rewards[i])
            achievement_effect = self._cohens_d(all_achievements[0], all_achievements[i])
            survival_effect = self._cohens_d(all_survival[0], all_survival[i])

            print(f"\n  {self.model_names[i]}:")
            print(f"    Reward:      {reward_effect:7.4f} ({self._interpret_effect_size(reward_effect)})")
            print(f"    Achievement: {achievement_effect:7.4f} ({self._interpret_effect_size(achievement_effect)})")
            print(f"    Survival:    {survival_effect:7.4f} ({self._interpret_effect_size(survival_effect)})")

    def _cohens_d(self, group1, group2):
        """Calculate Cohen's d effect size."""
        n1, n2 = len(group1), len(group2)
        var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
        pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
        return (np.mean(group2) - np.mean(group1)) / pooled_std

    def _interpret_effect_size(self, d):
        """Interpret Cohen's d effect size."""
        abs_d = abs(d)
        if abs_d < 0.2:
            return "negligible"
        elif abs_d < 0.5:
            return "small"
        elif abs_d < 0.8:
            return "medium"
        else:
            return "large"

    def plot_performance_metrics(self):
        """Create bar chart comparison of key metrics."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))

        x = np.arange(len(self.models))
        width = 0.7 / len(self.models) if len(self.models) > 2 else 0.35

        # 1. Average Rewards
        avg_rewards = [model['metrics']['average_reward'] for model in self.models]
        std_rewards = [model['metrics']['std_reward'] for model in self.models]

        bars = axes[0, 0].bar(x, avg_rewards, width, yerr=std_rewards, capsize=8,
                              color=self.colors, edgecolor='black', linewidth=1.5,
                              label=self.model_names)
        axes[0, 0].set_ylabel('Average Cumulative Reward', fontsize=12, fontweight='bold')
        axes[0, 0].set_title('Reward Comparison', fontsize=13, fontweight='bold')
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels(self.model_names, rotation=15, ha='right')
        axes[0, 0].grid(axis='y', alpha=0.3, linestyle='--')

        for i, (bar, val, std) in enumerate(zip(bars, avg_rewards, std_rewards)):
            axes[0, 0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + std + 0.5,
                            f'{val:.1f}', ha='center', va='bottom', fontweight='bold', fontsize=9)

        # 2. Average Achievements
        avg_achievements = [model['metrics']['average_achievements_per_episode'] for model in self.models]

        bars = axes[0, 1].bar(x, avg_achievements, width, color=self.colors,
                              edgecolor='black', linewidth=1.5)
        axes[0, 1].set_ylabel('Avg Achievements per Episode', fontsize=12, fontweight='bold')
        axes[0, 1].set_title('Achievement Count Comparison', fontsize=13, fontweight='bold')
        axes[0, 1].set_xticks(x)
        axes[0, 1].set_xticklabels(self.model_names, rotation=15, ha='right')
        axes[0, 1].grid(axis='y', alpha=0.3, linestyle='--')

        for i, (bar, val) in enumerate(zip(bars, avg_achievements)):
            axes[0, 1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                            f'{val:.2f}', ha='center', va='bottom', fontweight='bold', fontsize=9)

        # 3. Survival Time
        avg_survival = [model['metrics']['average_survival_time'] for model in self.models]
        std_survival = [model['metrics']['std_survival_time'] for model in self.models]

        bars = axes[1, 0].bar(x, avg_survival, width, yerr=std_survival, capsize=8,
                              color=self.colors, edgecolor='black', linewidth=1.5)
        axes[1, 0].set_ylabel('Average Survival Time (steps)', fontsize=12, fontweight='bold')
        axes[1, 0].set_title('Survival Time Comparison', fontsize=13, fontweight='bold')
        axes[1, 0].set_xticks(x)
        axes[1, 0].set_xticklabels(self.model_names, rotation=15, ha='right')
        axes[1, 0].grid(axis='y', alpha=0.3, linestyle='--')

        for i, (bar, val, std) in enumerate(zip(bars, avg_survival, std_survival)):
            axes[1, 0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + std + 5,
                            f'{val:.0f}', ha='center', va='bottom', fontweight='bold', fontsize=9)

        # 4. Geometric Mean
        geom_means = [model['metrics']['geometric_mean_achievements'] for model in self.models]

        bars = axes[1, 1].bar(x, geom_means, width, color=self.colors,
                              edgecolor='black', linewidth=1.5)
        axes[1, 1].set_ylabel('Geometric Mean', fontsize=12, fontweight='bold')
        axes[1, 1].set_title('Overall Achievement Score', fontsize=13, fontweight='bold')
        axes[1, 1].set_xticks(x)
        axes[1, 1].set_xticklabels(self.model_names, rotation=15, ha='right')
        axes[1, 1].grid(axis='y', alpha=0.3, linestyle='--')

        for i, (bar, val) in enumerate(zip(bars, geom_means)):
            axes[1, 1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.0005,
                            f'{val:.4f}', ha='center', va='bottom', fontweight='bold', fontsize=9)

        plt.tight_layout()
        save_path = self.output_dir / 'performance_metrics_comparison.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: {save_path.name}")
        plt.close()

    def plot_achievement_comparison(self):
        """Create detailed achievement unlock rate comparison."""
        # Get all unique achievements
        all_achievements = set()
        for model in self.models:
            all_achievements.update(model['achievement_unlock_rates'].keys())
        all_achievements = sorted(all_achievements)

        # Calculate average improvement vs baseline
        baseline_rates = self.models[0]['achievement_unlock_rates']
        avg_improvements = {}
        for ach in all_achievements:
            improvements = [model['achievement_unlock_rates'][ach] - baseline_rates[ach]
                           for model in self.models[1:]]
            avg_improvements[ach] = np.mean(improvements)

        sorted_achievements = sorted(all_achievements, key=lambda x: avg_improvements[x], reverse=True)

        fig, axes = plt.subplots(1, 2, figsize=(20, max(8, len(all_achievements) * 0.35)))

        # 1. Side-by-side comparison
        y_pos = np.arange(len(sorted_achievements))
        bar_width = 0.8 / len(self.models)

        for i, (model, name, color) in enumerate(zip(self.models, self.model_names, self.colors)):
            rates = [model['achievement_unlock_rates'][ach] * 100 for ach in sorted_achievements]
            offset = (i - len(self.models)/2 + 0.5) * bar_width
            axes[0].barh(y_pos + offset, rates, bar_width, label=name,
                        color=color, edgecolor='black', alpha=0.8)

        axes[0].set_yticks(y_pos)
        axes[0].set_yticklabels(sorted_achievements, fontsize=9)
        axes[0].set_xlabel('Unlock Rate (%)', fontsize=12, fontweight='bold')
        axes[0].set_title('Achievement Unlock Rates Comparison', fontsize=13, fontweight='bold')
        axes[0].legend(loc='lower right', fontsize=10)
        axes[0].grid(axis='x', alpha=0.3, linestyle='--')

        # 2. Improvement heatmap (vs baseline)
        improvement_matrix = []
        for ach in sorted_achievements:
            improvements = [model['achievement_unlock_rates'][ach] - baseline_rates[ach]
                           for model in self.models[1:]]
            improvement_matrix.append([x * 100 for x in improvements])

        improvement_matrix = np.array(improvement_matrix)

        im = axes[1].imshow(improvement_matrix, cmap='RdYlGn', aspect='auto',
                           vmin=-20, vmax=20)
        axes[1].set_yticks(y_pos)
        axes[1].set_yticklabels(sorted_achievements, fontsize=9)
        axes[1].set_xticks(range(len(self.models) - 1))
        axes[1].set_xticklabels(self.model_names[1:], rotation=15, ha='right')
        axes[1].set_title('Improvement vs Baseline (percentage points)',
                         fontsize=13, fontweight='bold')

        # Add colorbar
        cbar = plt.colorbar(im, ax=axes[1])
        cbar.set_label('Improvement (%)', fontsize=10, fontweight='bold')

        # Add text annotations for significant changes
        for i in range(len(sorted_achievements)):
            for j in range(len(self.models) - 1):
                val = improvement_matrix[i, j]
                if abs(val) > 5:  # Only show if > 5 percentage points
                    color = 'white' if abs(val) > 15 else 'black'
                    axes[1].text(j, i, f'{val:+.1f}', ha='center', va='center',
                               color=color, fontsize=7, fontweight='bold')

        plt.tight_layout()
        save_path = self.output_dir / 'achievement_comparison.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: {save_path.name}")
        plt.close()

    def plot_distributions(self):
        """Plot distribution comparisons."""
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))

        all_rewards = [np.array(model['episode_rewards']) for model in self.models]
        all_achievements = [np.array(model['achievements_per_episode']) for model in self.models]
        all_survival = [np.array(model['episode_lengths']) for model in self.models]

        # Row 1: Histograms
        for rewards, name, color in zip(all_rewards, self.model_names, self.colors):
            axes[0, 0].hist(rewards, bins=30, alpha=0.5, label=name,
                           color=color, edgecolor='black', linewidth=0.5)
            axes[0, 0].axvline(np.mean(rewards), color=color, linestyle='--', linewidth=2)

        axes[0, 0].set_xlabel('Cumulative Reward', fontweight='bold')
        axes[0, 0].set_ylabel('Frequency', fontweight='bold')
        axes[0, 0].set_title('Reward Distribution', fontsize=12, fontweight='bold')
        axes[0, 0].legend(fontsize=9)
        axes[0, 0].grid(alpha=0.3)

        # Achievements
        max_ach = max([ach.max() for ach in all_achievements])
        bins = np.arange(0, max_ach + 2) - 0.5

        for achievements, name, color in zip(all_achievements, self.model_names, self.colors):
            axes[0, 1].hist(achievements, bins=bins, alpha=0.5, label=name,
                           color=color, edgecolor='black', linewidth=0.5)
            axes[0, 1].axvline(np.mean(achievements), color=color, linestyle='--', linewidth=2)

        axes[0, 1].set_xlabel('Achievements per Episode', fontweight='bold')
        axes[0, 1].set_ylabel('Frequency', fontweight='bold')
        axes[0, 1].set_title('Achievement Distribution', fontsize=12, fontweight='bold')
        axes[0, 1].legend(fontsize=9)
        axes[0, 1].grid(alpha=0.3)

        # Survival
        for survival, name, color in zip(all_survival, self.model_names, self.colors):
            axes[0, 2].hist(survival, bins=30, alpha=0.5, label=name,
                           color=color, edgecolor='black', linewidth=0.5)
            axes[0, 2].axvline(np.mean(survival), color=color, linestyle='--', linewidth=2)

        axes[0, 2].set_xlabel('Survival Time (steps)', fontweight='bold')
        axes[0, 2].set_ylabel('Frequency', fontweight='bold')
        axes[0, 2].set_title('Survival Time Distribution', fontsize=12, fontweight='bold')
        axes[0, 2].legend(fontsize=9)
        axes[0, 2].grid(alpha=0.3)

        # Row 2: Box plots
        box_data_rewards = all_rewards
        bp = axes[1, 0].boxplot(box_data_rewards, labels=self.model_names,
                                patch_artist=True, showmeans=True)
        for patch, color in zip(bp['boxes'], self.colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        axes[1, 0].set_ylabel('Cumulative Reward', fontweight='bold')
        axes[1, 0].set_title('Reward Box Plot', fontsize=12, fontweight='bold')
        axes[1, 0].grid(alpha=0.3, axis='y')
        axes[1, 0].tick_params(axis='x', rotation=15)

        box_data_ach = all_achievements
        bp = axes[1, 1].boxplot(box_data_ach, labels=self.model_names,
                                patch_artist=True, showmeans=True)
        for patch, color in zip(bp['boxes'], self.colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        axes[1, 1].set_ylabel('Achievements per Episode', fontweight='bold')
        axes[1, 1].set_title('Achievement Box Plot', fontsize=12, fontweight='bold')
        axes[1, 1].grid(alpha=0.3, axis='y')
        axes[1, 1].tick_params(axis='x', rotation=15)

        box_data_surv = all_survival
        bp = axes[1, 2].boxplot(box_data_surv, labels=self.model_names,
                                patch_artist=True, showmeans=True)
        for patch, color in zip(bp['boxes'], self.colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        axes[1, 2].set_ylabel('Survival Time (steps)', fontweight='bold')
        axes[1, 2].set_title('Survival Time Box Plot', fontsize=12, fontweight='bold')
        axes[1, 2].grid(alpha=0.3, axis='y')
        axes[1, 2].tick_params(axis='x', rotation=15)

        plt.tight_layout()
        save_path = self.output_dir / 'distributions_comparison.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: {save_path.name}")
        plt.close()

    def plot_episode_traces(self):
        """Plot episode-by-episode performance traces."""
        fig, axes = plt.subplots(3, 1, figsize=(15, 12))

        all_rewards = [model['episode_rewards'] for model in self.models]
        all_achievements = [model['achievements_per_episode'] for model in self.models]
        all_survival = [model['episode_lengths'] for model in self.models]

        # Determine common episode range
        min_episodes = min(len(rewards) for rewards in all_rewards)
        episodes = range(1, min_episodes + 1)

        # Calculate moving average window
        window = min(50, min_episodes // 10)

        # Rewards trace
        for rewards, name, color in zip(all_rewards, self.model_names, self.colors):
            rewards_trimmed = rewards[:min_episodes]
            axes[0].plot(episodes, rewards_trimmed, alpha=0.3, linewidth=1, color=color)

            if window > 1:
                ma = np.convolve(rewards_trimmed, np.ones(window) / window, mode='valid')
                ma_episodes = range(window, min_episodes + 1)
                axes[0].plot(ma_episodes, ma, linewidth=2.5, color=color, label=name)

        axes[0].set_xlabel('Episode', fontweight='bold')
        axes[0].set_ylabel('Cumulative Reward', fontweight='bold')
        axes[0].set_title(f'Episode Rewards Over Time (MA window={window})',
                         fontsize=13, fontweight='bold')
        axes[0].legend(loc='best', fontsize=10)
        axes[0].grid(alpha=0.3)

        # Achievements trace
        for achievements, name, color in zip(all_achievements, self.model_names, self.colors):
            ach_trimmed = achievements[:min_episodes]
            axes[1].plot(episodes, ach_trimmed, alpha=0.3, linewidth=1, color=color)

            if window > 1:
                ma = np.convolve(ach_trimmed, np.ones(window) / window, mode='valid')
                ma_episodes = range(window, min_episodes + 1)
                axes[1].plot(ma_episodes, ma, linewidth=2.5, color=color, label=name)

        axes[1].set_xlabel('Episode', fontweight='bold')
        axes[1].set_ylabel('Achievements Unlocked', fontweight='bold')
        axes[1].set_title(f'Achievements Over Time (MA window={window})',
                         fontsize=13, fontweight='bold')
        axes[1].legend(loc='best', fontsize=10)
        axes[1].grid(alpha=0.3)

        # Survival trace
        for survival, name, color in zip(all_survival, self.model_names, self.colors):
            surv_trimmed = survival[:min_episodes]
            axes[2].plot(episodes, surv_trimmed, alpha=0.3, linewidth=1, color=color)

            if window > 1:
                ma = np.convolve(surv_trimmed, np.ones(window) / window, mode='valid')
                ma_episodes = range(window, min_episodes + 1)
                axes[2].plot(ma_episodes, ma, linewidth=2.5, color=color, label=name)

        axes[2].set_xlabel('Episode', fontweight='bold')
        axes[2].set_ylabel('Survival Time (steps)', fontweight='bold')
        axes[2].set_title(f'Survival Time Over Time (MA window={window})',
                         fontsize=13, fontweight='bold')
        axes[2].legend(loc='best', fontsize=10)
        axes[2].grid(alpha=0.3)

        plt.tight_layout()
        save_path = self.output_dir / 'episode_traces.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: {save_path.name}")
        plt.close()

    def plot_action_comparison(self):
        """Compare action distributions between models."""
        all_action_dists = [model['action_distribution'] for model in self.models]

        # Get all unique actions
        all_actions = set()
        for action_dist in all_action_dists:
            all_actions.update(action_dist.keys())
        all_actions = sorted(all_actions)

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        x = np.arange(len(all_actions))
        bar_width = 0.8 / len(self.models)

        # Calculate percentages
        all_pcts = []
        for action_dist in all_action_dists:
            counts = [action_dist.get(str(a), 0) for a in all_actions]
            total = sum(counts)
            pcts = [c / total * 100 if total > 0 else 0 for c in counts]
            all_pcts.append(pcts)

        # Side-by-side bar chart
        for i, (pcts, name, color) in enumerate(zip(all_pcts, self.model_names, self.colors)):
            offset = (i - len(self.models)/2 + 0.5) * bar_width
            axes[0].bar(x + offset, pcts, bar_width, label=name,
                       color=color, edgecolor='black', alpha=0.8)

        axes[0].set_xlabel('Action ID', fontsize=12, fontweight='bold')
        axes[0].set_ylabel('Percentage (%)', fontsize=12, fontweight='bold')
        axes[0].set_title('Action Distribution Comparison', fontsize=13, fontweight='bold')
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(all_actions)
        axes[0].legend(fontsize=10)
        axes[0].grid(axis='y', alpha=0.3)

        # Heatmap of differences vs baseline
        if len(self.models) > 1:
            diff_matrix = []
            for i in range(1, len(self.models)):
                differences = [all_pcts[i][j] - all_pcts[0][j] for j in range(len(all_actions))]
                diff_matrix.append(differences)

            diff_matrix = np.array(diff_matrix)

            im = axes[1].imshow(diff_matrix, cmap='RdBu_r', aspect='auto',
                               vmin=-10, vmax=10)
            axes[1].set_yticks(range(len(self.models) - 1))
            axes[1].set_yticklabels(self.model_names[1:])
            axes[1].set_xticks(range(len(all_actions)))
            axes[1].set_xticklabels(all_actions)
            axes[1].set_xlabel('Action ID', fontsize=12, fontweight='bold')
            axes[1].set_title('Action Distribution Change vs Baseline (pp)',
                             fontsize=13, fontweight='bold')

            cbar = plt.colorbar(im, ax=axes[1])
            cbar.set_label('Difference (%)', fontsize=10, fontweight='bold')

            # Add text annotations
            for i in range(len(self.models) - 1):
                for j in range(len(all_actions)):
                    val = diff_matrix[i, j]
                    if abs(val) > 2:
                        color = 'white' if abs(val) > 7 else 'black'
                        axes[1].text(j, i, f'{val:+.1f}', ha='center', va='center',
                                   color=color, fontsize=8, fontweight='bold')

        plt.tight_layout()
        save_path = self.output_dir / 'action_distribution_comparison.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: {save_path.name}")
        plt.close()

    def plot_progressive_improvement(self):
        """Visualize progressive improvements across model iterations."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        metrics = [
            ('average_reward', 'Average Reward', 'Cumulative Reward'),
            ('average_achievements_per_episode', 'Average Achievements', 'Achievements/Episode'),
            ('average_survival_time', 'Average Survival Time', 'Steps'),
            ('geometric_mean_achievements', 'Geometric Mean', 'Score')
        ]

        for ax, (metric, title, ylabel) in zip(axes.flat, metrics):
            values = [model['metrics'][metric] for model in self.models]
            baseline_val = values[0]

            # Plot line with markers
            ax.plot(range(len(values)), values, marker='o', markersize=10,
                   linewidth=2.5, color='#2c3e50')

            # Fill area under curve
            ax.fill_between(range(len(values)), baseline_val, values,
                           where=[v >= baseline_val for v in values],
                           alpha=0.3, color='#27ae60', label='Improvement')
            ax.fill_between(range(len(values)), baseline_val, values,
                           where=[v < baseline_val for v in values],
                           alpha=0.3, color='#c0392b', label='Degradation')

            # Add value labels
            for i, val in enumerate(values):
                improvement = ((val - baseline_val) / abs(baseline_val) * 100) if i > 0 else 0
                label = f'{val:.2f}' if metric == 'geometric_mean_achievements' else f'{val:.1f}'
                if i > 0:
                    label += f'\n({improvement:+.1f}%)'
                ax.text(i, val, label, ha='center', va='bottom' if val >= baseline_val else 'top',
                       fontweight='bold', fontsize=9)

            # Baseline reference line
            ax.axhline(y=baseline_val, color='#e74c3c', linestyle='--',
                      linewidth=2, alpha=0.7, label='Baseline')

            ax.set_xticks(range(len(self.model_names)))
            ax.set_xticklabels(self.model_names, rotation=15, ha='right')
            ax.set_ylabel(ylabel, fontweight='bold')
            ax.set_title(title, fontsize=12, fontweight='bold')
            ax.grid(alpha=0.3)
            if ax == axes[0, 0]:
                ax.legend(loc='best', fontsize=9)

        plt.tight_layout()
        save_path = self.output_dir / 'progressive_improvement.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: {save_path.name}")
        plt.close()

    def generate_report(self):
        """Generate a comprehensive text report."""
        report_path = self.output_dir / 'comparison_report.txt'

        with open(report_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("MULTI-MODEL COMPARISON REPORT\n")
            f.write("=" * 80 + "\n\n")

            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            f.write("MODELS COMPARED:\n")
            for i, (name, path, model) in enumerate(zip(self.model_names, self.model_paths, self.models)):
                f.write(f"  {i+1}. {name}:\n")
                f.write(f"     File: {path.name}\n")
                f.write(f"     Episodes: {model['num_episodes']}\n")
            f.write("\n")

            f.write("=" * 80 + "\n")
            f.write("PERFORMANCE SUMMARY\n")
            f.write("=" * 80 + "\n\n")

            # Rewards
            f.write("CUMULATIVE REWARD:\n")
            f.write(f"  {'Model':<20} {'Mean':>12} {'Std':>12} {'vs Baseline':>15}\n")
            f.write("  " + "-" * 62 + "\n")

            baseline_reward = self.models[0]['metrics']['average_reward']
            for name, model in zip(self.model_names, self.models):
                mean_val = model['metrics']['average_reward']
                std_val = model['metrics']['std_reward']
                if name == self.model_names[0]:
                    improvement = "-"
                else:
                    improvement = f"{((mean_val - baseline_reward) / abs(baseline_reward) * 100):+.2f}%"
                f.write(f"  {name:<20} {mean_val:>12.2f} {std_val:>12.2f} {improvement:>15}\n")
            f.write("\n")

            # Achievements
            f.write("ACHIEVEMENTS PER EPISODE:\n")
            f.write(f"  {'Model':<20} {'Mean':>12} {'vs Baseline':>15}\n")
            f.write("  " + "-" * 50 + "\n")

            baseline_ach = self.models[0]['metrics']['average_achievements_per_episode']
            for name, model in zip(self.model_names, self.models):
                mean_val = model['metrics']['average_achievements_per_episode']
                if name == self.model_names[0]:
                    improvement = "-"
                else:
                    improvement = f"{((mean_val - baseline_ach) / baseline_ach * 100):+.2f}%"
                f.write(f"  {name:<20} {mean_val:>12.2f} {improvement:>15}\n")
            f.write("\n")

            # Survival
            f.write("SURVIVAL TIME (steps):\n")
            f.write(f"  {'Model':<20} {'Mean':>12} {'Std':>12} {'vs Baseline':>15}\n")
            f.write("  " + "-" * 62 + "\n")

            baseline_surv = self.models[0]['metrics']['average_survival_time']
            for name, model in zip(self.model_names, self.models):
                mean_val = model['metrics']['average_survival_time']
                std_val = model['metrics']['std_survival_time']
                if name == self.model_names[0]:
                    improvement = "-"
                else:
                    improvement = f"{((mean_val - baseline_surv) / baseline_surv * 100):+.2f}%"
                f.write(f"  {name:<20} {mean_val:>12.2f} {std_val:>12.2f} {improvement:>15}\n")
            f.write("\n")

            # Geometric mean
            f.write("GEOMETRIC MEAN OF ACHIEVEMENTS:\n")
            f.write(f"  {'Model':<20} {'Score':>12} {'vs Baseline':>15}\n")
            f.write("  " + "-" * 50 + "\n")

            baseline_geom = self.models[0]['metrics']['geometric_mean_achievements']
            for name, model in zip(self.model_names, self.models):
                mean_val = model['metrics']['geometric_mean_achievements']
                if name == self.model_names[0]:
                    improvement = "-"
                else:
                    improvement = f"{((mean_val - baseline_geom) / baseline_geom * 100):+.2f}%"
                f.write(f"  {name:<20} {mean_val:>12.6f} {improvement:>15}\n")
            f.write("\n")

            # Achievement details
            f.write("=" * 80 + "\n")
            f.write("ACHIEVEMENT ANALYSIS\n")
            f.write("=" * 80 + "\n\n")

            # Get all achievements
            all_achievements = sorted(self.models[0]['achievement_unlock_rates'].keys())
            baseline_rates = self.models[0]['achievement_unlock_rates']

            # Find most improved achievements (comparing final model to baseline)
            if len(self.models) > 1:
                final_rates = self.models[-1]['achievement_unlock_rates']
                improvements = {ach: (final_rates[ach] - baseline_rates[ach]) * 100
                               for ach in all_achievements}

                sorted_improvements = sorted(improvements.items(), key=lambda x: x[1], reverse=True)

                f.write(f"Top 10 Most Improved Achievements ({self.model_names[-1]} vs {self.model_names[0]}):\n")
                f.write(f"  {'#':<4} {'Achievement':<28} {'Baseline':>10} {'Final':>10} {'Change':>10}\n")
                f.write("  " + "-" * 68 + "\n")

                for i, (ach, improvement) in enumerate(sorted_improvements[:10], 1):
                    base_rate = baseline_rates[ach] * 100
                    final_rate = final_rates[ach] * 100
                    f.write(f"  {i:<4} {ach:<28} {base_rate:>9.1f}% {final_rate:>9.1f}% "
                           f"{improvement:>9.1f}pp\n")
                f.write("\n")

                # Achievements that got worse
                worse = [ach for ach, imp in improvements.items() if imp < -1.0]
                if worse:
                    f.write(f"Achievements with Decreased Rates (> 1 pp): {len(worse)}\n")
                    for ach in sorted(worse, key=lambda x: improvements[x]):
                        improvement = improvements[ach]
                        base_rate = baseline_rates[ach] * 100
                        final_rate = final_rates[ach] * 100
                        f.write(f"  - {ach:<28} {base_rate:>9.1f}% → {final_rate:>9.1f}% "
                               f"({improvement:>+9.1f}pp)\n")
                    f.write("\n")

            # Progressive improvement table
            f.write("=" * 80 + "\n")
            f.write("PROGRESSIVE IMPROVEMENT TABLE\n")
            f.write("=" * 80 + "\n\n")

            f.write("Achievement unlock rates across all models:\n\n")

            # Header
            header = f"  {'Achievement':<28}"
            for name in self.model_names:
                header += f" {name[:10]:>10}"
            f.write(header + "\n")
            f.write("  " + "-" * (28 + 11 * len(self.model_names)) + "\n")

            # Each achievement
            for ach in all_achievements:
                row = f"  {ach:<28}"
                for model in self.models:
                    rate = model['achievement_unlock_rates'][ach] * 100
                    row += f" {rate:>9.1f}%"
                f.write(row + "\n")

            f.write("\n")
            f.write("=" * 80 + "\n")
            f.write("END OF REPORT\n")
            f.write("=" * 80 + "\n")

        print(f"  ✓ Saved: {report_path.name}")


def main():
    parser = argparse.ArgumentParser(
        description='Compare multiple DQN model variants'
    )
    parser.add_argument(
        '--models',
        type=str,
        nargs='+',
        required=True,
        help='Paths to model results JSON files (in order: baseline, frame_stack, improved, etc.)'
    )
    parser.add_argument(
        '--names',
        type=str,
        nargs='+',
        default=None,
        help='Names for each model (optional, same order as --models)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./comparison_results/',
        help='Directory to save comparison outputs'
    )

    args = parser.parse_args()

    # Validate inputs
    if args.names is not None and len(args.names) != len(args.models):
        print("ERROR: Number of names must match number of models")
        return

    # Use default names if not provided
    if args.names is None:
        if len(args.models) == 2:
            args.names = ['Baseline', 'Improved']
        elif len(args.models) == 3:
            args.names = ['Baseline', 'Frame Stacking', 'Improved']
        else:
            args.names = [f'Model {i+1}' for i in range(len(args.models))]

    print("\n" + "=" * 70)
    print("MULTI-MODEL COMPARISON")
    print("=" * 70)
    for i, (name, path) in enumerate(zip(args.names, args.models), 1):
        print(f"{i}. {name:<20} {path}")
    print(f"Output: {args.output_dir}")
    print("=" * 70 + "\n")

    # Create comparison instance
    comparison = ModelComparison(
        model_paths=args.models,
        model_names=args.names,
        output_dir=args.output_dir
    )

    # Generate all comparisons
    comparison.generate_full_comparison()

    print("\n✓ Comparison complete!\n")


if __name__ == "__main__":
    main()
"""
Model Comparison Script for DQN Baseline vs Improved Models
Loads results from JSON files and creates comprehensive comparisons.
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
    """Compare DQN baseline and improved models using saved results."""

    def __init__(self, baseline_path, improved_path, output_dir='./comparison_results/'):
        """
        Initialize comparison with paths to result files.

        Args:
            baseline_path: Path to baseline model results JSON
            improved_path: Path to improved model results JSON
            output_dir: Directory to save comparison outputs
        """
        self.baseline_path = Path(baseline_path)
        self.improved_path = Path(improved_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Load results
        self.baseline = self._load_results(self.baseline_path)
        self.improved = self._load_results(self.improved_path)

        print(f"Loaded baseline results: {self.baseline['num_episodes']} episodes")
        print(f"Loaded improved results: {self.improved['num_episodes']} episodes")

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

        # 7. Generate text report
        print("7. Generating comparison report...")
        self.generate_report()

        print("\n" + "=" * 70)
        print(f"All comparison results saved to: {self.output_dir}")
        print("=" * 70)

    def statistical_comparison(self):
        """Perform statistical tests and print results."""
        baseline_rewards = np.array(self.baseline['episode_rewards'])
        improved_rewards = np.array(self.improved['episode_rewards'])

        baseline_achievements = np.array(self.baseline['achievements_per_episode'])
        improved_achievements = np.array(self.improved['achievements_per_episode'])

        baseline_survival = np.array(self.baseline['episode_lengths'])
        improved_survival = np.array(self.improved['episode_lengths'])

        print("\n" + "=" * 70)
        print("STATISTICAL ANALYSIS")
        print("=" * 70)

        # Reward comparison
        print("\n📊 REWARD COMPARISON")
        print("-" * 50)
        t_stat, p_value = stats.ttest_ind(baseline_rewards, improved_rewards)
        print(f"  Baseline Mean:  {np.mean(baseline_rewards):8.2f} ± {np.std(baseline_rewards):.2f}")
        print(f"  Improved Mean:  {np.mean(improved_rewards):8.2f} ± {np.std(improved_rewards):.2f}")
        print(f"  Difference:     {np.mean(improved_rewards) - np.mean(baseline_rewards):8.2f}")
        print(
            f"  Improvement:    {((np.mean(improved_rewards) - np.mean(baseline_rewards)) / abs(np.mean(baseline_rewards)) * 100):7.2f}%")
        print(f"  T-statistic:    {t_stat:8.4f}")
        print(f"  P-value:        {p_value:8.6f}")
        if p_value < 0.05:
            print(f"  ✓ Statistically significant difference (p < 0.05)")
        else:
            print(f"  ✗ No statistically significant difference (p >= 0.05)")

        # Achievement comparison
        print("\n🏆 ACHIEVEMENT COMPARISON")
        print("-" * 50)
        t_stat, p_value = stats.ttest_ind(baseline_achievements, improved_achievements)
        print(f"  Baseline Mean:  {np.mean(baseline_achievements):8.2f} ± {np.std(baseline_achievements):.2f}")
        print(f"  Improved Mean:  {np.mean(improved_achievements):8.2f} ± {np.std(improved_achievements):.2f}")
        print(f"  Difference:     {np.mean(improved_achievements) - np.mean(baseline_achievements):8.2f}")
        print(
            f"  Improvement:    {((np.mean(improved_achievements) - np.mean(baseline_achievements)) / np.mean(baseline_achievements) * 100):7.2f}%")
        print(f"  T-statistic:    {t_stat:8.4f}")
        print(f"  P-value:        {p_value:8.6f}")
        if p_value < 0.05:
            print(f"  ✓ Statistically significant difference (p < 0.05)")
        else:
            print(f"  ✗ No statistically significant difference (p >= 0.05)")

        # Survival time comparison
        print("\n⏱️  SURVIVAL TIME COMPARISON")
        print("-" * 50)
        t_stat, p_value = stats.ttest_ind(baseline_survival, improved_survival)
        print(f"  Baseline Mean:  {np.mean(baseline_survival):8.2f} ± {np.std(baseline_survival):.2f} steps")
        print(f"  Improved Mean:  {np.mean(improved_survival):8.2f} ± {np.std(improved_survival):.2f} steps")
        print(f"  Difference:     {np.mean(improved_survival) - np.mean(baseline_survival):8.2f} steps")
        print(
            f"  Improvement:    {((np.mean(improved_survival) - np.mean(baseline_survival)) / np.mean(baseline_survival) * 100):7.2f}%")
        print(f"  T-statistic:    {t_stat:8.4f}")
        print(f"  P-value:        {p_value:8.6f}")
        if p_value < 0.05:
            print(f"  ✓ Statistically significant difference (p < 0.05)")
        else:
            print(f"  ✗ No statistically significant difference (p >= 0.05)")

        # Effect sizes (Cohen's d)
        print("\n📏 EFFECT SIZES (Cohen's d)")
        print("-" * 50)
        reward_effect = (np.mean(improved_rewards) - np.mean(baseline_rewards)) / np.sqrt(
            (np.std(baseline_rewards) ** 2 + np.std(improved_rewards) ** 2) / 2
        )
        achievement_effect = (np.mean(improved_achievements) - np.mean(baseline_achievements)) / np.sqrt(
            (np.std(baseline_achievements) ** 2 + np.std(improved_achievements) ** 2) / 2
        )
        survival_effect = (np.mean(improved_survival) - np.mean(baseline_survival)) / np.sqrt(
            (np.std(baseline_survival) ** 2 + np.std(improved_survival) ** 2) / 2
        )

        print(f"  Reward Effect Size:       {reward_effect:7.4f} ({self._interpret_effect_size(reward_effect)})")
        print(
            f"  Achievement Effect Size:  {achievement_effect:7.4f} ({self._interpret_effect_size(achievement_effect)})")
        print(f"  Survival Effect Size:     {survival_effect:7.4f} ({self._interpret_effect_size(survival_effect)})")

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
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        models = ['Baseline', 'Improved']
        colors = ['#3498db', '#e74c3c']

        # 1. Average Rewards
        avg_rewards = [
            self.baseline['metrics']['average_reward'],
            self.improved['metrics']['average_reward']
        ]
        std_rewards = [
            self.baseline['metrics']['std_reward'],
            self.improved['metrics']['std_reward']
        ]
        bars = axes[0, 0].bar(models, avg_rewards, yerr=std_rewards, capsize=10,
                              color=colors, edgecolor='black', linewidth=1.5)
        axes[0, 0].set_ylabel('Average Cumulative Reward', fontsize=12, fontweight='bold')
        axes[0, 0].set_title('Reward Comparison (Standard Environment)', fontsize=13, fontweight='bold')
        axes[0, 0].grid(axis='y', alpha=0.3, linestyle='--')
        # Add value labels
        for i, (bar, val, std) in enumerate(zip(bars, avg_rewards, std_rewards)):
            axes[0, 0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + std + 0.5,
                            f'{val:.1f}', ha='center', va='bottom', fontweight='bold')

        # 2. Average Achievements
        avg_achievements = [
            self.baseline['metrics']['average_achievements_per_episode'],
            self.improved['metrics']['average_achievements_per_episode']
        ]
        bars = axes[0, 1].bar(models, avg_achievements, color=colors,
                              edgecolor='black', linewidth=1.5)
        axes[0, 1].set_ylabel('Avg Achievements per Episode', fontsize=12, fontweight='bold')
        axes[0, 1].set_title('Achievement Count Comparison', fontsize=13, fontweight='bold')
        axes[0, 1].grid(axis='y', alpha=0.3, linestyle='--')
        for i, (bar, val) in enumerate(zip(bars, avg_achievements)):
            axes[0, 1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                            f'{val:.2f}', ha='center', va='bottom', fontweight='bold')

        # 3. Survival Time
        avg_survival = [
            self.baseline['metrics']['average_survival_time'],
            self.improved['metrics']['average_survival_time']
        ]
        std_survival = [
            self.baseline['metrics']['std_survival_time'],
            self.improved['metrics']['std_survival_time']
        ]
        bars = axes[1, 0].bar(models, avg_survival, yerr=std_survival, capsize=10,
                              color=colors, edgecolor='black', linewidth=1.5)
        axes[1, 0].set_ylabel('Average Survival Time (steps)', fontsize=12, fontweight='bold')
        axes[1, 0].set_title('Survival Time Comparison', fontsize=13, fontweight='bold')
        axes[1, 0].grid(axis='y', alpha=0.3, linestyle='--')
        for i, (bar, val, std) in enumerate(zip(bars, avg_survival, std_survival)):
            axes[1, 0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + std + 5,
                            f'{val:.0f}', ha='center', va='bottom', fontweight='bold')

        # 4. Geometric Mean
        geom_means = [
            self.baseline['metrics']['geometric_mean_achievements'],
            self.improved['metrics']['geometric_mean_achievements']
        ]
        bars = axes[1, 1].bar(models, geom_means, color=colors,
                              edgecolor='black', linewidth=1.5)
        axes[1, 1].set_ylabel('Geometric Mean of Achievement Rates', fontsize=12, fontweight='bold')
        axes[1, 1].set_title('Overall Achievement Score', fontsize=13, fontweight='bold')
        axes[1, 1].grid(axis='y', alpha=0.3, linestyle='--')
        for i, (bar, val) in enumerate(zip(bars, geom_means)):
            axes[1, 1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001,
                            f'{val:.4f}', ha='center', va='bottom', fontweight='bold')

        plt.tight_layout()
        save_path = self.output_dir / 'performance_metrics_comparison.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: {save_path.name}")
        plt.close()

    def plot_achievement_comparison(self):
        """Create detailed achievement unlock rate comparison."""
        baseline_rates = self.baseline['achievement_unlock_rates']
        improved_rates = self.improved['achievement_unlock_rates']

        # Get all achievements and sort by improvement
        all_achievements = sorted(baseline_rates.keys())
        improvements = {ach: improved_rates[ach] - baseline_rates[ach]
                        for ach in all_achievements}
        sorted_achievements = sorted(all_achievements,
                                     key=lambda x: improvements[x], reverse=True)

        fig, axes = plt.subplots(1, 2, figsize=(18, max(8, len(all_achievements) * 0.35)))

        # 1. Side-by-side comparison
        y_pos = np.arange(len(sorted_achievements))
        width = 0.35

        baseline_vals = [baseline_rates[ach] * 100 for ach in sorted_achievements]
        improved_vals = [improved_rates[ach] * 100 for ach in sorted_achievements]

        bars1 = axes[0].barh(y_pos - width / 2, baseline_vals, width,
                             label='Baseline', color='#3498db', edgecolor='black')
        bars2 = axes[0].barh(y_pos + width / 2, improved_vals, width,
                             label='Improved', color='#e74c3c', edgecolor='black')

        axes[0].set_yticks(y_pos)
        axes[0].set_yticklabels(sorted_achievements)
        axes[0].set_xlabel('Unlock Rate (%)', fontsize=12, fontweight='bold')
        axes[0].set_title('Achievement Unlock Rates Comparison', fontsize=13, fontweight='bold')
        axes[0].legend(loc='lower right', fontsize=11)
        axes[0].grid(axis='x', alpha=0.3, linestyle='--')

        # 2. Improvement delta
        deltas = [improvements[ach] * 100 for ach in sorted_achievements]
        colors_delta = ['#27ae60' if d >= 0 else '#c0392b' for d in deltas]

        bars = axes[1].barh(y_pos, deltas, color=colors_delta, edgecolor='black')
        axes[1].set_yticks(y_pos)
        axes[1].set_yticklabels(sorted_achievements)
        axes[1].set_xlabel('Improvement (percentage points)', fontsize=12, fontweight='bold')
        axes[1].set_title('Achievement Unlock Rate Improvement', fontsize=13, fontweight='bold')
        axes[1].axvline(x=0, color='black', linestyle='-', linewidth=1)
        axes[1].grid(axis='x', alpha=0.3, linestyle='--')

        # Add value labels for significant improvements
        for i, (bar, delta) in enumerate(zip(bars, deltas)):
            if abs(delta) >= 5:  # Only label if improvement is >= 5 percentage points
                label_x = delta + (2 if delta > 0 else -2)
                axes[1].text(label_x, bar.get_y() + bar.get_height() / 2,
                             f'{delta:+.1f}', va='center',
                             ha='left' if delta > 0 else 'right',
                             fontweight='bold', fontsize=9)

        plt.tight_layout()
        save_path = self.output_dir / 'achievement_comparison.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: {save_path.name}")
        plt.close()

    def plot_distributions(self):
        """Plot distribution comparisons for rewards, achievements, and survival."""
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))

        baseline_rewards = np.array(self.baseline['episode_rewards'])
        improved_rewards = np.array(self.improved['episode_rewards'])
        baseline_achievements = np.array(self.baseline['achievements_per_episode'])
        improved_achievements = np.array(self.improved['achievements_per_episode'])
        baseline_survival = np.array(self.baseline['episode_lengths'])
        improved_survival = np.array(self.improved['episode_lengths'])

        # Row 1: Histograms
        # Rewards histogram
        axes[0, 0].hist(baseline_rewards, bins=30, alpha=0.6, label='Baseline',
                        color='#3498db', edgecolor='black')
        axes[0, 0].hist(improved_rewards, bins=30, alpha=0.6, label='Improved',
                        color='#e74c3c', edgecolor='black')
        axes[0, 0].axvline(np.mean(baseline_rewards), color='#3498db',
                           linestyle='--', linewidth=2, label='Baseline Mean')
        axes[0, 0].axvline(np.mean(improved_rewards), color='#e74c3c',
                           linestyle='--', linewidth=2, label='Improved Mean')
        axes[0, 0].set_xlabel('Cumulative Reward', fontweight='bold')
        axes[0, 0].set_ylabel('Frequency', fontweight='bold')
        axes[0, 0].set_title('Reward Distribution', fontsize=12, fontweight='bold')
        axes[0, 0].legend()
        axes[0, 0].grid(alpha=0.3)

        # Achievements histogram
        max_ach = max(baseline_achievements.max(), improved_achievements.max())
        bins = np.arange(0, max_ach + 2) - 0.5
        axes[0, 1].hist(baseline_achievements, bins=bins, alpha=0.6, label='Baseline',
                        color='#3498db', edgecolor='black')
        axes[0, 1].hist(improved_achievements, bins=bins, alpha=0.6, label='Improved',
                        color='#e74c3c', edgecolor='black')
        axes[0, 1].axvline(np.mean(baseline_achievements), color='#3498db',
                           linestyle='--', linewidth=2)
        axes[0, 1].axvline(np.mean(improved_achievements), color='#e74c3c',
                           linestyle='--', linewidth=2)
        axes[0, 1].set_xlabel('Achievements per Episode', fontweight='bold')
        axes[0, 1].set_ylabel('Frequency', fontweight='bold')
        axes[0, 1].set_title('Achievement Distribution', fontsize=12, fontweight='bold')
        axes[0, 1].legend()
        axes[0, 1].grid(alpha=0.3)

        # Survival histogram
        axes[0, 2].hist(baseline_survival, bins=30, alpha=0.6, label='Baseline',
                        color='#3498db', edgecolor='black')
        axes[0, 2].hist(improved_survival, bins=30, alpha=0.6, label='Improved',
                        color='#e74c3c', edgecolor='black')
        axes[0, 2].axvline(np.mean(baseline_survival), color='#3498db',
                           linestyle='--', linewidth=2)
        axes[0, 2].axvline(np.mean(improved_survival), color='#e74c3c',
                           linestyle='--', linewidth=2)
        axes[0, 2].set_xlabel('Survival Time (steps)', fontweight='bold')
        axes[0, 2].set_ylabel('Frequency', fontweight='bold')
        axes[0, 2].set_title('Survival Time Distribution', fontsize=12, fontweight='bold')
        axes[0, 2].legend()
        axes[0, 2].grid(alpha=0.3)

        # Row 2: Box plots
        axes[1, 0].boxplot([baseline_rewards, improved_rewards],
                           labels=['Baseline', 'Improved'],
                           patch_artist=True,
                           boxprops=dict(facecolor='lightblue', edgecolor='black'),
                           medianprops=dict(color='red', linewidth=2))
        axes[1, 0].set_ylabel('Cumulative Reward', fontweight='bold')
        axes[1, 0].set_title('Reward Box Plot', fontsize=12, fontweight='bold')
        axes[1, 0].grid(alpha=0.3, axis='y')

        axes[1, 1].boxplot([baseline_achievements, improved_achievements],
                           labels=['Baseline', 'Improved'],
                           patch_artist=True,
                           boxprops=dict(facecolor='lightgreen', edgecolor='black'),
                           medianprops=dict(color='red', linewidth=2))
        axes[1, 1].set_ylabel('Achievements per Episode', fontweight='bold')
        axes[1, 1].set_title('Achievement Box Plot', fontsize=12, fontweight='bold')
        axes[1, 1].grid(alpha=0.3, axis='y')

        axes[1, 2].boxplot([baseline_survival, improved_survival],
                           labels=['Baseline', 'Improved'],
                           patch_artist=True,
                           boxprops=dict(facecolor='lightyellow', edgecolor='black'),
                           medianprops=dict(color='red', linewidth=2))
        axes[1, 2].set_ylabel('Survival Time (steps)', fontweight='bold')
        axes[1, 2].set_title('Survival Time Box Plot', fontsize=12, fontweight='bold')
        axes[1, 2].grid(alpha=0.3, axis='y')

        plt.tight_layout()
        save_path = self.output_dir / 'distributions_comparison.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: {save_path.name}")
        plt.close()

    def plot_episode_traces(self):
        """Plot episode-by-episode performance traces."""
        fig, axes = plt.subplots(3, 1, figsize=(15, 12))

        baseline_rewards = self.baseline['episode_rewards']
        improved_rewards = self.improved['episode_rewards']
        baseline_achievements = self.baseline['achievements_per_episode']
        improved_achievements = self.improved['achievements_per_episode']
        baseline_survival = self.baseline['episode_lengths']
        improved_survival = self.improved['episode_lengths']

        episodes = range(1, len(baseline_rewards) + 1)

        # Rewards trace
        axes[0].plot(episodes, baseline_rewards, alpha=0.5, linewidth=1,
                     color='#3498db', label='Baseline')
        axes[0].plot(episodes, improved_rewards, alpha=0.5, linewidth=1,
                     color='#e74c3c', label='Improved')

        # Add moving averages
        window = min(50, len(baseline_rewards) // 10)
        if window > 1:
            baseline_ma = np.convolve(baseline_rewards, np.ones(window) / window, mode='valid')
            improved_ma = np.convolve(improved_rewards, np.ones(window) / window, mode='valid')
            ma_episodes = range(window, len(baseline_rewards) + 1)
            axes[0].plot(ma_episodes, baseline_ma, linewidth=2.5, color='#2c3e50',
                         label=f'Baseline MA({window})')
            axes[0].plot(ma_episodes, improved_ma, linewidth=2.5, color='#8e44ad',
                         label=f'Improved MA({window})')

        axes[0].set_xlabel('Episode', fontweight='bold')
        axes[0].set_ylabel('Cumulative Reward', fontweight='bold')
        axes[0].set_title('Episode Rewards Over Time', fontsize=13, fontweight='bold')
        axes[0].legend(loc='best')
        axes[0].grid(alpha=0.3)

        # Achievements trace
        axes[1].plot(episodes, baseline_achievements, alpha=0.5, linewidth=1,
                     color='#3498db', label='Baseline')
        axes[1].plot(episodes, improved_achievements, alpha=0.5, linewidth=1,
                     color='#e74c3c', label='Improved')

        if window > 1:
            baseline_ma = np.convolve(baseline_achievements, np.ones(window) / window, mode='valid')
            improved_ma = np.convolve(improved_achievements, np.ones(window) / window, mode='valid')
            axes[1].plot(ma_episodes, baseline_ma, linewidth=2.5, color='#2c3e50',
                         label=f'Baseline MA({window})')
            axes[1].plot(ma_episodes, improved_ma, linewidth=2.5, color='#8e44ad',
                         label=f'Improved MA({window})')

        axes[1].set_xlabel('Episode', fontweight='bold')
        axes[1].set_ylabel('Achievements Unlocked', fontweight='bold')
        axes[1].set_title('Achievements Over Time', fontsize=13, fontweight='bold')
        axes[1].legend(loc='best')
        axes[1].grid(alpha=0.3)

        # Survival trace
        axes[2].plot(episodes, baseline_survival, alpha=0.5, linewidth=1,
                     color='#3498db', label='Baseline')
        axes[2].plot(episodes, improved_survival, alpha=0.5, linewidth=1,
                     color='#e74c3c', label='Improved')

        if window > 1:
            baseline_ma = np.convolve(baseline_survival, np.ones(window) / window, mode='valid')
            improved_ma = np.convolve(improved_survival, np.ones(window) / window, mode='valid')
            axes[2].plot(ma_episodes, baseline_ma, linewidth=2.5, color='#2c3e50',
                         label=f'Baseline MA({window})')
            axes[2].plot(ma_episodes, improved_ma, linewidth=2.5, color='#8e44ad',
                         label=f'Improved MA({window})')

        axes[2].set_xlabel('Episode', fontweight='bold')
        axes[2].set_ylabel('Survival Time (steps)', fontweight='bold')
        axes[2].set_title('Survival Time Over Time', fontsize=13, fontweight='bold')
        axes[2].legend(loc='best')
        axes[2].grid(alpha=0.3)

        plt.tight_layout()
        save_path = self.output_dir / 'episode_traces.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: {save_path.name}")
        plt.close()

    def plot_action_comparison(self):
        """Compare action distributions between models."""
        baseline_actions = self.baseline['action_distribution']
        improved_actions = self.improved['action_distribution']

        # Get all actions
        all_actions = sorted(set(list(baseline_actions.keys()) + list(improved_actions.keys())))

        baseline_counts = [baseline_actions.get(str(a), 0) for a in all_actions]
        improved_counts = [improved_actions.get(str(a), 0) for a in all_actions]

        baseline_total = sum(baseline_counts)
        improved_total = sum(improved_counts)

        baseline_pcts = [c / baseline_total * 100 for c in baseline_counts]
        improved_pcts = [c / improved_total * 100 for c in improved_counts]

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        x = np.arange(len(all_actions))
        width = 0.35

        # Side-by-side bar chart
        bars1 = axes[0].bar(x - width / 2, baseline_pcts, width, label='Baseline',
                            color='#3498db', edgecolor='black')
        bars2 = axes[0].bar(x + width / 2, improved_pcts, width, label='Improved',
                            color='#e74c3c', edgecolor='black')

        axes[0].set_xlabel('Action ID', fontsize=12, fontweight='bold')
        axes[0].set_ylabel('Percentage (%)', fontsize=12, fontweight='bold')
        axes[0].set_title('Action Distribution Comparison', fontsize=13, fontweight='bold')
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(all_actions)
        axes[0].legend()
        axes[0].grid(axis='y', alpha=0.3)

        # Difference plot
        differences = [imp - base for base, imp in zip(baseline_pcts, improved_pcts)]
        colors = ['#27ae60' if d >= 0 else '#c0392b' for d in differences]

        bars = axes[1].bar(x, differences, color=colors, edgecolor='black')
        axes[1].set_xlabel('Action ID', fontsize=12, fontweight='bold')
        axes[1].set_ylabel('Difference (percentage points)', fontsize=12, fontweight='bold')
        axes[1].set_title('Action Distribution Difference (Improved - Baseline)',
                          fontsize=13, fontweight='bold')
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(all_actions)
        axes[1].axhline(y=0, color='black', linestyle='-', linewidth=1)
        axes[1].grid(axis='y', alpha=0.3)

        plt.tight_layout()
        save_path = self.output_dir / 'action_distribution_comparison.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: {save_path.name}")
        plt.close()

    def generate_report(self):
        """Generate a comprehensive text report."""
        report_path = self.output_dir / 'comparison_report.txt'

        with open(report_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("DQN MODEL COMPARISON REPORT\n")
            f.write("=" * 80 + "\n\n")

            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            f.write("MODELS COMPARED:\n")
            f.write(f"  Baseline: {self.baseline_path.name}\n")
            f.write(f"  Improved: {self.improved_path.name}\n")
            f.write(f"  Episodes: {self.baseline['num_episodes']}\n\n")

            f.write("=" * 80 + "\n")
            f.write("PERFORMANCE SUMMARY\n")
            f.write("=" * 80 + "\n\n")

            # Rewards
            base_reward = self.baseline['metrics']['average_reward']
            imp_reward = self.improved['metrics']['average_reward']
            reward_improvement = ((imp_reward - base_reward) / abs(base_reward) * 100)

            f.write("CUMULATIVE REWARD:\n")
            f.write(f"  Baseline: {base_reward:.2f} ± {self.baseline['metrics']['std_reward']:.2f}\n")
            f.write(f"  Improved: {imp_reward:.2f} ± {self.improved['metrics']['std_reward']:.2f}\n")
            f.write(f"  Change:   {imp_reward - base_reward:+.2f} ({reward_improvement:+.2f}%)\n\n")

            # Achievements
            base_ach = self.baseline['metrics']['average_achievements_per_episode']
            imp_ach = self.improved['metrics']['average_achievements_per_episode']
            ach_improvement = ((imp_ach - base_ach) / base_ach * 100)

            f.write("ACHIEVEMENTS PER EPISODE:\n")
            f.write(f"  Baseline: {base_ach:.2f}\n")
            f.write(f"  Improved: {imp_ach:.2f}\n")
            f.write(f"  Change:   {imp_ach - base_ach:+.2f} ({ach_improvement:+.2f}%)\n\n")

            # Survival
            base_surv = self.baseline['metrics']['average_survival_time']
            imp_surv = self.improved['metrics']['average_survival_time']
            surv_improvement = ((imp_surv - base_surv) / base_surv * 100)

            f.write("SURVIVAL TIME (steps):\n")
            f.write(f"  Baseline: {base_surv:.2f} ± {self.baseline['metrics']['std_survival_time']:.2f}\n")
            f.write(f"  Improved: {imp_surv:.2f} ± {self.improved['metrics']['std_survival_time']:.2f}\n")
            f.write(f"  Change:   {imp_surv - base_surv:+.2f} ({surv_improvement:+.2f}%)\n\n")

            # Geometric mean
            base_geom = self.baseline['metrics']['geometric_mean_achievements']
            imp_geom = self.improved['metrics']['geometric_mean_achievements']

            f.write("GEOMETRIC MEAN OF ACHIEVEMENTS:\n")
            f.write(f"  Baseline: {base_geom:.6f}\n")
            f.write(f"  Improved: {imp_geom:.6f}\n")
            f.write(f"  Change:   {imp_geom - base_geom:+.6f}\n\n")

            f.write("=" * 80 + "\n")
            f.write("ACHIEVEMENT IMPROVEMENTS\n")
            f.write("=" * 80 + "\n\n")

            baseline_rates = self.baseline['achievement_unlock_rates']
            improved_rates = self.improved['achievement_unlock_rates']

            improvements = {ach: (improved_rates[ach] - baseline_rates[ach]) * 100
                            for ach in baseline_rates.keys()}

            sorted_improvements = sorted(improvements.items(), key=lambda x: x[1], reverse=True)

            f.write("Top 10 Most Improved Achievements:\n")
            for i, (ach, improvement) in enumerate(sorted_improvements[:10], 1):
                base_rate = baseline_rates[ach] * 100
                imp_rate = improved_rates[ach] * 100
                f.write(f"{i:2d}. {ach:25s}: {base_rate:5.1f}% → {imp_rate:5.1f}% "
                        f"({improvement:+.1f} pp)\n")

            f.write("\n")

            # Count achievements that got worse
            worse = [ach for ach, imp in improvements.items() if imp < -1.0]
            if worse:
                f.write(f"Achievements with Decreased Rates (> 1 pp): {len(worse)}\n")
                for ach in sorted(worse, key=lambda x: improvements[x]):
                    improvement = improvements[ach]
                    base_rate = baseline_rates[ach] * 100
                    imp_rate = improved_rates[ach] * 100
                    f.write(f"  - {ach:25s}: {base_rate:5.1f}% → {imp_rate:5.1f}% "
                            f"({improvement:.1f} pp)\n")

            f.write("\n")
            f.write("=" * 80 + "\n")
            f.write("END OF REPORT\n")
            f.write("=" * 80 + "\n")

        print(f"  ✓ Saved: {report_path.name}")


def main():
    parser = argparse.ArgumentParser(
        description='Compare DQN baseline and improved models'
    )
    parser.add_argument(
        '--baseline',
        type=str,
        default='./results/dqn_baseline_results.json',
        help='Path to baseline results JSON'
    )
    parser.add_argument(
        '--improved',
        type=str,
        default='./results/dqn_improved_results.json',
        help='Path to improved results JSON'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./comparison_results/',
        help='Directory to save comparison outputs'
    )

    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("DQN MODEL COMPARISON")
    print("=" * 70)
    print(f"Baseline: {args.baseline}")
    print(f"Improved: {args.improved}")
    print(f"Output:   {args.output_dir}")
    print("=" * 70 + "\n")

    # Create comparison instance
    comparison = ModelComparison(
        baseline_path=args.baseline,
        improved_path=args.improved,
        output_dir=args.output_dir
    )

    # Generate all comparisons
    comparison.generate_full_comparison()

    print("\n✓ Comparison complete!\n")


if __name__ == "__main__":
    main()
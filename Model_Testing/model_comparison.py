"""
Enhanced Model Comparison Script for DQN Models
Supports flexible comparisons: baseline vs reward, baseline vs preprocessed, etc.
Uses appropriate visualization formats (graphs over tables where suitable).
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

    def __init__(self, model_paths, model_names=None, output_dir='./comparison_results/', comparison_type='full'):
        """
        Initialize comparison with paths to result files.

        Args:
            model_paths: List of paths to model results JSONs
            model_names: List of names for each model (optional)
            output_dir: Directory to save comparison outputs
            comparison_type: Type of comparison ('full', 'baseline_vs_reward', 'baseline_vs_preprocessed', 'custom')
        """
        self.model_paths = [Path(p) for p in model_paths]
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.comparison_type = comparison_type

        # Load all models
        self.models = [self._load_results(path) for path in self.model_paths]

        # Set model names
        if model_names is None:
            self.model_names = [f"Model {i+1}" for i in range(len(self.models))]
        else:
            self.model_names = model_names

        # Define colors for each model (max 5 models)
        self.colors = ['#3498db', '#f39c12', '#e74c3c', '#9b59b6', '#27ae60'][:len(self.models)]

        # Identify baseline (always the first model)
        self.baseline_idx = 0

        print(f"\n{'='*70}")
        print(f"COMPARISON TYPE: {comparison_type.upper().replace('_', ' ')}")
        print(f"{'='*70}")
        print(f"\nLoaded {len(self.models)} models for comparison:")
        for i, (name, model) in enumerate(zip(self.model_names, self.models)):
            baseline_marker = " [BASELINE]" if i == self.baseline_idx else ""
            print(f"  {name}{baseline_marker}: {model['num_episodes']} episodes")

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

        # 2. Performance metrics comparison (GRAPHS, not tables)
        print("2. Creating performance metrics comparison (graphs)...")
        self.plot_performance_metrics_separate()

        # 3. Achievement comparison
        print("3. Creating achievement comparison...")
        self.plot_achievement_comparison()

        # 4. Distribution comparison (GRAPHS)
        print("4. Creating distribution comparisons (graphs)...")
        self.plot_distributions_separate()

        # 5. Episode-by-episode comparison (LINE GRAPHS)
        print("5. Creating episode traces (line graphs)...")
        self.plot_episode_traces()

        # 6. Action distribution comparison
        print("6. Creating action distribution comparison...")
        self.plot_action_comparison()

        # 7. Progressive improvement visualization (only if 3+ models)
        if len(self.models) >= 3:
            print("7. Creating progressive improvement visualization...")
            self.plot_progressive_improvement()
        else:
            print("7. Skipping progressive improvement (requires 3+ models)...")

        # 8. Direct comparison plots (for 2-model comparisons)
        if len(self.models) == 2:
            print("8. Creating direct comparison visualizations...")
            self.plot_direct_comparison()

        # 9. Generate text report
        print(f"{8 if len(self.models) == 2 else 9}. Generating comparison report...")
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

        baseline_name = self.model_names[self.baseline_idx]

        # Reward comparison
        print("\n📊 REWARD COMPARISON (Standard Rewards)")
        print("-" * 70)
        print(f"{'Model':<25} {'Mean':>12} {'Std':>12} {'vs Baseline':>15}")
        print("-" * 70)

        baseline_reward_mean = np.mean(all_rewards[self.baseline_idx])
        for i, (name, rewards) in enumerate(zip(self.model_names, all_rewards)):
            mean_val = np.mean(rewards)
            std_val = np.std(rewards)
            if i == self.baseline_idx:
                improvement = "-"
            else:
                improvement = f"{((mean_val - baseline_reward_mean) / abs(baseline_reward_mean) * 100):+.2f}%"
            print(f"{name:<25} {mean_val:>12.2f} {std_val:>12.2f} {improvement:>15}")

        # Pairwise t-tests for rewards (focus on vs baseline)
        print("\n  Statistical Tests vs Baseline (Rewards):")
        for i in range(len(self.models)):
            if i != self.baseline_idx:
                t_stat, p_value = stats.ttest_ind(all_rewards[self.baseline_idx], all_rewards[i])
                sig = "✓ Significant" if p_value < 0.05 else "✗ Not significant"
                print(f"    {baseline_name} vs {self.model_names[i]}: "
                      f"p={p_value:.6f} ({sig})")

        # Achievement comparison
        print("\n🏆 ACHIEVEMENT COMPARISON")
        print("-" * 70)
        print(f"{'Model':<25} {'Mean':>12} {'Std':>12} {'vs Baseline':>15}")
        print("-" * 70)

        baseline_ach_mean = np.mean(all_achievements[self.baseline_idx])
        for i, (name, achievements) in enumerate(zip(self.model_names, all_achievements)):
            mean_val = np.mean(achievements)
            std_val = np.std(achievements)
            if i == self.baseline_idx:
                improvement = "-"
            else:
                improvement = f"{((mean_val - baseline_ach_mean) / baseline_ach_mean * 100):+.2f}%"
            print(f"{name:<25} {mean_val:>12.2f} {std_val:>12.2f} {improvement:>15}")

        # Pairwise t-tests for achievements (focus on vs baseline)
        print("\n  Statistical Tests vs Baseline (Achievements):")
        for i in range(len(self.models)):
            if i != self.baseline_idx:
                t_stat, p_value = stats.ttest_ind(all_achievements[self.baseline_idx], all_achievements[i])
                sig = "✓ Significant" if p_value < 0.05 else "✗ Not significant"
                print(f"    {baseline_name} vs {self.model_names[i]}: "
                      f"p={p_value:.6f} ({sig})")

        # Survival time comparison
        print("\n⏱️  SURVIVAL TIME COMPARISON")
        print("-" * 70)
        print(f"{'Model':<25} {'Mean (steps)':>15} {'Std':>12} {'vs Baseline':>15}")
        print("-" * 70)

        baseline_surv_mean = np.mean(all_survival[self.baseline_idx])
        for i, (name, survival) in enumerate(zip(self.model_names, all_survival)):
            mean_val = np.mean(survival)
            std_val = np.std(survival)
            if i == self.baseline_idx:
                improvement = "-"
            else:
                improvement = f"{((mean_val - baseline_surv_mean) / baseline_surv_mean * 100):+.2f}%"
            print(f"{name:<25} {mean_val:>15.2f} {std_val:>12.2f} {improvement:>15}")

        # Effect sizes vs baseline
        print("\n📏 EFFECT SIZES (Cohen's d vs Baseline)")
        print("-" * 70)
        for i in range(len(self.models)):
            if i != self.baseline_idx:
                reward_effect = self._cohens_d(all_rewards[self.baseline_idx], all_rewards[i])
                achievement_effect = self._cohens_d(all_achievements[self.baseline_idx], all_achievements[i])
                survival_effect = self._cohens_d(all_survival[self.baseline_idx], all_survival[i])

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

    def plot_performance_metrics_separate(self):
        """Create SEPARATE bar charts for each metric (GRAPHS, not tables)."""
        x = np.arange(len(self.models))
        width = 0.6

        # 1. Average Rewards
        fig, ax = plt.subplots(figsize=(10, 6))
        avg_rewards = [model['metrics']['average_reward'] for model in self.models]
        std_rewards = [model['metrics']['std_reward'] for model in self.models]

        bars = ax.bar(x, avg_rewards, width, yerr=std_rewards, capsize=8,
                      color=self.colors, edgecolor='black', linewidth=1.5, alpha=0.8)

        # Highlight baseline
        bars[self.baseline_idx].set_edgecolor('red')
        bars[self.baseline_idx].set_linewidth(3)

        ax.set_ylabel('Average Cumulative Reward (Standard)', fontsize=12, fontweight='bold')
        ax.set_title('Reward Comparison Across Models', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(self.model_names, rotation=15, ha='right')
        ax.grid(axis='y', alpha=0.3, linestyle='--')

        for bar, val, std in zip(bars, avg_rewards, std_rewards):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + std + 0.5,
                    f'{val:.1f}', ha='center', va='bottom', fontweight='bold', fontsize=10)

        plt.tight_layout()
        save_path = self.output_dir / 'metric_reward_comparison.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: {save_path.name}")
        plt.close()

        # 2. Average Achievements
        fig, ax = plt.subplots(figsize=(10, 6))
        avg_achievements = [model['metrics']['average_achievements_per_episode'] for model in self.models]

        bars = ax.bar(x, avg_achievements, width, color=self.colors,
                      edgecolor='black', linewidth=1.5, alpha=0.8)
        bars[self.baseline_idx].set_edgecolor('red')
        bars[self.baseline_idx].set_linewidth(3)

        ax.set_ylabel('Avg Achievements per Episode', fontsize=12, fontweight='bold')
        ax.set_title('Achievement Count Comparison', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(self.model_names, rotation=15, ha='right')
        ax.grid(axis='y', alpha=0.3, linestyle='--')

        for bar, val in zip(bars, avg_achievements):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                    f'{val:.2f}', ha='center', va='bottom', fontweight='bold', fontsize=10)

        plt.tight_layout()
        save_path = self.output_dir / 'metric_achievement_comparison.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: {save_path.name}")
        plt.close()

        # 3. Survival Time
        fig, ax = plt.subplots(figsize=(10, 6))
        avg_survival = [model['metrics']['average_survival_time'] for model in self.models]
        std_survival = [model['metrics']['std_survival_time'] for model in self.models]

        bars = ax.bar(x, avg_survival, width, yerr=std_survival, capsize=8,
                      color=self.colors, edgecolor='black', linewidth=1.5, alpha=0.8)
        bars[self.baseline_idx].set_edgecolor('red')
        bars[self.baseline_idx].set_linewidth(3)

        ax.set_ylabel('Average Survival Time (steps)', fontsize=12, fontweight='bold')
        ax.set_title('Survival Time Comparison', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(self.model_names, rotation=15, ha='right')
        ax.grid(axis='y', alpha=0.3, linestyle='--')

        for bar, val, std in zip(bars, avg_survival, std_survival):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + std + 5,
                    f'{val:.0f}', ha='center', va='bottom', fontweight='bold', fontsize=10)

        plt.tight_layout()
        save_path = self.output_dir / 'metric_survival_comparison.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: {save_path.name}")
        plt.close()

        # 4. Geometric Mean
        fig, ax = plt.subplots(figsize=(10, 6))
        geom_means = [model['metrics']['geometric_mean_achievements'] for model in self.models]

        bars = ax.bar(x, geom_means, width, color=self.colors,
                      edgecolor='black', linewidth=1.5, alpha=0.8)
        bars[self.baseline_idx].set_edgecolor('red')
        bars[self.baseline_idx].set_linewidth(3)

        ax.set_ylabel('Geometric Mean Score', fontsize=12, fontweight='bold')
        ax.set_title('Overall Achievement Score (Geometric Mean)', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(self.model_names, rotation=15, ha='right')
        ax.grid(axis='y', alpha=0.3, linestyle='--')

        for bar, val in zip(bars, geom_means):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.0005,
                    f'{val:.4f}', ha='center', va='bottom', fontweight='bold', fontsize=10)

        plt.tight_layout()
        save_path = self.output_dir / 'metric_geometric_mean_comparison.png'
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
        baseline_rates = self.models[self.baseline_idx]['achievement_unlock_rates']
        avg_improvements = {}
        for ach in all_achievements:
            improvements = [model['achievement_unlock_rates'][ach] - baseline_rates[ach]
                           for i, model in enumerate(self.models) if i != self.baseline_idx]
            avg_improvements[ach] = np.mean(improvements) if improvements else 0

        sorted_achievements = sorted(all_achievements, key=lambda x: avg_improvements[x], reverse=True)

        fig, axes = plt.subplots(1, 2, figsize=(20, max(8, len(all_achievements) * 0.35)))

        # 1. Side-by-side comparison
        y_pos = np.arange(len(sorted_achievements))
        bar_width = 0.8 / len(self.models)

        for i, (model, name, color) in enumerate(zip(self.models, self.model_names, self.colors)):
            rates = [model['achievement_unlock_rates'][ach] * 100 for ach in sorted_achievements]
            offset = (i - len(self.models)/2 + 0.5) * bar_width
            alpha = 1.0 if i == self.baseline_idx else 0.7
            linewidth = 2 if i == self.baseline_idx else 1
            axes[0].barh(y_pos + offset, rates, bar_width, label=name,
                        color=color, edgecolor='black', alpha=alpha, linewidth=linewidth)

        axes[0].set_yticks(y_pos)
        axes[0].set_yticklabels(sorted_achievements, fontsize=9)
        axes[0].set_xlabel('Unlock Rate (%)', fontsize=12, fontweight='bold')
        axes[0].set_title('Achievement Unlock Rates Comparison', fontsize=13, fontweight='bold')
        axes[0].legend(loc='lower right', fontsize=10)
        axes[0].grid(axis='x', alpha=0.3, linestyle='--')

        # 2. Improvement heatmap (vs baseline)
        if len(self.models) > 1:
            improvement_matrix = []
            for ach in sorted_achievements:
                improvements = [model['achievement_unlock_rates'][ach] - baseline_rates[ach]
                               for i, model in enumerate(self.models) if i != self.baseline_idx]
                improvement_matrix.append([x * 100 for x in improvements])

            improvement_matrix = np.array(improvement_matrix)

            im = axes[1].imshow(improvement_matrix, cmap='RdYlGn', aspect='auto',
                               vmin=-20, vmax=20)
            axes[1].set_yticks(y_pos)
            axes[1].set_yticklabels(sorted_achievements, fontsize=9)
            axes[1].set_xticks(range(len(self.models) - 1))
            non_baseline_names = [name for i, name in enumerate(self.model_names) if i != self.baseline_idx]
            axes[1].set_xticklabels(non_baseline_names, rotation=15, ha='right')
            axes[1].set_title(f'Improvement vs {self.model_names[self.baseline_idx]} (percentage points)',
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
        else:
            axes[1].text(0.5, 0.5, 'Heatmap requires\nmultiple models',
                        ha='center', va='center', transform=axes[1].transAxes,
                        fontsize=14, fontweight='bold')
            axes[1].set_xticks([])
            axes[1].set_yticks([])

        plt.tight_layout()
        save_path = self.output_dir / 'achievement_comparison.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: {save_path.name}")
        plt.close()

    def plot_distributions_separate(self):
        """Plot SEPARATE distribution comparisons (GRAPHS)."""
        all_rewards = [np.array(model['episode_rewards']) for model in self.models]
        all_achievements = [np.array(model['achievements_per_episode']) for model in self.models]
        all_survival = [np.array(model['episode_lengths']) for model in self.models]

        # 1. Reward Histogram
        fig, ax = plt.subplots(figsize=(10, 6))
        for i, (rewards, name, color) in enumerate(zip(all_rewards, self.model_names, self.colors)):
            alpha = 0.6 if i == self.baseline_idx else 0.4
            linewidth = 2.5 if i == self.baseline_idx else 2
            ax.hist(rewards, bins=30, alpha=alpha, label=name,
                   color=color, edgecolor='black', linewidth=0.5)
            ax.axvline(np.mean(rewards), color=color, linestyle='--', linewidth=linewidth)

        ax.set_xlabel('Cumulative Reward (Standard)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
        ax.set_title('Reward Distribution Comparison', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(alpha=0.3)

        plt.tight_layout()
        save_path = self.output_dir / 'dist_reward_histogram.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: {save_path.name}")
        plt.close()

        # 2. Achievement Histogram
        fig, ax = plt.subplots(figsize=(10, 6))
        max_ach = max([ach.max() for ach in all_achievements])
        bins = np.arange(0, max_ach + 2) - 0.5

        for i, (achievements, name, color) in enumerate(zip(all_achievements, self.model_names, self.colors)):
            alpha = 0.6 if i == self.baseline_idx else 0.4
            linewidth = 2.5 if i == self.baseline_idx else 2
            ax.hist(achievements, bins=bins, alpha=alpha, label=name,
                   color=color, edgecolor='black', linewidth=0.5)
            ax.axvline(np.mean(achievements), color=color, linestyle='--', linewidth=linewidth)

        ax.set_xlabel('Achievements per Episode', fontsize=12, fontweight='bold')
        ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
        ax.set_title('Achievement Distribution Comparison', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(alpha=0.3)

        plt.tight_layout()
        save_path = self.output_dir / 'dist_achievement_histogram.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: {save_path.name}")
        plt.close()

        # 3. Survival Histogram
        fig, ax = plt.subplots(figsize=(10, 6))
        for i, (survival, name, color) in enumerate(zip(all_survival, self.model_names, self.colors)):
            alpha = 0.6 if i == self.baseline_idx else 0.4
            linewidth = 2.5 if i == self.baseline_idx else 2
            ax.hist(survival, bins=30, alpha=alpha, label=name,
                   color=color, edgecolor='black', linewidth=0.5)
            ax.axvline(np.mean(survival), color=color, linestyle='--', linewidth=linewidth)

        ax.set_xlabel('Survival Time (steps)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
        ax.set_title('Survival Time Distribution Comparison', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(alpha=0.3)

        plt.tight_layout()
        save_path = self.output_dir / 'dist_survival_histogram.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: {save_path.name}")
        plt.close()

        # 4-6. Box Plots
        for data_list, ylabel, title, filename in [
            (all_rewards, 'Cumulative Reward (Standard)', 'Reward Box Plot Comparison', 'dist_reward_boxplot.png'),
            (all_achievements, 'Achievements per Episode', 'Achievement Box Plot Comparison', 'dist_achievement_boxplot.png'),
            (all_survival, 'Survival Time (steps)', 'Survival Time Box Plot Comparison', 'dist_survival_boxplot.png')
        ]:
            fig, ax = plt.subplots(figsize=(10, 6))
            bp = ax.boxplot(data_list, labels=self.model_names,
                            patch_artist=True, showmeans=True)
            for i, (patch, color) in enumerate(zip(bp['boxes'], self.colors)):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
                if i == self.baseline_idx:
                    patch.set_edgecolor('red')
                    patch.set_linewidth(2.5)
            ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')
            ax.set_title(title, fontsize=14, fontweight='bold')
            ax.grid(alpha=0.3, axis='y')
            ax.tick_params(axis='x', rotation=15)

            plt.tight_layout()
            save_path = self.output_dir / filename
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"  ✓ Saved: {save_path.name}")
            plt.close()

    def plot_episode_traces(self):
        """Plot episode-by-episode performance traces (LINE GRAPHS)."""
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
        for i, (rewards, name, color) in enumerate(zip(all_rewards, self.model_names, self.colors)):
            rewards_trimmed = rewards[:min_episodes]
            alpha = 0.4 if i == self.baseline_idx else 0.2
            linewidth_raw = 1.2 if i == self.baseline_idx else 1
            linewidth_ma = 3 if i == self.baseline_idx else 2.5

            axes[0].plot(episodes, rewards_trimmed, alpha=alpha, linewidth=linewidth_raw, color=color)

            if window > 1:
                ma = np.convolve(rewards_trimmed, np.ones(window) / window, mode='valid')
                ma_episodes = range(window, min_episodes + 1)
                axes[0].plot(ma_episodes, ma, linewidth=linewidth_ma, color=color, label=name)

        axes[0].set_xlabel('Episode', fontweight='bold')
        axes[0].set_ylabel('Cumulative Reward (Standard)', fontweight='bold')
        axes[0].set_title(f'Episode Rewards Over Time (MA window={window})',
                         fontsize=13, fontweight='bold')
        axes[0].legend(loc='best', fontsize=10)
        axes[0].grid(alpha=0.3)

        # Achievements trace
        for i, (achievements, name, color) in enumerate(zip(all_achievements, self.model_names, self.colors)):
            ach_trimmed = achievements[:min_episodes]
            alpha = 0.4 if i == self.baseline_idx else 0.2
            linewidth_raw = 1.2 if i == self.baseline_idx else 1
            linewidth_ma = 3 if i == self.baseline_idx else 2.5

            axes[1].plot(episodes, ach_trimmed, alpha=alpha, linewidth=linewidth_raw, color=color)

            if window > 1:
                ma = np.convolve(ach_trimmed, np.ones(window) / window, mode='valid')
                ma_episodes = range(window, min_episodes + 1)
                axes[1].plot(ma_episodes, ma, linewidth=linewidth_ma, color=color, label=name)

        axes[1].set_xlabel('Episode', fontweight='bold')
        axes[1].set_ylabel('Achievements Unlocked', fontweight='bold')
        axes[1].set_title(f'Achievements Over Time (MA window={window})',
                         fontsize=13, fontweight='bold')
        axes[1].legend(loc='best', fontsize=10)
        axes[1].grid(alpha=0.3)

        # Survival trace
        for i, (survival, name, color) in enumerate(zip(all_survival, self.model_names, self.colors)):
            surv_trimmed = survival[:min_episodes]
            alpha = 0.4 if i == self.baseline_idx else 0.2
            linewidth_raw = 1.2 if i == self.baseline_idx else 1
            linewidth_ma = 3 if i == self.baseline_idx else 2.5

            axes[2].plot(episodes, surv_trimmed, alpha=alpha, linewidth=linewidth_raw, color=color)

            if window > 1:
                ma = np.convolve(surv_trimmed, np.ones(window) / window, mode='valid')
                ma_episodes = range(window, min_episodes + 1)
                axes[2].plot(ma_episodes, ma, linewidth=linewidth_ma, color=color, label=name)

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
        """Compare action distributions across models."""
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # Get action distributions
        action_dists = []
        max_action = 0
        for model in self.models:
            action_dist = model.get('action_distribution', {})
            if action_dist:
                max_action = max(max_action, max(map(int, action_dist.keys())))
                action_dists.append(action_dist)
            else:
                action_dists.append({})

        if not any(action_dists):
            print("  ⚠ Warning: No action distribution data available")
            plt.close()
            return

        # Prepare action counts for all models
        action_labels = [f"Action {i}" for i in range(max_action + 1)]
        x = np.arange(len(action_labels))
        bar_width = 0.8 / len(self.models)

        # 1. Raw action counts
        for i, (action_dist, name, color) in enumerate(zip(action_dists, self.model_names, self.colors)):
            if action_dist:
                counts = [action_dist.get(j, 0) for j in range(max_action + 1)]
                offset = (i - len(self.models)/2 + 0.5) * bar_width
                alpha = 0.9 if i == self.baseline_idx else 0.7
                linewidth = 2 if i == self.baseline_idx else 1
                axes[0].bar(x + offset, counts, bar_width, label=name,
                           color=color, edgecolor='black', alpha=alpha, linewidth=linewidth)

        axes[0].set_xlabel('Action', fontsize=12, fontweight='bold')
        axes[0].set_ylabel('Count', fontsize=12, fontweight='bold')
        axes[0].set_title('Action Distribution (Raw Counts)', fontsize=13, fontweight='bold')
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(action_labels, rotation=45, ha='right')
        axes[0].legend(fontsize=10)
        axes[0].grid(alpha=0.3, axis='y')

        # 2. Normalized action proportions
        for i, (action_dist, name, color) in enumerate(zip(action_dists, self.model_names, self.colors)):
            if action_dist:
                total = sum(action_dist.values())
                proportions = [(action_dist.get(j, 0) / total * 100) for j in range(max_action + 1)]
                offset = (i - len(self.models)/2 + 0.5) * bar_width
                alpha = 0.9 if i == self.baseline_idx else 0.7
                linewidth = 2 if i == self.baseline_idx else 1
                axes[1].bar(x + offset, proportions, bar_width, label=name,
                           color=color, edgecolor='black', alpha=alpha, linewidth=linewidth)

        axes[1].set_xlabel('Action', fontsize=12, fontweight='bold')
        axes[1].set_ylabel('Proportion (%)', fontsize=12, fontweight='bold')
        axes[1].set_title('Action Distribution (Normalized)', fontsize=13, fontweight='bold')
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(action_labels, rotation=45, ha='right')
        axes[1].legend(fontsize=10)
        axes[1].grid(alpha=0.3, axis='y')

        plt.tight_layout()
        save_path = self.output_dir / 'action_distribution_comparison.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: {save_path.name}")
        plt.close()

    def plot_progressive_improvement(self):
        """Visualize progressive improvement through model iterations (3+ models)."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Extract key metrics
        rewards = [model['metrics']['average_reward'] for model in self.models]
        achievements = [model['metrics']['average_achievements_per_episode'] for model in self.models]
        survival = [model['metrics']['average_survival_time'] for model in self.models]
        geom_means = [model['metrics']['geometric_mean_achievements'] for model in self.models]

        x = np.arange(len(self.models))

        # 1. Reward improvement
        axes[0, 0].plot(x, rewards, marker='o', linewidth=2.5, markersize=10,
                       color='#3498db', markerfacecolor='white', markeredgewidth=2)
        axes[0, 0].scatter(self.baseline_idx, rewards[self.baseline_idx],
                          s=200, color='red', marker='*', zorder=5, label='Baseline')
        for i, (xi, yi) in enumerate(zip(x, rewards)):
            axes[0, 0].text(xi, yi + 0.5, f'{yi:.1f}', ha='center', va='bottom',
                           fontweight='bold', fontsize=10)
        axes[0, 0].set_ylabel('Avg Cumulative Reward', fontweight='bold')
        axes[0, 0].set_title('Reward Progression', fontsize=12, fontweight='bold')
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels(self.model_names, rotation=15, ha='right')
        axes[0, 0].legend()
        axes[0, 0].grid(alpha=0.3)

        # 2. Achievement improvement
        axes[0, 1].plot(x, achievements, marker='s', linewidth=2.5, markersize=10,
                       color='#f39c12', markerfacecolor='white', markeredgewidth=2)
        axes[0, 1].scatter(self.baseline_idx, achievements[self.baseline_idx],
                          s=200, color='red', marker='*', zorder=5)
        for i, (xi, yi) in enumerate(zip(x, achievements)):
            axes[0, 1].text(xi, yi + 0.05, f'{yi:.2f}', ha='center', va='bottom',
                           fontweight='bold', fontsize=10)
        axes[0, 1].set_ylabel('Avg Achievements', fontweight='bold')
        axes[0, 1].set_title('Achievement Progression', fontsize=12, fontweight='bold')
        axes[0, 1].set_xticks(x)
        axes[0, 1].set_xticklabels(self.model_names, rotation=15, ha='right')
        axes[0, 1].grid(alpha=0.3)

        # 3. Survival improvement
        axes[1, 0].plot(x, survival, marker='^', linewidth=2.5, markersize=10,
                       color='#27ae60', markerfacecolor='white', markeredgewidth=2)
        axes[1, 0].scatter(self.baseline_idx, survival[self.baseline_idx],
                          s=200, color='red', marker='*', zorder=5)
        for i, (xi, yi) in enumerate(zip(x, survival)):
            axes[1, 0].text(xi, yi + 5, f'{yi:.0f}', ha='center', va='bottom',
                           fontweight='bold', fontsize=10)
        axes[1, 0].set_ylabel('Avg Survival Time (steps)', fontweight='bold')
        axes[1, 0].set_title('Survival Progression', fontsize=12, fontweight='bold')
        axes[1, 0].set_xticks(x)
        axes[1, 0].set_xticklabels(self.model_names, rotation=15, ha='right')
        axes[1, 0].grid(alpha=0.3)

        # 4. Geometric mean improvement
        axes[1, 1].plot(x, geom_means, marker='D', linewidth=2.5, markersize=10,
                       color='#e74c3c', markerfacecolor='white', markeredgewidth=2)
        axes[1, 1].scatter(self.baseline_idx, geom_means[self.baseline_idx],
                          s=200, color='red', marker='*', zorder=5)
        for i, (xi, yi) in enumerate(zip(x, geom_means)):
            axes[1, 1].text(xi, yi + 0.0005, f'{yi:.4f}', ha='center', va='bottom',
                           fontweight='bold', fontsize=10)
        axes[1, 1].set_ylabel('Geometric Mean Score', fontweight='bold')
        axes[1, 1].set_title('Overall Score Progression', fontsize=12, fontweight='bold')
        axes[1, 1].set_xticks(x)
        axes[1, 1].set_xticklabels(self.model_names, rotation=15, ha='right')
        axes[1, 1].grid(alpha=0.3)

        plt.tight_layout()
        save_path = self.output_dir / 'progressive_improvement.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: {save_path.name}")
        plt.close()

    def plot_direct_comparison(self):
        """Create direct comparison visualizations for 2-model comparisons."""
        if len(self.models) != 2:
            return

        baseline_name = self.model_names[self.baseline_idx]
        other_idx = 1 - self.baseline_idx
        other_name = self.model_names[other_idx]

        # Create a 2x2 comparison grid
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'{baseline_name} vs {other_name}: Direct Comparison',
                     fontsize=16, fontweight='bold')

        # 1. Scatter: Reward comparison
        baseline_rewards = np.array(self.models[self.baseline_idx]['episode_rewards'])
        other_rewards = np.array(self.models[other_idx]['episode_rewards'])
        min_len = min(len(baseline_rewards), len(other_rewards))

        axes[0, 0].scatter(baseline_rewards[:min_len], other_rewards[:min_len],
                          alpha=0.5, s=30, color=self.colors[other_idx])
        max_val = max(baseline_rewards[:min_len].max(), other_rewards[:min_len].max())
        min_val = min(baseline_rewards[:min_len].min(), other_rewards[:min_len].min())
        axes[0, 0].plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=2, label='Equal Performance')
        axes[0, 0].set_xlabel(f'{baseline_name} Reward', fontweight='bold')
        axes[0, 0].set_ylabel(f'{other_name} Reward', fontweight='bold')
        axes[0, 0].set_title('Episode Reward Comparison', fontweight='bold')
        axes[0, 0].legend()
        axes[0, 0].grid(alpha=0.3)

        # 2. Scatter: Achievement comparison
        baseline_ach = np.array(self.models[self.baseline_idx]['achievements_per_episode'])
        other_ach = np.array(self.models[other_idx]['achievements_per_episode'])
        min_len = min(len(baseline_ach), len(other_ach))

        axes[0, 1].scatter(baseline_ach[:min_len], other_ach[:min_len],
                          alpha=0.5, s=30, color=self.colors[other_idx])
        max_val = max(baseline_ach[:min_len].max(), other_ach[:min_len].max())
        min_val = min(baseline_ach[:min_len].min(), other_ach[:min_len].min())
        axes[0, 1].plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=2, label='Equal Performance')
        axes[0, 1].set_xlabel(f'{baseline_name} Achievements', fontweight='bold')
        axes[0, 1].set_ylabel(f'{other_name} Achievements', fontweight='bold')
        axes[0, 1].set_title('Achievement Comparison', fontweight='bold')
        axes[0, 1].legend()
        axes[0, 1].grid(alpha=0.3)

        # 3. Difference over episodes: Reward
        reward_diff = other_rewards[:min_len] - baseline_rewards[:min_len]
        episodes = np.arange(1, min_len + 1)

        axes[1, 0].plot(episodes, reward_diff, alpha=0.3, linewidth=1, color=self.colors[other_idx])
        window = min(50, min_len // 10)
        if window > 1:
            ma = np.convolve(reward_diff, np.ones(window) / window, mode='valid')
            ma_episodes = np.arange(window, min_len + 1)
            axes[1, 0].plot(ma_episodes, ma, linewidth=2.5, color=self.colors[other_idx])
        axes[1, 0].axhline(0, color='black', linestyle='--', linewidth=1.5)
        axes[1, 0].fill_between(episodes, 0, reward_diff, where=(reward_diff >= 0),
                                alpha=0.3, color='green', label=f'{other_name} better')
        axes[1, 0].fill_between(episodes, 0, reward_diff, where=(reward_diff < 0),
                                alpha=0.3, color='red', label=f'{baseline_name} better')
        axes[1, 0].set_xlabel('Episode', fontweight='bold')
        axes[1, 0].set_ylabel(f'Reward Difference\n({other_name} - {baseline_name})', fontweight='bold')
        axes[1, 0].set_title(f'Reward Advantage Over Time (MA={window})', fontweight='bold')
        axes[1, 0].legend()
        axes[1, 0].grid(alpha=0.3)

        # 4. Difference over episodes: Achievements
        ach_diff = other_ach[:min_len] - baseline_ach[:min_len]

        axes[1, 1].plot(episodes, ach_diff, alpha=0.3, linewidth=1, color=self.colors[other_idx])
        if window > 1:
            ma = np.convolve(ach_diff, np.ones(window) / window, mode='valid')
            ma_episodes = np.arange(window, min_len + 1)
            axes[1, 1].plot(ma_episodes, ma, linewidth=2.5, color=self.colors[other_idx])
        axes[1, 1].axhline(0, color='black', linestyle='--', linewidth=1.5)
        axes[1, 1].fill_between(episodes, 0, ach_diff, where=(ach_diff >= 0),
                                alpha=0.3, color='green', label=f'{other_name} better')
        axes[1, 1].fill_between(episodes, 0, ach_diff, where=(ach_diff < 0),
                                alpha=0.3, color='red', label=f'{baseline_name} better')
        axes[1, 1].set_xlabel('Episode', fontweight='bold')
        axes[1, 1].set_ylabel(f'Achievement Difference\n({other_name} - {baseline_name})', fontweight='bold')
        axes[1, 1].set_title(f'Achievement Advantage Over Time (MA={window})', fontweight='bold')
        axes[1, 1].legend()
        axes[1, 1].grid(alpha=0.3)

        plt.tight_layout()
        save_path = self.output_dir / 'direct_comparison.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: {save_path.name}")
        plt.close()

    def generate_report(self):
        """Generate a comprehensive text report."""
        report_path = self.output_dir / 'comparison_report.txt'

        with open(report_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("COMPREHENSIVE MODEL COMPARISON REPORT\n")
            f.write("=" * 80 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Comparison Type: {self.comparison_type.replace('_', ' ').title()}\n")
            f.write(f"Models Compared: {len(self.models)}\n")
            f.write(f"Baseline Model: {self.model_names[self.baseline_idx]}\n")
            f.write("=" * 80 + "\n\n")

            # Model overview
            f.write("MODEL OVERVIEW\n")
            f.write("-" * 80 + "\n")
            for i, (name, model) in enumerate(zip(self.model_names, self.models)):
                baseline_marker = " [BASELINE]" if i == self.baseline_idx else ""
                f.write(f"\n{i+1}. {name}{baseline_marker}\n")
                f.write(f"   Model Type: {model.get('model', 'Unknown')}\n")
                f.write(f"   Episodes Evaluated: {model['num_episodes']}\n")
                f.write(f"   Evaluation Type: {model.get('evaluation_type', 'Unknown')}\n")
                f.write(f"   Preprocessing: {model.get('preprocessing', False)}\n")
                f.write(f"   Timestamp: {model.get('timestamp', 'Unknown')}\n")

            # Performance metrics
            f.write("\n" + "=" * 80 + "\n")
            f.write("PERFORMANCE METRICS SUMMARY\n")
            f.write("=" * 80 + "\n\n")

            f.write(f"{'Model':<30} {'Avg Reward':>12} {'Avg Ach':>10} {'Survival':>12} {'Geom Mean':>12}\n")
            f.write("-" * 80 + "\n")

            for i, (name, model) in enumerate(zip(self.model_names, self.models)):
                baseline_marker = " *" if i == self.baseline_idx else ""
                metrics = model['metrics']
                f.write(f"{name:<30}{baseline_marker} "
                       f"{metrics['average_reward']:>11.2f} "
                       f"{metrics['average_achievements_per_episode']:>10.2f} "
                       f"{metrics['average_survival_time']:>12.1f} "
                       f"{metrics['geometric_mean_achievements']:>12.6f}\n")

            # Relative improvements vs baseline
            f.write("\n" + "=" * 80 + "\n")
            f.write(f"RELATIVE IMPROVEMENTS (vs {self.model_names[self.baseline_idx]})\n")
            f.write("=" * 80 + "\n\n")

            baseline_metrics = self.models[self.baseline_idx]['metrics']

            for i, (name, model) in enumerate(zip(self.model_names, self.models)):
                if i == self.baseline_idx:
                    continue

                metrics = model['metrics']
                f.write(f"\n{name}:\n")

                reward_imp = ((metrics['average_reward'] - baseline_metrics['average_reward']) /
                             abs(baseline_metrics['average_reward']) * 100)
                f.write(f"  Reward: {reward_imp:+.2f}%\n")

                ach_imp = ((metrics['average_achievements_per_episode'] -
                           baseline_metrics['average_achievements_per_episode']) /
                          baseline_metrics['average_achievements_per_episode'] * 100)
                f.write(f"  Achievements: {ach_imp:+.2f}%\n")

                surv_imp = ((metrics['average_survival_time'] - baseline_metrics['average_survival_time']) /
                           baseline_metrics['average_survival_time'] * 100)
                f.write(f"  Survival: {surv_imp:+.2f}%\n")

                geom_imp = ((metrics['geometric_mean_achievements'] -
                            baseline_metrics['geometric_mean_achievements']) /
                           baseline_metrics['geometric_mean_achievements'] * 100)
                f.write(f"  Geometric Mean: {geom_imp:+.2f}%\n")

            # Top achievements
            f.write("\n" + "=" * 80 + "\n")
            f.write("TOP 10 ACHIEVEMENTS (by unlock rate)\n")
            f.write("=" * 80 + "\n\n")

            for name, model in zip(self.model_names, self.models):
                f.write(f"\n{name}:\n")
                sorted_ach = sorted(model['achievement_unlock_rates'].items(),
                                  key=lambda x: x[1], reverse=True)[:10]
                for ach, rate in sorted_ach:
                    f.write(f"  {ach:<30} {rate*100:6.2f}%\n")

            # Achievement improvements vs baseline
            f.write("\n" + "=" * 80 + "\n")
            f.write(f"ACHIEVEMENT UNLOCK RATE CHANGES (vs {self.model_names[self.baseline_idx]})\n")
            f.write("=" * 80 + "\n\n")

            baseline_ach_rates = self.models[self.baseline_idx]['achievement_unlock_rates']

            for i, (name, model) in enumerate(zip(self.model_names, self.models)):
                if i == self.baseline_idx:
                    continue

                f.write(f"\n{name}:\n")
                f.write("  Largest Improvements:\n")

                improvements = {}
                for ach, rate in model['achievement_unlock_rates'].items():
                    baseline_rate = baseline_ach_rates[ach]
                    improvements[ach] = (rate - baseline_rate) * 100

                sorted_improvements = sorted(improvements.items(), key=lambda x: x[1], reverse=True)
                for ach, imp in sorted_improvements[:5]:
                    f.write(f"    {ach:<30} {imp:+6.2f} pp\n")

                f.write("  Largest Declines:\n")
                for ach, imp in sorted_improvements[-5:]:
                    f.write(f"    {ach:<30} {imp:+6.2f} pp\n")

            # Summary and recommendations
            f.write("\n" + "=" * 80 + "\n")
            f.write("SUMMARY AND RECOMMENDATIONS\n")
            f.write("=" * 80 + "\n\n")

            # Find best model for each metric
            best_reward_idx = np.argmax([m['metrics']['average_reward'] for m in self.models])
            best_ach_idx = np.argmax([m['metrics']['average_achievements_per_episode'] for m in self.models])
            best_surv_idx = np.argmax([m['metrics']['average_survival_time'] for m in self.models])
            best_geom_idx = np.argmax([m['metrics']['geometric_mean_achievements'] for m in self.models])

            f.write(f"Best Average Reward: {self.model_names[best_reward_idx]}\n")
            f.write(f"Best Achievement Count: {self.model_names[best_ach_idx]}\n")
            f.write(f"Best Survival Time: {self.model_names[best_surv_idx]}\n")
            f.write(f"Best Overall Score (Geometric Mean): {self.model_names[best_geom_idx]}\n")

            # Comparison type specific insights
            f.write(f"\nComparison Insights ({self.comparison_type}):\n")

            if len(self.models) == 2:
                other_idx = 1 - self.baseline_idx
                other_metrics = self.models[other_idx]['metrics']
                baseline_metrics = self.models[self.baseline_idx]['metrics']

                reward_better = other_metrics['average_reward'] > baseline_metrics['average_reward']
                ach_better = other_metrics['average_achievements_per_episode'] > baseline_metrics['average_achievements_per_episode']
                surv_better = other_metrics['average_survival_time'] > baseline_metrics['average_survival_time']

                wins = sum([reward_better, ach_better, surv_better])

                f.write(f"  {self.model_names[other_idx]} vs {self.model_names[self.baseline_idx]}:\n")
                f.write(f"    Metrics improved: {wins}/3\n")
                f.write(f"    Reward: {'✓ Better' if reward_better else '✗ Worse'}\n")
                f.write(f"    Achievements: {'✓ Better' if ach_better else '✗ Worse'}\n")
                f.write(f"    Survival: {'✓ Better' if surv_better else '✗ Worse'}\n")

            f.write("\n" + "=" * 80 + "\n")
            f.write("END OF REPORT\n")
            f.write("=" * 80 + "\n")

        print(f"  ✓ Saved: {report_path.name}")


def main():
    parser = argparse.ArgumentParser(description='Compare DQN model performances')
    parser.add_argument('--model_paths', nargs='+', required=True,
                       help='Paths to model result JSON files (first is baseline)')
    parser.add_argument('--model_names', nargs='+', default=None,
                       help='Names for each model (optional)')
    parser.add_argument('--output_dir', type=str, default='./comparison_results/',
                       help='Directory to save comparison results')
    parser.add_argument('--comparison_type', type=str, default='custom',
                       choices=['full', 'baseline_vs_reward', 'baseline_vs_preprocessed', 'custom'],
                       help='Type of comparison to perform')

    args = parser.parse_args()

    if args.model_names and len(args.model_names) != len(args.model_paths):
        print("Error: Number of model names must match number of model paths")
        return

    # Set default names based on comparison type if not provided
    if not args.model_names:
        if args.comparison_type == 'baseline_vs_reward':
            args.model_names = ['Baseline', 'Reward Shaped']
        elif args.comparison_type == 'baseline_vs_preprocessed':
            args.model_names = ['Baseline', 'Preprocessed']
        elif args.comparison_type == 'full':
            args.model_names = ['Baseline', 'Reward Shaped', 'Preprocessed + Reward Shaped']

    # Create comparison object
    comparison = ModelComparison(
        model_paths=args.model_paths,
        model_names=args.model_names,
        output_dir=args.output_dir,
        comparison_type=args.comparison_type
    )

    # Generate full comparison
    comparison.generate_full_comparison()


if __name__ == "__main__":
    main()
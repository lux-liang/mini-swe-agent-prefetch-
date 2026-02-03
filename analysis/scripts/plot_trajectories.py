"""
Trajectory Visualization for mini-swe-agent
Generates publication-quality figures for trajectory analysis.
"""

import os
import sys
import json
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path
from collections import Counter

try:
    from scipy import stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("Warning: scipy not available, some statistics will be skipped")

# Add scripts dir to path
sys.path.insert(0, os.path.dirname(__file__))
from parse_traj import parse_run, parse_trajectory

# Configure matplotlib for publication quality
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 16,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})


def plot_step_distribution(instances_df: pd.DataFrame, output_dir: str, tag: str):
    """Plot distribution of total steps per instance."""
    fig, ax = plt.subplots(figsize=(10, 5))

    steps = instances_df['total_steps'].values
    ax.hist(steps, bins=20, edgecolor='black', alpha=0.7, color='steelblue')
    ax.axvline(np.mean(steps), color='red', linestyle='--', label=f'Mean: {np.mean(steps):.1f}')
    ax.axvline(np.median(steps), color='orange', linestyle='--', label=f'Median: {np.median(steps):.1f}')

    ax.set_xlabel('Total Steps')
    ax.set_ylabel('Frequency')
    ax.set_title(f'Distribution of Agent Steps ({tag})')
    ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'fig1_step_distribution_{tag}.png'))
    plt.savefig(os.path.join(output_dir, f'fig1_step_distribution_{tag}.pdf'))
    plt.close()


def plot_command_breakdown(instances_df: pd.DataFrame, output_dir: str, tag: str):
    """Plot breakdown of command types."""
    cmd_cols = [c for c in instances_df.columns if c.endswith('_count') and c != 'api_calls']
    cmd_sums = {c.replace('_count', ''): instances_df[c].sum() for c in cmd_cols}
    cmd_sums = {k: v for k, v in sorted(cmd_sums.items(), key=lambda x: -x[1]) if v > 0}

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = plt.cm.Set3(np.linspace(0, 1, len(cmd_sums)))
    bars = ax.bar(range(len(cmd_sums)), list(cmd_sums.values()), color=colors, edgecolor='black')

    ax.set_xticks(range(len(cmd_sums)))
    ax.set_xticklabels(list(cmd_sums.keys()), rotation=45, ha='right')
    ax.set_xlabel('Command Type')
    ax.set_ylabel('Total Count')
    ax.set_title(f'Command Type Distribution ({tag})')

    # Add value labels on bars
    for bar, val in zip(bars, cmd_sums.values()):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                str(val), ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'fig2_command_breakdown_{tag}.png'))
    plt.savefig(os.path.join(output_dir, f'fig2_command_breakdown_{tag}.pdf'))
    plt.close()


def plot_cost_vs_steps(instances_df: pd.DataFrame, output_dir: str, tag: str):
    """Plot cost vs total steps scatter."""
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.scatter(instances_df['total_steps'], instances_df['instance_cost'],
               alpha=0.7, s=80, c='steelblue', edgecolors='black')

    # Add regression line
    x = instances_df['total_steps'].values
    y = instances_df['instance_cost'].values
    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)
    ax.plot(sorted(x), p(sorted(x)), "r--", alpha=0.8, label=f'Linear fit')

    # Correlation
    if HAS_SCIPY:
        corr, pval = stats.pearsonr(x, y)
        ax.text(0.05, 0.95, f'r = {corr:.3f}\np = {pval:.3e}',
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    else:
        # Simple correlation without scipy
        corr = np.corrcoef(x, y)[0, 1]
        ax.text(0.05, 0.95, f'r = {corr:.3f}',
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    ax.set_xlabel('Total Steps')
    ax.set_ylabel('Cost ($)')
    ax.set_title(f'Cost vs Steps ({tag})')
    ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'fig3_cost_vs_steps_{tag}.png'))
    plt.savefig(os.path.join(output_dir, f'fig3_cost_vs_steps_{tag}.pdf'))
    plt.close()


def plot_instance_command_heatmap(instances_df: pd.DataFrame, output_dir: str, tag: str):
    """Plot heatmap of command types per instance."""
    cmd_cols = [c for c in instances_df.columns if c.endswith('_count') and c != 'api_calls']
    cmd_cols = [c for c in cmd_cols if instances_df[c].sum() > 0]

    # Sort by total steps
    df_sorted = instances_df.sort_values('total_steps', ascending=True)

    data = df_sorted[cmd_cols].values
    data_log = np.log1p(data)  # Log transform for better visualization

    fig, ax = plt.subplots(figsize=(12, 8))

    im = ax.imshow(data_log, aspect='auto', cmap='YlOrRd')

    ax.set_xticks(range(len(cmd_cols)))
    ax.set_xticklabels([c.replace('_count', '') for c in cmd_cols], rotation=45, ha='right')

    ax.set_yticks(range(len(df_sorted)))
    ax.set_yticklabels(df_sorted['instance_id'].apply(lambda x: x.split('-')[-1]), fontsize=8)

    ax.set_xlabel('Command Type')
    ax.set_ylabel('Instance (sorted by total steps)')
    ax.set_title(f'Command Type Heatmap ({tag})')

    plt.colorbar(im, ax=ax, label='log1p(count)')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'fig4_command_heatmap_{tag}.png'))
    plt.savefig(os.path.join(output_dir, f'fig4_command_heatmap_{tag}.pdf'))
    plt.close()


def plot_trajectory_timeline(steps_df: pd.DataFrame, instances_df: pd.DataFrame,
                             output_dir: str, tag: str, n_samples: int = 5):
    """Plot timeline visualization of trajectories."""
    # Select instances with varying lengths
    sorted_instances = instances_df.sort_values('total_steps')
    n = len(sorted_instances)
    indices = [0, n//4, n//2, 3*n//4, n-1]
    sample_ids = sorted_instances.iloc[indices]['instance_id'].tolist()

    cmd_types = ['read_file', 'search', 'browse', 'edit', 'git', 'python', 'test', 'write_file', 'other']
    colors = {t: plt.cm.Set1(i/len(cmd_types)) for i, t in enumerate(cmd_types)}

    fig, axes = plt.subplots(len(sample_ids), 1, figsize=(14, 2*len(sample_ids)), sharex=False)
    if len(sample_ids) == 1:
        axes = [axes]

    for ax, iid in zip(axes, sample_ids):
        instance_steps = steps_df[steps_df['instance_id'] == iid].sort_values('step_idx')

        for _, step in instance_steps.iterrows():
            cmd_type = step['command_type']
            color = colors.get(cmd_type, 'gray')
            ax.barh(0, 1, left=step['step_idx'], height=0.8, color=color, edgecolor='black', linewidth=0.5)

        ax.set_xlim(-0.5, instance_steps['step_idx'].max() + 1.5)
        ax.set_ylim(-0.5, 0.5)
        ax.set_yticks([])
        ax.set_ylabel(iid.split('-')[-1], rotation=0, ha='right', va='center', fontsize=9)
        ax.set_xlabel('Step')

    # Add legend to last axis
    handles = [plt.Rectangle((0,0),1,1, color=colors[t]) for t in cmd_types]
    axes[-1].legend(handles, cmd_types, loc='upper center', bbox_to_anchor=(0.5, -0.3),
                    ncol=5, fontsize=8)

    fig.suptitle(f'Trajectory Timeline ({tag})', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'fig5_trajectory_timeline_{tag}.png'))
    plt.savefig(os.path.join(output_dir, f'fig5_trajectory_timeline_{tag}.pdf'))
    plt.close()


def plot_temporal_command_pattern(steps_df: pd.DataFrame, output_dir: str, tag: str, n_bins: int = 10):
    """Plot average command type distribution across normalized trajectory time."""
    cmd_types = ['read_file', 'search', 'browse', 'edit', 'python', 'test', 'other']

    # Normalize step position to [0, 1] for each instance
    steps_df = steps_df.copy()
    for iid in steps_df['instance_id'].unique():
        mask = steps_df['instance_id'] == iid
        max_step = steps_df.loc[mask, 'step_idx'].max()
        if max_step > 0:
            steps_df.loc[mask, 'norm_pos'] = steps_df.loc[mask, 'step_idx'] / max_step
        else:
            steps_df.loc[mask, 'norm_pos'] = 0

    # Bin the normalized positions
    steps_df['bin'] = pd.cut(steps_df['norm_pos'], bins=n_bins, labels=range(n_bins))

    # Count command types per bin
    bin_counts = steps_df.groupby(['bin', 'command_type']).size().unstack(fill_value=0)

    # Normalize to proportions
    bin_props = bin_counts.div(bin_counts.sum(axis=1), axis=0)

    fig, ax = plt.subplots(figsize=(12, 6))

    colors = plt.cm.Set2(np.linspace(0, 1, len(cmd_types)))
    bottom = np.zeros(n_bins)

    for i, cmd_type in enumerate(cmd_types):
        if cmd_type in bin_props.columns:
            values = bin_props[cmd_type].values
            ax.bar(range(n_bins), values, bottom=bottom, label=cmd_type, color=colors[i])
            bottom += values

    ax.set_xticks(range(n_bins))
    ax.set_xticklabels([f'{i/n_bins:.1f}-{(i+1)/n_bins:.1f}' for i in range(n_bins)], rotation=45)
    ax.set_xlabel('Normalized Trajectory Position')
    ax.set_ylabel('Proportion')
    ax.set_title(f'Command Type Distribution Over Trajectory ({tag})')
    ax.legend(loc='upper right', bbox_to_anchor=(1.15, 1))

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'fig6_temporal_pattern_{tag}.png'))
    plt.savefig(os.path.join(output_dir, f'fig6_temporal_pattern_{tag}.pdf'))
    plt.close()


def plot_api_calls_distribution(instances_df: pd.DataFrame, output_dir: str, tag: str):
    """Plot distribution of API calls."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # API calls histogram
    axes[0].hist(instances_df['api_calls'], bins=15, edgecolor='black', alpha=0.7, color='coral')
    axes[0].axvline(instances_df['api_calls'].mean(), color='red', linestyle='--',
                    label=f'Mean: {instances_df["api_calls"].mean():.1f}')
    axes[0].set_xlabel('API Calls')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Distribution of API Calls')
    axes[0].legend()

    # Cost histogram
    axes[1].hist(instances_df['instance_cost'], bins=15, edgecolor='black', alpha=0.7, color='seagreen')
    axes[1].axvline(instances_df['instance_cost'].mean(), color='red', linestyle='--',
                    label=f'Mean: ${instances_df["instance_cost"].mean():.3f}')
    axes[1].set_xlabel('Cost ($)')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Distribution of Instance Cost')
    axes[1].legend()

    fig.suptitle(f'API Usage Statistics ({tag})', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'fig7_api_usage_{tag}.png'))
    plt.savefig(os.path.join(output_dir, f'fig7_api_usage_{tag}.pdf'))
    plt.close()


def plot_search_vs_edit_ratio(instances_df: pd.DataFrame, output_dir: str, tag: str):
    """Plot ratio of search/read operations vs edit operations."""
    instances_df = instances_df.copy()

    # Calculate exploration vs exploitation
    instances_df['exploration'] = instances_df['read_file_count'] + instances_df['search_count'] + instances_df['browse_count']
    instances_df['exploitation'] = instances_df['edit_count'] + instances_df['write_file_count']

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.scatter(instances_df['exploration'], instances_df['exploitation'],
               s=instances_df['total_steps']*5, alpha=0.6, c='purple', edgecolors='black')

    ax.set_xlabel('Exploration (read + search + browse)')
    ax.set_ylabel('Exploitation (edit + write)')
    ax.set_title(f'Exploration vs Exploitation ({tag})\n(bubble size = total steps)')

    # Add diagonal line
    max_val = max(instances_df['exploration'].max(), instances_df['exploitation'].max())
    ax.plot([0, max_val], [0, max_val], 'k--', alpha=0.3, label='1:1 ratio')
    ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'fig8_explore_vs_exploit_{tag}.png'))
    plt.savefig(os.path.join(output_dir, f'fig8_explore_vs_exploit_{tag}.pdf'))
    plt.close()


def generate_all_figures(run_dir: str, output_dir: str, tag: str):
    """Generate all figures for a run."""
    os.makedirs(output_dir, exist_ok=True)

    print(f"Parsing trajectories from {run_dir}...")
    steps_df, instances_df = parse_run(run_dir)

    print(f"Generating figures for {len(instances_df)} instances...")

    plot_step_distribution(instances_df, output_dir, tag)
    print("  - fig1: step distribution")

    plot_command_breakdown(instances_df, output_dir, tag)
    print("  - fig2: command breakdown")

    plot_cost_vs_steps(instances_df, output_dir, tag)
    print("  - fig3: cost vs steps")

    plot_instance_command_heatmap(instances_df, output_dir, tag)
    print("  - fig4: command heatmap")

    plot_trajectory_timeline(steps_df, instances_df, output_dir, tag)
    print("  - fig5: trajectory timeline")

    plot_temporal_command_pattern(steps_df, output_dir, tag)
    print("  - fig6: temporal pattern")

    plot_api_calls_distribution(instances_df, output_dir, tag)
    print("  - fig7: API usage")

    plot_search_vs_edit_ratio(instances_df, output_dir, tag)
    print("  - fig8: explore vs exploit")

    # Save dataframes
    steps_df.to_csv(os.path.join(output_dir, f'steps_{tag}.csv'), index=False)
    instances_df.to_csv(os.path.join(output_dir, f'instances_{tag}.csv'), index=False)
    print(f"  - saved CSVs to {output_dir}")

    return steps_df, instances_df


if __name__ == "__main__":
    # Default run
    run_dir = "analysis/runs/20260128_193355"
    output_dir = "analysis/figures"
    tag = "verified_20"

    generate_all_figures(run_dir, output_dir, tag)
    print(f"\nAll figures saved to {output_dir}/")

"""
Ablation Analysis for Prefetch Experiments
Compares baseline (V0) vs prefetch versions on key metrics.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))
from parse_traj import parse_run, calculate_paging_metrics, calculate_explore_metrics

# Configure matplotlib
plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})


def analyze_baseline_metrics(run_dir: str, output_dir: str, tag: str = "baseline"):
    """Analyze baseline run and extract all ablation metrics."""
    os.makedirs(output_dir, exist_ok=True)

    print(f"Parsing trajectories from {run_dir}...")
    steps_df, instances_df = parse_run(run_dir)

    print(f"Analyzed {len(instances_df)} instances, {len(steps_df)} steps")

    # Key metrics summary
    metrics = {
        'tag': tag,
        'n_instances': len(instances_df),
        'total_steps': int(instances_df['total_steps'].sum()),
        'total_cost': float(instances_df['instance_cost'].sum()),
        'avg_steps': float(instances_df['total_steps'].mean()),
        'std_steps': float(instances_df['total_steps'].std()),
        'avg_cost': float(instances_df['instance_cost'].mean()),
        'avg_manual_paging_ops': float(instances_df['manual_paging_ops'].mean()),
        'total_manual_paging_ops': int(instances_df['manual_paging_ops'].sum()),
        'avg_read_duplication_rate': float(instances_df['read_duplication_rate'].mean()),
        'avg_front_25_explore_ratio': float(instances_df['front_25_explore_ratio'].mean()),
        'avg_front_50_explore_ratio': float(instances_df['front_50_explore_ratio'].mean()),
        'avg_explore_chain_len': float(instances_df['explore_chain_len'].mean()),
        'avg_explore_ratio': float(instances_df['explore_ratio'].mean()),
        'avg_total_lines_paged': float(instances_df['total_lines_paged'].mean()),
        'avg_overlap_ratio': float(instances_df['overlap_ratio'].mean()),
    }

    # Save metrics
    metrics_path = os.path.join(output_dir, f'metrics_{tag}.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)

    # Save detailed CSV
    instances_df.to_csv(os.path.join(output_dir, f'instances_{tag}.csv'), index=False)
    steps_df.to_csv(os.path.join(output_dir, f'steps_{tag}.csv'), index=False)

    return metrics, instances_df, steps_df


def print_baseline_report(metrics: dict, instances_df: pd.DataFrame):
    """Print formatted baseline analysis report."""
    print("\n" + "="*70)
    print("BASELINE (V0) ANALYSIS REPORT")
    print("="*70)

    print(f"\n📊 Overview")
    print(f"  Instances: {metrics['n_instances']}")
    print(f"  Total Steps: {metrics['total_steps']}")
    print(f"  Total Cost: ${metrics['total_cost']:.2f}")
    print(f"  Avg Steps/Instance: {metrics['avg_steps']:.1f} ± {metrics['std_steps']:.1f}")
    print(f"  Avg Cost/Instance: ${metrics['avg_cost']:.3f}")

    print(f"\n📄 Manual Paging (nl|sed) Metrics")
    print(f"  Total Paging Ops: {metrics['total_manual_paging_ops']}")
    print(f"  Avg Paging Ops/Instance: {metrics['avg_manual_paging_ops']:.1f}")
    print(f"  Avg Lines Paged/Instance: {metrics['avg_total_lines_paged']:.0f}")
    print(f"  Read Duplication Rate: {metrics['avg_read_duplication_rate']:.1%}")
    print(f"  Line Overlap Ratio: {metrics['avg_overlap_ratio']:.1%}")

    print(f"\n🔍 Exploration Metrics")
    print(f"  Front 25% Explore Ratio: {metrics['avg_front_25_explore_ratio']:.1%}")
    print(f"  Front 50% Explore Ratio: {metrics['avg_front_50_explore_ratio']:.1%}")
    print(f"  Overall Explore Ratio: {metrics['avg_explore_ratio']:.1%}")
    print(f"  Max Explore Chain Length: {metrics['avg_explore_chain_len']:.1f}")

    # Per-instance breakdown
    print(f"\n📋 Per-Instance Paging Breakdown")
    print("-"*70)
    cols = ['instance_id', 'total_steps', 'manual_paging_ops', 'total_lines_paged', 'front_25_explore_ratio']
    print(instances_df[cols].to_string(index=False))

    print("\n" + "="*70)


def plot_baseline_metrics(instances_df: pd.DataFrame, output_dir: str, tag: str):
    """Generate baseline analysis plots."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # 1. Steps distribution
    ax = axes[0, 0]
    ax.hist(instances_df['total_steps'], bins=15, edgecolor='black', alpha=0.7, color='steelblue')
    ax.axvline(instances_df['total_steps'].mean(), color='red', linestyle='--',
               label=f'Mean: {instances_df["total_steps"].mean():.1f}')
    ax.set_xlabel('Total Steps')
    ax.set_ylabel('Frequency')
    ax.set_title('Steps Distribution')
    ax.legend()

    # 2. Manual paging ops distribution
    ax = axes[0, 1]
    ax.hist(instances_df['manual_paging_ops'], bins=15, edgecolor='black', alpha=0.7, color='coral')
    ax.axvline(instances_df['manual_paging_ops'].mean(), color='red', linestyle='--',
               label=f'Mean: {instances_df["manual_paging_ops"].mean():.1f}')
    ax.set_xlabel('Manual Paging Operations (nl|sed)')
    ax.set_ylabel('Frequency')
    ax.set_title('Manual Paging Distribution')
    ax.legend()

    # 3. Front 25% explore ratio
    ax = axes[0, 2]
    ax.hist(instances_df['front_25_explore_ratio'], bins=15, edgecolor='black', alpha=0.7, color='seagreen')
    ax.axvline(instances_df['front_25_explore_ratio'].mean(), color='red', linestyle='--',
               label=f'Mean: {instances_df["front_25_explore_ratio"].mean():.1%}')
    ax.set_xlabel('Front 25% Exploration Ratio')
    ax.set_ylabel('Frequency')
    ax.set_title('Early Exploration Burden')
    ax.legend()

    # 4. Paging ops vs steps scatter
    ax = axes[1, 0]
    ax.scatter(instances_df['total_steps'], instances_df['manual_paging_ops'],
               alpha=0.7, s=80, c='purple', edgecolors='black')
    ax.set_xlabel('Total Steps')
    ax.set_ylabel('Manual Paging Ops')
    ax.set_title('Paging vs Steps')
    # Add correlation
    corr = instances_df['total_steps'].corr(instances_df['manual_paging_ops'])
    ax.text(0.05, 0.95, f'r = {corr:.3f}', transform=ax.transAxes,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # 5. Cost vs steps
    ax = axes[1, 1]
    ax.scatter(instances_df['total_steps'], instances_df['instance_cost'],
               alpha=0.7, s=80, c='steelblue', edgecolors='black')
    ax.set_xlabel('Total Steps')
    ax.set_ylabel('Cost ($)')
    ax.set_title('Cost vs Steps')
    corr = instances_df['total_steps'].corr(instances_df['instance_cost'])
    ax.text(0.05, 0.95, f'r = {corr:.3f}', transform=ax.transAxes,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # 6. Explore chain length distribution
    ax = axes[1, 2]
    ax.hist(instances_df['explore_chain_len'], bins=15, edgecolor='black', alpha=0.7, color='orange')
    ax.axvline(instances_df['explore_chain_len'].mean(), color='red', linestyle='--',
               label=f'Mean: {instances_df["explore_chain_len"].mean():.1f}')
    ax.set_xlabel('Max Consecutive Explore Steps')
    ax.set_ylabel('Frequency')
    ax.set_title('Exploration Chain Length')
    ax.legend()

    plt.suptitle(f'Baseline Analysis: {tag}', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'baseline_analysis_{tag}.png'))
    plt.savefig(os.path.join(output_dir, f'baseline_analysis_{tag}.pdf'))
    plt.close()
    print(f"Saved baseline plots to {output_dir}/baseline_analysis_{tag}.png")


def compare_runs(baseline_metrics: dict, prefetch_metrics: dict, output_dir: str, tag: str):
    """Generate comparison report between baseline and prefetch."""
    print("\n" + "="*70)
    print("ABLATION COMPARISON: BASELINE vs PREFETCH")
    print("="*70)

    comparison_metrics = [
        ('avg_steps', 'Avg Steps', True),
        ('avg_cost', 'Avg Cost ($)', True),
        ('avg_manual_paging_ops', 'Avg Paging Ops', True),
        ('avg_read_duplication_rate', 'Read Duplication Rate', True),
        ('avg_front_25_explore_ratio', 'Front 25% Explore Ratio', True),
        ('avg_explore_chain_len', 'Avg Explore Chain Len', True),
    ]

    results = []
    print(f"\n{'Metric':<30} {'Baseline':>12} {'Prefetch':>12} {'Change':>12} {'Status':>10}")
    print("-"*76)

    for key, label, lower_is_better in comparison_metrics:
        b_val = baseline_metrics.get(key, 0)
        p_val = prefetch_metrics.get(key, 0)

        if b_val != 0:
            change_pct = (p_val - b_val) / b_val * 100
        else:
            change_pct = 0

        improved = (change_pct < 0) if lower_is_better else (change_pct > 0)
        status = "✓ Better" if improved else "✗ Worse" if change_pct != 0 else "—"

        # Format values
        if 'ratio' in key.lower() or 'rate' in key.lower():
            b_str = f"{b_val:.1%}"
            p_str = f"{p_val:.1%}"
        elif 'cost' in key.lower():
            b_str = f"${b_val:.3f}"
            p_str = f"${p_val:.3f}"
        else:
            b_str = f"{b_val:.2f}"
            p_str = f"{p_val:.2f}"

        print(f"{label:<30} {b_str:>12} {p_str:>12} {change_pct:>+11.1f}% {status:>10}")

        results.append({
            'metric': label,
            'baseline': b_val,
            'prefetch': p_val,
            'change_pct': change_pct,
            'improved': improved,
        })

    print("-"*76)

    # Save comparison
    comparison_df = pd.DataFrame(results)
    comparison_df.to_csv(os.path.join(output_dir, f'comparison_{tag}.csv'), index=False)

    return comparison_df


def plot_ablation_comparison(baseline_df: pd.DataFrame, prefetch_df: pd.DataFrame,
                             output_dir: str, tag: str):
    """Generate comparison plots for ablation study."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Fig A: Steps Distribution Comparison
    ax = axes[0, 0]
    ax.hist(baseline_df['total_steps'], bins=15, alpha=0.6, label='Baseline (V0)', color='gray')
    ax.hist(prefetch_df['total_steps'], bins=15, alpha=0.6, label='Prefetch (V2)', color='steelblue')
    ax.axvline(baseline_df['total_steps'].mean(), color='gray', linestyle='--', linewidth=2)
    ax.axvline(prefetch_df['total_steps'].mean(), color='steelblue', linestyle='--', linewidth=2)
    ax.set_xlabel('Total Steps')
    ax.set_ylabel('Frequency')
    ax.set_title('Steps Distribution: Baseline vs Prefetch')
    ax.legend()

    # Fig B: Manual Paging Ops Comparison
    ax = axes[0, 1]
    data = [baseline_df['manual_paging_ops'], prefetch_df['manual_paging_ops']]
    bp = ax.boxplot(data, labels=['Baseline (V0)', 'Prefetch (V2)'], patch_artist=True)
    bp['boxes'][0].set_facecolor('gray')
    bp['boxes'][1].set_facecolor('steelblue')
    ax.set_ylabel('Manual Paging Operations (nl|sed)')
    ax.set_title('Manual Paging: Baseline vs Prefetch')

    # Calculate improvement
    baseline_mean = baseline_df['manual_paging_ops'].mean()
    prefetch_mean = prefetch_df['manual_paging_ops'].mean()
    if baseline_mean > 0:
        improvement = (baseline_mean - prefetch_mean) / baseline_mean * 100
        color = 'green' if improvement > 0 else 'red'
        ax.text(0.95, 0.95, f'{"↓" if improvement > 0 else "↑"}{abs(improvement):.1f}%',
                transform=ax.transAxes, ha='right', va='top', fontsize=14, color=color,
                fontweight='bold')

    # Fig C: Front 25% Explore Ratio
    ax = axes[1, 0]
    data = [baseline_df['front_25_explore_ratio'], prefetch_df['front_25_explore_ratio']]
    bp = ax.boxplot(data, labels=['Baseline (V0)', 'Prefetch (V2)'], patch_artist=True)
    bp['boxes'][0].set_facecolor('gray')
    bp['boxes'][1].set_facecolor('steelblue')
    ax.set_ylabel('Front 25% Exploration Ratio')
    ax.set_title('Early Exploration Burden')

    # Fig D: Cost vs Steps with Regression
    ax = axes[1, 1]
    ax.scatter(baseline_df['total_steps'], baseline_df['instance_cost'],
               alpha=0.6, label='Baseline (V0)', color='gray', s=60, edgecolors='black')
    ax.scatter(prefetch_df['total_steps'], prefetch_df['instance_cost'],
               alpha=0.6, label='Prefetch (V2)', color='steelblue', s=60, edgecolors='black')
    ax.set_xlabel('Total Steps')
    ax.set_ylabel('Cost ($)')
    ax.set_title('Cost vs Steps')
    ax.legend()

    plt.suptitle(f'Ablation Study: {tag}', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'ablation_comparison_{tag}.png'))
    plt.savefig(os.path.join(output_dir, f'ablation_comparison_{tag}.pdf'))
    plt.close()
    print(f"Saved comparison plots to {output_dir}/ablation_comparison_{tag}.png")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Ablation analysis for prefetch experiments")
    parser.add_argument("--baseline-dir", type=str, required=True, help="Baseline run directory")
    parser.add_argument("--prefetch-dir", type=str, default=None, help="Prefetch run directory (optional)")
    parser.add_argument("--output-dir", type=str, default="analysis/ablation", help="Output directory")
    parser.add_argument("--tag", type=str, default="verified_20", help="Experiment tag")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Analyze baseline
    baseline_metrics, baseline_df, baseline_steps = analyze_baseline_metrics(
        args.baseline_dir, args.output_dir, f"baseline_{args.tag}"
    )
    print_baseline_report(baseline_metrics, baseline_df)
    plot_baseline_metrics(baseline_df, args.output_dir, f"baseline_{args.tag}")

    # If prefetch dir provided, compare
    if args.prefetch_dir:
        prefetch_metrics, prefetch_df, prefetch_steps = analyze_baseline_metrics(
            args.prefetch_dir, args.output_dir, f"prefetch_{args.tag}"
        )
        print_baseline_report(prefetch_metrics, prefetch_df)

        compare_runs(baseline_metrics, prefetch_metrics, args.output_dir, args.tag)
        plot_ablation_comparison(baseline_df, prefetch_df, args.output_dir, args.tag)

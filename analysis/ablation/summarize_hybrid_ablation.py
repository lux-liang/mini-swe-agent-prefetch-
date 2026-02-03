#!/usr/bin/env python3
"""
Summarize Hybrid Ablation Experiment Results

Analyzes results from Prefetch V2 ablation experiments across
5 stratified sampling windows (Sequential, Boundary, Anchor).

Outputs:
- hybrid_ablation_table.csv: Per-group×window metrics
- HYBRID_ABLATION_REPORT.md: Full analysis report
"""

import json
import glob
import os
import sys
import re
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional
import numpy as np

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    print("Warning: pandas not available, using basic output")

# Add scripts path for parse_traj
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
try:
    from parse_traj import parse_run, parse_trajectory
    HAS_PARSE = True
except ImportError:
    HAS_PARSE = False
    print("Warning: parse_traj not available")


@dataclass
class RunMetrics:
    """Metrics for a single run (group × window)."""
    group: str
    slice_str: str
    window_type: str
    n_instances: int = 0
    n_success: int = 0
    success_rate: float = 0.0
    avg_steps: float = 0.0
    median_steps: float = 0.0
    p90_steps: float = 0.0
    max_steps: int = 0
    avg_cost: float = 0.0
    total_cost: float = 0.0
    p90_cost: float = 0.0
    max_cost: float = 0.0
    pager_ops: float = 0.0  # avg manual_paging_ops
    read_dup_rate: float = 0.0  # avg read_duplication_rate


def get_window_type(slice_str: str, slices_data: dict) -> str:
    """Determine window type from slice string."""
    for w in slices_data.get("windows", []):
        if w.get("slice_str") == slice_str:
            return w.get("window_type", "unknown")
    return "unknown"


def find_run_directories(results_dir: str, groups: list[str]) -> dict:
    """Find all run directories matching the naming pattern."""
    runs = defaultdict(dict)

    for group in groups:
        pattern = os.path.join(results_dir, f"{group}_verified_*")
        dirs = glob.glob(pattern)

        for d in dirs:
            if not os.path.isdir(d):
                continue
            # Extract slice from directory name: {group}_verified_{start}_{end}
            basename = os.path.basename(d)
            match = re.search(r'_verified_(\d+)_(\d+)$', basename)
            if match:
                slice_str = f"{match.group(1)}:{match.group(2)}"
                runs[group][slice_str] = d

    return runs


def parse_run_directory(run_dir: str) -> dict:
    """Parse metrics from a run directory."""
    metrics = {
        "n_instances": 0,
        "n_success": 0,
        "steps": [],
        "costs": [],
        "pager_ops": [],
        "read_dup_rates": [],
        "exit_statuses": [],
    }

    # Try to find trajectory files
    traj_pattern = os.path.join(run_dir, "*", "*.traj.json")
    traj_files = glob.glob(traj_pattern)

    # Also check Modal volume structure
    if not traj_files:
        traj_pattern = os.path.join(run_dir, "**", "*.traj.json")
        traj_files = glob.glob(traj_pattern, recursive=True)

    for traj_file in traj_files:
        try:
            with open(traj_file, 'r') as f:
                traj = json.load(f)

            info = traj.get("info", {})
            messages = traj.get("messages", [])

            metrics["n_instances"] += 1

            # Exit status / success
            exit_status = info.get("exit_status", "unknown")
            metrics["exit_statuses"].append(exit_status)
            if exit_status == "Submitted":
                metrics["n_success"] += 1

            # Steps (count bash blocks in assistant messages)
            steps = sum(1 for m in messages
                       if m.get("role") == "assistant"
                       and "```bash" in m.get("content", ""))
            metrics["steps"].append(steps)

            # Cost
            model_stats = info.get("model_stats", {})
            cost = model_stats.get("instance_cost", 0)
            metrics["costs"].append(cost)

            # Manual paging ops (count nl|sed patterns)
            pager_ops = 0
            for m in messages:
                if m.get("role") == "assistant":
                    content = m.get("content", "")
                    pager_ops += len(re.findall(r'nl\s+.*\|\s*sed\s+-n', content))
            metrics["pager_ops"].append(pager_ops)

            # Read duplication (simplified: count repeated file reads)
            file_reads = []
            for m in messages:
                if m.get("role") == "assistant":
                    content = m.get("content", "")
                    # Extract files from cat/head/tail/sed commands
                    files = re.findall(r'(?:cat|head|tail|sed\s+-n[^|]+)\s+([^\s|>]+\.py)', content)
                    file_reads.extend(files)

            if file_reads:
                unique_files = len(set(file_reads))
                dup_rate = 1 - (unique_files / len(file_reads))
            else:
                dup_rate = 0
            metrics["read_dup_rates"].append(dup_rate)

        except Exception as e:
            print(f"  Warning: Failed to parse {traj_file}: {e}")

    return metrics


def compute_run_metrics(group: str, slice_str: str, window_type: str,
                        raw_metrics: dict) -> RunMetrics:
    """Compute summary metrics from raw data."""
    n = raw_metrics["n_instances"]
    if n == 0:
        return RunMetrics(group=group, slice_str=slice_str, window_type=window_type)

    steps = np.array(raw_metrics["steps"]) if raw_metrics["steps"] else np.array([0])
    costs = np.array(raw_metrics["costs"]) if raw_metrics["costs"] else np.array([0])
    pager = np.array(raw_metrics["pager_ops"]) if raw_metrics["pager_ops"] else np.array([0])
    dup_rates = np.array(raw_metrics["read_dup_rates"]) if raw_metrics["read_dup_rates"] else np.array([0])

    return RunMetrics(
        group=group,
        slice_str=slice_str,
        window_type=window_type,
        n_instances=n,
        n_success=raw_metrics["n_success"],
        success_rate=raw_metrics["n_success"] / n if n > 0 else 0,
        avg_steps=float(np.mean(steps)),
        median_steps=float(np.median(steps)),
        p90_steps=float(np.percentile(steps, 90)) if len(steps) > 1 else float(steps[0]),
        max_steps=int(np.max(steps)),
        avg_cost=float(np.mean(costs)),
        total_cost=float(np.sum(costs)),
        p90_cost=float(np.percentile(costs, 90)) if len(costs) > 1 else float(costs[0]),
        max_cost=float(np.max(costs)),
        pager_ops=float(np.mean(pager)),
        read_dup_rate=float(np.mean(dup_rates)),
    )


def aggregate_metrics(all_metrics: list[RunMetrics], group: str) -> dict:
    """Aggregate metrics across windows for a group (weighted by N)."""
    group_metrics = [m for m in all_metrics if m.group == group]
    if not group_metrics:
        return {}

    total_n = sum(m.n_instances for m in group_metrics)
    if total_n == 0:
        return {}

    # Weighted averages
    agg = {
        "group": group,
        "n_instances": total_n,
        "n_success": sum(m.n_success for m in group_metrics),
        "success_rate": sum(m.n_success for m in group_metrics) / total_n,
        "avg_steps": sum(m.avg_steps * m.n_instances for m in group_metrics) / total_n,
        "avg_cost": sum(m.avg_cost * m.n_instances for m in group_metrics) / total_n,
        "total_cost": sum(m.total_cost for m in group_metrics),
        "pager_ops": sum(m.pager_ops * m.n_instances for m in group_metrics) / total_n,
        "read_dup_rate": sum(m.read_dup_rate * m.n_instances for m in group_metrics) / total_n,
        "max_steps": max(m.max_steps for m in group_metrics),
        "max_cost": max(m.max_cost for m in group_metrics),
    }

    return agg


def generate_comparison(anchor_agg: dict, compare_agg: dict) -> dict:
    """Generate comparison between anchor and another group."""
    if not anchor_agg or not compare_agg:
        return {}

    def pct_change(old, new):
        if old == 0:
            return 0
        return (new - old) / old * 100

    return {
        "group": compare_agg["group"],
        "vs_anchor": "prefetch_v2",
        "steps_change_pct": pct_change(anchor_agg["avg_steps"], compare_agg["avg_steps"]),
        "cost_change_pct": pct_change(anchor_agg["avg_cost"], compare_agg["avg_cost"]),
        "success_rate_change_pct": pct_change(anchor_agg["success_rate"], compare_agg["success_rate"]),
        "pager_ops_change_pct": pct_change(anchor_agg["pager_ops"], compare_agg["pager_ops"]),
        "read_dup_rate_change_pct": pct_change(anchor_agg["read_dup_rate"], compare_agg["read_dup_rate"]),
        "is_efficiency_improvement": compare_agg["avg_steps"] < anchor_agg["avg_steps"],
        "is_quality_preserved": compare_agg["success_rate"] >= anchor_agg["success_rate"] * 0.95,
        "is_valid_improvement": (
            compare_agg["avg_steps"] < anchor_agg["avg_steps"] and
            compare_agg["success_rate"] >= anchor_agg["success_rate"] * 0.95
        ),
    }


def generate_report(all_metrics: list[RunMetrics], aggregates: dict,
                    comparisons: list[dict], slices_data: dict,
                    output_dir: str):
    """Generate the markdown report."""
    report = []

    report.append("# Hybrid Ablation Experiment Report")
    report.append("")
    report.append("**Protocol**: Repo-Aware Stratified Sampling")
    report.append(f"**Date**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}" if HAS_PANDAS else "")
    report.append("")

    # Why hybrid sampling
    report.append("## 1. Background: Why Hybrid Sampling?")
    report.append("")
    report.append("SWE-bench Verified is sorted alphabetically by `instance_id` (which starts with repo name),")
    report.append("causing severe repo concentration in contiguous slices:")
    report.append("")
    report.append("```")
    report.append("Index Range    Repo                  Count")
    report.append("─────────────────────────────────────────────")
    report.append("  0-21         astropy/astropy        22")
    report.append(" 22-252        django/django         231  ← 46% of dataset!")
    report.append("253-286        matplotlib/matplotlib  34")
    report.append("...")
    report.append("```")
    report.append("")
    report.append("**Problem**: Pure sequential sampling (e.g., 0:100) would be ~80% django,")
    report.append("making efficiency metrics misleading for cross-repo generalization.")
    report.append("")
    report.append("**Solution**: Repo-Aware Stratified Sampling with 3 window types:")
    report.append("- **Sequential**: Baseline efficiency on dominant repos")
    report.append("- **Boundary**: Cold-start behavior when switching repos")
    report.append("- **Anchor**: Long-tail generalization (rare repos)")
    report.append("")

    # Sampling windows
    report.append("## 2. Sampling Windows")
    report.append("")
    windows = slices_data.get("windows", [])
    report.append("| Type | Slice | Repos | Distribution |")
    report.append("|------|-------|-------|--------------|")
    for w in windows:
        repos = w.get("unique_repo_count", 0)
        dist = w.get("repo_distribution", {})
        dist_str = ", ".join(f"{k.split('/')[-1]}:{v}" for k, v in list(dist.items())[:3])
        report.append(f"| {w['window_type']} | {w['slice_str']} | {repos} | {dist_str} |")
    report.append("")
    report.append(f"**Combined Coverage**: {slices_data.get('combined_unique_repos', 0)} unique repos")
    report.append("")

    # Per-group × window results
    report.append("## 3. Detailed Results")
    report.append("")

    if HAS_PANDAS and all_metrics:
        rows = []
        for m in all_metrics:
            rows.append({
                "Group": m.group,
                "Slice": m.slice_str,
                "Type": m.window_type,
                "N": m.n_instances,
                "Success%": f"{m.success_rate*100:.1f}%",
                "Avg Steps": f"{m.avg_steps:.1f}",
                "P90 Steps": f"{m.p90_steps:.1f}",
                "Avg Cost": f"${m.avg_cost:.3f}",
                "Pager Ops": f"{m.pager_ops:.1f}",
                "Read Dup%": f"{m.read_dup_rate*100:.1f}%",
            })
        df = pd.DataFrame(rows)
        report.append("### 3.1 Per-Window Metrics")
        report.append("")
        report.append(df.to_markdown(index=False))
        report.append("")

    # Aggregate results
    report.append("### 3.2 Aggregate Results (Weighted by N)")
    report.append("")
    report.append("| Group | N | Success% | Avg Steps | Avg Cost | Pager Ops | Read Dup% |")
    report.append("|-------|---|----------|-----------|----------|-----------|-----------|")
    for group, agg in aggregates.items():
        if agg:
            report.append(f"| {group} | {agg['n_instances']} | {agg['success_rate']*100:.1f}% | "
                         f"{agg['avg_steps']:.1f} | ${agg['avg_cost']:.3f} | "
                         f"{agg['pager_ops']:.1f} | {agg['read_dup_rate']*100:.1f}% |")
    report.append("")

    # Comparisons vs anchor
    report.append("## 4. Comparison vs Anchor (prefetch_v2)")
    report.append("")
    report.append("| Group | Steps Δ | Cost Δ | Success Δ | Pager Ops Δ | Valid? |")
    report.append("|-------|---------|--------|-----------|-------------|--------|")
    for comp in comparisons:
        if comp:
            valid = "✓" if comp.get("is_valid_improvement") else "✗"
            report.append(f"| {comp['group']} | {comp['steps_change_pct']:+.1f}% | "
                         f"{comp['cost_change_pct']:+.1f}% | {comp['success_rate_change_pct']:+.1f}% | "
                         f"{comp['pager_ops_change_pct']:+.1f}% | {valid} |")
    report.append("")

    # Quality warnings
    report.append("## 5. Quality Warnings")
    report.append("")
    for comp in comparisons:
        if comp and not comp.get("is_quality_preserved"):
            report.append(f"⚠️ **{comp['group']}**: Success rate dropped by "
                         f"{abs(comp['success_rate_change_pct']):.1f}% - "
                         f"efficiency gains may be FALSE POSITIVE")
            report.append("")
    if all(comp.get("is_quality_preserved", True) for comp in comparisons if comp):
        report.append("✓ All variants maintain success rate within 5% of anchor.")
    report.append("")

    # Key findings
    report.append("## 6. Key Findings")
    report.append("")
    report.append("### Cache Module (prefetch_v2_cache)")
    cache_comp = next((c for c in comparisons if c and c["group"] == "prefetch_v2_cache"), None)
    if cache_comp:
        report.append(f"- Read duplication rate: {cache_comp['read_dup_rate_change_pct']:+.1f}%")
        report.append(f"- Steps: {cache_comp['steps_change_pct']:+.1f}%")
        report.append(f"- Valid improvement: {'Yes' if cache_comp['is_valid_improvement'] else 'No'}")
    else:
        report.append("- Data not available")
    report.append("")

    report.append("### Pager Module (prefetch_v2_pager)")
    pager_comp = next((c for c in comparisons if c and c["group"] == "prefetch_v2_pager"), None)
    if pager_comp:
        report.append(f"- Pager operations: {pager_comp['pager_ops_change_pct']:+.1f}%")
        report.append(f"- Steps: {pager_comp['steps_change_pct']:+.1f}%")
        report.append(f"- Valid improvement: {'Yes' if pager_comp['is_valid_improvement'] else 'No'}")
    else:
        report.append("- Data not available")
    report.append("")

    report.append("### Combined (prefetch_v2_all)")
    all_comp = next((c for c in comparisons if c and c["group"] == "prefetch_v2_all"), None)
    if all_comp:
        report.append(f"- Steps: {all_comp['steps_change_pct']:+.1f}%")
        report.append(f"- Cost: {all_comp['cost_change_pct']:+.1f}%")
        cache_steps = cache_comp['steps_change_pct'] if cache_comp else 0
        pager_steps = pager_comp['steps_change_pct'] if pager_comp else 0
        expected = cache_steps + pager_steps
        synergy = all_comp['steps_change_pct'] - expected if expected != 0 else 0
        report.append(f"- Synergy vs additive: {synergy:+.1f}% (1+1{'>' if synergy > 0 else '<'}2)")
    else:
        report.append("- Data not available")
    report.append("")

    # Missing metrics
    report.append("## 7. Missing Metrics / Data Gaps")
    report.append("")
    report.append("The following metrics could not be computed from current logs:")
    report.append("- `resolve_rate`: Requires SWE-bench evaluation harness")
    report.append("- `cache_hit_rate`: Requires read_cache module logging")
    report.append("- `pager_expansion_count`: Requires pager_policy module logging")
    report.append("")
    report.append("To enable these metrics, add logging to:")
    report.append("- `src/minisweagent/prefetch/scroll_consolidation.py`")
    report.append("- `src/minisweagent/prefetch/__init__.py` (cache/pager modules)")
    report.append("")

    # How to run
    report.append("## 8. Reproduction")
    report.append("")
    report.append("```bash")
    report.append("# Generate sampling windows")
    report.append("uv run python analysis/ablation/select_hybrid_slices.py")
    report.append("")
    report.append("# Run full experiment")
    report.append("./scripts/run_hybrid_ablation.sh run")
    report.append("")
    report.append("# Generate this report")
    report.append("uv run python analysis/ablation/summarize_hybrid_ablation.py")
    report.append("```")
    report.append("")

    # Write report
    report_path = os.path.join(output_dir, "HYBRID_ABLATION_REPORT.md")
    with open(report_path, "w") as f:
        f.write("\n".join(report))

    print(f"Report saved to: {report_path}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Summarize hybrid ablation results")
    parser.add_argument("--results-dir", default="results",
                       help="Results directory")
    parser.add_argument("--slices-json", default="analysis/ablation/slices_selected.json",
                       help="Slices configuration JSON")
    parser.add_argument("--output-dir", default="analysis/ablation",
                       help="Output directory")
    args = parser.parse_args()

    print("=" * 60)
    print("Hybrid Ablation Summary")
    print("=" * 60)

    # Load slices configuration
    if os.path.exists(args.slices_json):
        with open(args.slices_json) as f:
            slices_data = json.load(f)
        print(f"Loaded slices from: {args.slices_json}")
        print(f"Slices: {slices_data.get('slices', [])}")
    else:
        print(f"Warning: Slices file not found: {args.slices_json}")
        slices_data = {"slices": [], "windows": []}

    # Groups to analyze
    groups = ["prefetch_v2", "prefetch_v2_cache", "prefetch_v2_pager", "prefetch_v2_all"]

    # Find run directories
    print(f"\nSearching for runs in: {args.results_dir}")
    runs = find_run_directories(args.results_dir, groups)

    for group, group_runs in runs.items():
        print(f"  {group}: {len(group_runs)} runs")

    # Parse all runs
    all_metrics = []
    for group in groups:
        group_runs = runs.get(group, {})
        for slice_str, run_dir in group_runs.items():
            print(f"\nParsing: {group} / {slice_str}")
            print(f"  Directory: {run_dir}")

            raw = parse_run_directory(run_dir)
            window_type = get_window_type(slice_str, slices_data)
            metrics = compute_run_metrics(group, slice_str, window_type, raw)
            all_metrics.append(metrics)

            print(f"  N={metrics.n_instances}, success={metrics.success_rate:.1%}, "
                  f"steps={metrics.avg_steps:.1f}, cost=${metrics.avg_cost:.3f}")

    # Aggregate per group
    aggregates = {}
    for group in groups:
        aggregates[group] = aggregate_metrics(all_metrics, group)

    # Comparisons vs anchor
    anchor_agg = aggregates.get("prefetch_v2", {})
    comparisons = []
    for group in ["prefetch_v2_cache", "prefetch_v2_pager", "prefetch_v2_all"]:
        comp = generate_comparison(anchor_agg, aggregates.get(group, {}))
        comparisons.append(comp)

    # Save CSV
    os.makedirs(args.output_dir, exist_ok=True)
    if HAS_PANDAS and all_metrics:
        rows = [vars(m) for m in all_metrics]
        df = pd.DataFrame(rows)
        csv_path = os.path.join(args.output_dir, "hybrid_ablation_table.csv")
        df.to_csv(csv_path, index=False)
        print(f"\nSaved metrics to: {csv_path}")

    # Generate report
    generate_report(all_metrics, aggregates, comparisons, slices_data, args.output_dir)

    print("\nDone!")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Hybrid Ablation Analysis for LSR-Engine v2 Integration.

Provides:
- Paired comparison (same instance_id across groups)
- Paging metrics parsing (nl|sed patterns)
- Comprehensive metrics: steps, cost, paging ops, unique lines, etc.
- Visualization plots

Usage:
    uv run python analysis/ablation/summarize_hybrid_ablation.py \
        --results-dir results/ \
        --output-dir analysis/ablation \
        --plots
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Import LSR parser for consistent paging detection
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

try:
    from minisweagent.prefetch.lsr_engine_v2 import CommandPagerParser
except ImportError:
    # Fallback: inline parser if module not available
    class CommandPagerParser:
        _PATTERN = re.compile(
            r"""nl\s+(?:-\w+\s+)*"""
            r"""(?P<file>[^\s|]+)\s*"""
            r"""\|\s*sed\s+-n\s*"""
            r"""['"]?"""
            r"""(?P<start>\d+)"""
            r""","""
            r"""(?P<end>\d+)"""
            r"""p"""
            r"""['"]?""",
            re.VERBOSE,
        )

        @classmethod
        def parse(cls, cmd: str):
            m = cls._PATTERN.search(cmd)
            if not m:
                return None
            return {
                "file_path": m.group("file"),
                "start_line": int(m.group("start")),
                "end_line": int(m.group("end")),
            }


class HybridAblationAnalyzer:
    """Analyzer for LSR-Engine v2 ablation experiments."""

    def __init__(self, results_dir: Path):
        self.results_dir = results_dir
        self.trajectories: Dict[Tuple[str, str], Dict[str, Any]] = {}
        self.pager_parser = CommandPagerParser()

    def load_all_trajectories(self) -> int:
        """Load all trajectory files. Returns count loaded."""
        loaded = 0

        # Support multiple directory structures
        patterns = [
            "**/*.traj.json",
            "**/trajectories/*.json",
        ]

        for pattern in patterns:
            for traj_file in self.results_dir.rglob(pattern):
                try:
                    with open(traj_file) as f:
                        data = json.load(f)

                    # Extract group and instance_id from path
                    group, instance_id = self._extract_identifiers(traj_file, data)

                    if group and instance_id:
                        key = (group, instance_id)
                        self.trajectories[key] = data
                        loaded += 1

                except (json.JSONDecodeError, KeyError) as e:
                    print(f"Warning: Failed to load {traj_file}: {e}")

        print(f"Loaded {loaded} trajectories")
        return loaded

    def _extract_identifiers(
        self, path: Path, data: Dict[str, Any]
    ) -> Tuple[Optional[str], Optional[str]]:
        """Extract group name and instance_id from path or data."""
        group = None
        instance_id = None

        # Try path-based extraction
        parts = path.parts
        for p in parts:
            # Group patterns: v0_verified_*, prefetch_v2_*, etc.
            if any(prefix in p for prefix in ["v0_", "baseline_", "prefetch_"]):
                # Extract group name (before _verified or _lite)
                m = re.match(r"(v0|baseline|prefetch_v2(?:_\w+)?)", p)
                if m:
                    group = m.group(1)

            # Instance patterns: django__django-11039
            if "__" in p and not p.endswith(".json"):
                instance_id = p

        # Fallback: check filename
        if not instance_id:
            fname = path.stem
            if "__" in fname:
                instance_id = fname.replace(".traj", "")

        # Fallback: check data content
        if not instance_id and "info" in data:
            # Try to find instance_id in various locations
            for key in ["instance_id", "task_id"]:
                if key in data.get("info", {}):
                    instance_id = data["info"][key]
                    break

        return group, instance_id

    def parse_paging_from_messages(self, messages: List[Dict]) -> Dict[str, Any]:
        """Extract paging metrics from conversation messages."""
        paging_ops = 0
        total_lines = 0
        spans_by_file: Dict[str, List[Tuple[int, int]]] = defaultdict(list)

        for msg in messages:
            if msg.get("role") != "assistant":
                continue

            content = msg.get("content", "")

            # Extract bash blocks
            bash_blocks = re.findall(r"```bash\s*\n(.*?)\n```", content, re.DOTALL)

            for block in bash_blocks:
                # Also check raw content for nl|sed patterns
                for text in [block, content]:
                    result = self.pager_parser.parse(text)
                    if result:
                        if isinstance(result, dict):
                            fpath = result["file_path"]
                            start = result["start_line"]
                            end = result["end_line"]
                        else:
                            # CodeSpan object
                            fpath = result.file_path
                            start = result.start_line
                            end = result.end_line

                        paging_ops += 1
                        lines = end - start + 1
                        total_lines += lines
                        spans_by_file[fpath].append((start, end))
                        break  # Avoid double-counting

        # Compute unique lines (union per file)
        unique_lines = 0
        for fpath, ranges in spans_by_file.items():
            if not ranges:
                continue
            ranges.sort()
            merged = [ranges[0]]
            for s, e in ranges[1:]:
                prev_s, prev_e = merged[-1]
                if s <= prev_e + 1:
                    merged[-1] = (prev_s, max(prev_e, e))
                else:
                    merged.append((s, e))
            for s, e in merged:
                unique_lines += (e - s + 1)

        read_dup_rate = (
            1.0 - (unique_lines / total_lines) if total_lines > 0 else 0.0
        )

        return {
            "manual_paging_ops": paging_ops,
            "paging_span_total_lines": total_lines,
            "paging_span_unique_lines": unique_lines,
            "read_duplication_rate": read_dup_rate,
        }

    def compute_front_25_explore(self, messages: List[Dict]) -> float:
        """Compute fraction of first 25% steps spent exploring."""
        assistant_msgs = [m for m in messages if m.get("role") == "assistant"]
        n_steps = len(assistant_msgs)
        if n_steps == 0:
            return 0.0

        front_25_count = max(1, n_steps // 4)

        explore_patterns = [
            r"\bgrep\b", r"\bfind\b", r"\bls\b", r"\brg\b", r"\bag\b",
            r"\bwc\b", r"\bhead\b", r"\btail\b",
        ]
        explore_count = 0

        for msg in assistant_msgs[:front_25_count]:
            content = msg.get("content", "")
            if any(re.search(p, content) for p in explore_patterns):
                explore_count += 1

        return explore_count / front_25_count

    def compute_instance_metrics(self, traj: Dict[str, Any]) -> Dict[str, Any]:
        """Compute all metrics for a single trajectory."""
        messages = traj.get("messages", [])
        info = traj.get("info", {})

        # Basic metrics
        n_steps = sum(1 for m in messages if m.get("role") == "assistant")
        cost = info.get("model_stats", {}).get("instance_cost", 0.0)
        exit_status = info.get("exit_status", "Unknown")
        success = exit_status == "Submitted"

        # Paging metrics (parsed from messages)
        paging = self.parse_paging_from_messages(messages)

        # Exploration ratio
        front_25 = self.compute_front_25_explore(messages)

        # Prefetch aggregate metrics (if available from PrefetchAgent)
        prefetch_agg = info.get("prefetch_aggregate", {})

        # Cache metrics (if available)
        cache_hits = prefetch_agg.get("cache_hits", 0)
        cache_misses = prefetch_agg.get("cache_misses", 0)
        cache_hit_rate = (
            cache_hits / (cache_hits + cache_misses)
            if (cache_hits + cache_misses) > 0 else None
        )

        return {
            "n_steps": n_steps,
            "cost": cost,
            "success": success,
            "exit_status": exit_status,
            "manual_paging_ops": paging["manual_paging_ops"],
            "paging_span_total_lines": paging["paging_span_total_lines"],
            "paging_span_unique_lines": paging["paging_span_unique_lines"],
            "read_duplication_rate": paging["read_duplication_rate"],
            "front_25_explore_ratio": front_25,
            "cache_hits": cache_hits,
            "cache_misses": cache_misses,
            "cache_hit_rate": cache_hit_rate,
            # Include prefetch aggregate if present
            **{k: v for k, v in prefetch_agg.items() if k not in ["cache_hits", "cache_misses"]},
        }

    def compute_paired_deltas(
        self,
        anchor_group: str,
        variant_group: str,
    ) -> Dict[str, List[float]]:
        """Compute paired deltas for instances present in both groups."""
        deltas: Dict[str, List[float]] = defaultdict(list)

        anchor_instances = {k[1]: v for k, v in self.trajectories.items() if k[0] == anchor_group}
        variant_instances = {k[1]: v for k, v in self.trajectories.items() if k[0] == variant_group}

        common_ids = set(anchor_instances.keys()) & set(variant_instances.keys())
        print(f"  Paired instances ({anchor_group} vs {variant_group}): {len(common_ids)}")

        for iid in common_ids:
            anchor_m = self.compute_instance_metrics(anchor_instances[iid])
            variant_m = self.compute_instance_metrics(variant_instances[iid])

            # Only compare if both ran (not necessarily both succeeded)
            if anchor_m["n_steps"] > 0 and variant_m["n_steps"] > 0:
                deltas["steps_delta"].append(variant_m["n_steps"] - anchor_m["n_steps"])
                deltas["cost_delta"].append(variant_m["cost"] - anchor_m["cost"])
                deltas["paging_ops_delta"].append(
                    variant_m["manual_paging_ops"] - anchor_m["manual_paging_ops"]
                )
                deltas["unique_lines_delta"].append(
                    variant_m["paging_span_unique_lines"] - anchor_m["paging_span_unique_lines"]
                )
                deltas["read_dup_delta"].append(
                    variant_m["read_duplication_rate"] - anchor_m["read_duplication_rate"]
                )

                # Success comparison
                if anchor_m["success"] and not variant_m["success"]:
                    deltas["regressions"].append(iid)
                elif not anchor_m["success"] and variant_m["success"]:
                    deltas["improvements"].append(iid)

        return dict(deltas)

    def compute_group_summary(self, group: str, slice_filter: Optional[str] = None) -> Dict[str, Any]:
        """Compute summary statistics for a group, optionally filtered by slice."""
        if slice_filter:
            # Filter by slice in group name
            group_trajs = {
                k[1]: v for k, v in self.trajectories.items()
                if k[0] == group and slice_filter in str(k)
            }
        else:
            group_trajs = {k[1]: v for k, v in self.trajectories.items() if k[0] == group}

        if not group_trajs:
            return {"error": f"No data for group {group}"}

        metrics_list = [self.compute_instance_metrics(t) for t in group_trajs.values()]

        all_instances = metrics_list
        successes = [m for m in metrics_list if m["success"]]

        def safe_mean(values: List[float]) -> float:
            return statistics.mean(values) if values else 0.0

        def safe_median(values: List[float]) -> float:
            return statistics.median(values) if values else 0.0

        def safe_stdev(values: List[float]) -> float:
            return statistics.stdev(values) if len(values) > 1 else 0.0

        return {
            "n_instances": len(all_instances),
            "n_success": len(successes),
            "success_rate": len(successes) / len(all_instances) if all_instances else 0.0,

            # Steps (all instances)
            "avg_steps": safe_mean([m["n_steps"] for m in all_instances]),
            "median_steps": safe_median([m["n_steps"] for m in all_instances]),
            "stdev_steps": safe_stdev([m["n_steps"] for m in all_instances]),

            # Steps (success only)
            "avg_steps_success": safe_mean([m["n_steps"] for m in successes]),
            "median_steps_success": safe_median([m["n_steps"] for m in successes]),

            # Cost
            "avg_cost": safe_mean([m["cost"] for m in all_instances]),
            "total_cost": sum(m["cost"] for m in all_instances),

            # Paging metrics
            "avg_paging_ops": safe_mean([m["manual_paging_ops"] for m in all_instances]),
            "total_paging_ops": sum(m["manual_paging_ops"] for m in all_instances),
            "avg_total_lines": safe_mean([m["paging_span_total_lines"] for m in all_instances]),
            "avg_unique_lines": safe_mean([m["paging_span_unique_lines"] for m in all_instances]),
            "avg_read_dup_rate": safe_mean([m["read_duplication_rate"] for m in all_instances]),

            # Exploration
            "avg_front_25_explore": safe_mean([m["front_25_explore_ratio"] for m in all_instances]),

            # Cache (if available)
            "avg_cache_hit_rate": safe_mean([
                m["cache_hit_rate"] for m in all_instances
                if m["cache_hit_rate"] is not None
            ]),
        }

    def generate_report(self, output_path: Path) -> None:
        """Generate full ablation report in Markdown format."""
        groups = sorted(set(k[0] for k in self.trajectories.keys()))

        report_lines = [
            "# Hybrid Ablation Experiment Report",
            "",
            f"**Date**: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M')}",
            f"**Groups**: {', '.join(groups)}",
            f"**Total Trajectories**: {len(self.trajectories)}",
            "",
            "---",
            "",
            "## 1. Group Summaries",
            "",
        ]

        # Per-group summaries
        group_summaries = {}
        for group in groups:
            summary = self.compute_group_summary(group)
            group_summaries[group] = summary

            report_lines.extend([
                f"### {group}",
                "",
                f"| Metric | Value |",
                f"|--------|-------|",
                f"| Instances | {summary['n_instances']} |",
                f"| Success Rate | {summary['success_rate']:.1%} |",
                f"| Avg Steps | {summary['avg_steps']:.1f} |",
                f"| Median Steps | {summary['median_steps']:.1f} |",
                f"| Avg Cost | ${summary['avg_cost']:.4f} |",
                f"| Avg Paging Ops | {summary['avg_paging_ops']:.1f} |",
                f"| Avg Unique Lines | {summary['avg_unique_lines']:.0f} |",
                f"| Read Dup Rate | {summary['avg_read_dup_rate']:.1%} |",
                f"| Front 25% Explore | {summary['avg_front_25_explore']:.1%} |",
                "",
            ])

        # Paired comparisons
        anchor_group = "v0" if "v0" in groups else (groups[0] if groups else None)
        if anchor_group and len(groups) > 1:
            report_lines.extend([
                "---",
                "",
                f"## 2. Paired Comparisons (vs {anchor_group})",
                "",
            ])

            for variant in groups:
                if variant == anchor_group:
                    continue

                deltas = self.compute_paired_deltas(anchor_group, variant)
                if not deltas or not deltas.get("steps_delta"):
                    report_lines.append(f"### {variant} — No paired data\n")
                    continue

                n_pairs = len(deltas["steps_delta"])

                report_lines.extend([
                    f"### {variant} vs {anchor_group}",
                    "",
                    f"**Paired instances**: {n_pairs}",
                    "",
                    f"| Metric | Mean Δ | Median Δ | Direction |",
                    f"|--------|--------|----------|-----------|",
                ])

                for metric, key in [
                    ("Steps", "steps_delta"),
                    ("Cost ($)", "cost_delta"),
                    ("Paging Ops", "paging_ops_delta"),
                    ("Unique Lines", "unique_lines_delta"),
                    ("Read Dup Rate", "read_dup_delta"),
                ]:
                    vals = deltas.get(key, [])
                    if vals:
                        mean_d = statistics.mean(vals)
                        median_d = statistics.median(vals)
                        direction = "↓ better" if mean_d < 0 else ("↑ worse" if mean_d > 0 else "=")
                        report_lines.append(
                            f"| {metric} | {mean_d:+.2f} | {median_d:+.2f} | {direction} |"
                        )

                report_lines.append("")

                # Regressions/improvements
                regs = deltas.get("regressions", [])
                imps = deltas.get("improvements", [])
                if regs or imps:
                    report_lines.append(f"- **Regressions**: {len(regs)}")
                    report_lines.append(f"- **Improvements**: {len(imps)}")
                    report_lines.append("")

        # Comparison table
        report_lines.extend([
            "---",
            "",
            "## 3. Summary Table",
            "",
            "| Group | N | Success% | Avg Steps | Median | Avg Cost | Paging Ops | Unique Lines | Read Dup% |",
            "|-------|---|----------|-----------|--------|----------|------------|--------------|-----------|",
        ])

        for group in groups:
            s = group_summaries[group]
            report_lines.append(
                f"| {group} | {s['n_instances']} | {s['success_rate']:.1%} | "
                f"{s['avg_steps']:.1f} | {s['median_steps']:.0f} | "
                f"${s['avg_cost']:.4f} | {s['avg_paging_ops']:.1f} | "
                f"{s['avg_unique_lines']:.0f} | {s['avg_read_dup_rate']:.1%} |"
            )

        report_lines.append("")

        # Write report
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("\n".join(report_lines))
        print(f"Report saved to {output_path}")

        # Also save JSON for downstream processing
        json_path = output_path.with_suffix(".json")
        with open(json_path, "w") as f:
            json.dump({
                "groups": groups,
                "summaries": group_summaries,
                "n_trajectories": len(self.trajectories),
            }, f, indent=2)
        print(f"JSON data saved to {json_path}")


def generate_plots(analyzer: HybridAblationAnalyzer, output_dir: Path) -> None:
    """Generate visualization plots."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
    except ImportError:
        print("Warning: matplotlib not available, skipping plots")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    groups = sorted(set(k[0] for k in analyzer.trajectories.keys()))

    if not groups:
        print("No groups to plot")
        return

    colors = plt.cm.tab10.colors

    # 1. Steps Distribution
    fig, ax = plt.subplots(figsize=(10, 6))
    data = []
    for group in groups:
        trajs = [v for k, v in analyzer.trajectories.items() if k[0] == group]
        steps = [sum(1 for m in t.get("messages", []) if m.get("role") == "assistant") for t in trajs]
        data.append(steps)
    ax.boxplot(data, labels=groups)
    ax.set_ylabel("Steps")
    ax.set_title("Steps Distribution by Group")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(output_dir / "steps_distribution.png", dpi=150)
    plt.close()
    print(f"  Saved: steps_distribution.png")

    # 2. Cost vs Steps
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, group in enumerate(groups):
        trajs = [v for k, v in analyzer.trajectories.items() if k[0] == group]
        steps = [sum(1 for m in t.get("messages", []) if m.get("role") == "assistant") for t in trajs]
        costs = [t.get("info", {}).get("model_stats", {}).get("instance_cost", 0) for t in trajs]
        ax.scatter(steps, costs, label=group, alpha=0.6, color=colors[i % len(colors)])
    ax.set_xlabel("Steps")
    ax.set_ylabel("Cost ($)")
    ax.set_title("Cost vs Steps by Group")
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "cost_vs_steps.png", dpi=150)
    plt.close()
    print(f"  Saved: cost_vs_steps.png")

    # 3. Paging Ops Distribution
    fig, ax = plt.subplots(figsize=(10, 6))
    data = []
    for group in groups:
        trajs = [v for k, v in analyzer.trajectories.items() if k[0] == group]
        paging_ops = [
            analyzer.parse_paging_from_messages(t.get("messages", []))["manual_paging_ops"]
            for t in trajs
        ]
        data.append(paging_ops)
    ax.boxplot(data, labels=groups)
    ax.set_ylabel("Manual Paging Ops (nl|sed)")
    ax.set_title("Paging Operations by Group")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(output_dir / "paging_ops_distribution.png", dpi=150)
    plt.close()
    print(f"  Saved: paging_ops_distribution.png")

    # 4. Unique Lines Distribution
    fig, ax = plt.subplots(figsize=(10, 6))
    data = []
    for group in groups:
        trajs = [v for k, v in analyzer.trajectories.items() if k[0] == group]
        unique_lines = [
            analyzer.parse_paging_from_messages(t.get("messages", []))["paging_span_unique_lines"]
            for t in trajs
        ]
        data.append(unique_lines)
    ax.boxplot(data, labels=groups)
    ax.set_ylabel("Unique Lines Read")
    ax.set_title("Unique Paging Lines by Group")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(output_dir / "unique_lines_distribution.png", dpi=150)
    plt.close()
    print(f"  Saved: unique_lines_distribution.png")

    # 5. Front 25% Exploration Ratio
    fig, ax = plt.subplots(figsize=(10, 6))
    ratios = []
    for group in groups:
        trajs = [v for k, v in analyzer.trajectories.items() if k[0] == group]
        group_ratios = [analyzer.compute_front_25_explore(t.get("messages", [])) for t in trajs]
        ratios.append(statistics.mean(group_ratios) if group_ratios else 0)
    ax.bar(groups, ratios, color=[colors[i % len(colors)] for i in range(len(groups))])
    ax.set_ylabel("Explore Ratio")
    ax.set_title("Front 25% Steps: Exploration Ratio by Group")
    ax.set_ylim(0, 1)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(output_dir / "front_25_explore.png", dpi=150)
    plt.close()
    print(f"  Saved: front_25_explore.png")

    print(f"Plots saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze LSR-Engine v2 ablation experiment results"
    )
    parser.add_argument(
        "--results-dir", type=Path, required=True,
        help="Directory containing trajectory files"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("analysis/ablation"),
        help="Output directory for report and plots"
    )
    parser.add_argument(
        "--plots", action="store_true",
        help="Generate visualization plots"
    )
    parser.add_argument(
        "--slices-json", type=Path,
        help="Optional: slices configuration JSON"
    )

    args = parser.parse_args()

    print(f"Loading trajectories from {args.results_dir}...")
    analyzer = HybridAblationAnalyzer(args.results_dir)
    n_loaded = analyzer.load_all_trajectories()

    if n_loaded == 0:
        print("Error: No trajectories found")
        return 1

    # Generate report
    report_path = args.output_dir / "ABLATION_REPORT.md"
    analyzer.generate_report(report_path)

    # Generate plots if requested
    if args.plots:
        print("\nGenerating plots...")
        generate_plots(analyzer, args.output_dir / "plots")

    return 0


if __name__ == "__main__":
    exit(main())

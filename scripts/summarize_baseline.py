#!/usr/bin/env python3
"""
Baseline Results Summarization Script

Analyzes SWE-bench trajectory files and generates a summary report.
"""

import json
import os
import sys
from pathlib import Path
from collections import defaultdict
import re


def load_trajectory(traj_path: Path) -> dict:
    """Load a trajectory JSON file."""
    with open(traj_path) as f:
        return json.load(f)


def analyze_trajectory(traj: dict) -> dict:
    """Extract metrics from a trajectory."""
    info = traj.get("info", {})
    messages = traj.get("messages", [])

    # Basic info
    result = {
        "instance_id": traj.get("instance_id", "unknown"),
        "exit_status": info.get("exit_status", "unknown"),
        "submission": info.get("submission", ""),
        "traceback": info.get("traceback", ""),
    }

    # Model stats
    model_stats = info.get("model_stats", {})
    result["cost"] = model_stats.get("instance_cost", 0)
    result["api_calls"] = model_stats.get("api_calls", 0)

    # Count steps/turns
    result["num_messages"] = len(messages)
    result["num_turns"] = sum(1 for m in messages if m.get("role") == "assistant")

    # Analyze message content
    result["tool_calls"] = 0
    result["commands"] = []

    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, str):
            # Count bash code blocks as tool calls
            bash_blocks = re.findall(r"```bash\n(.*?)\n```", content, re.DOTALL)
            result["tool_calls"] += len(bash_blocks)
            result["commands"].extend(bash_blocks)

    # Determine success/failure
    result["resolved"] = False
    if result["exit_status"] == "submitted":
        # Check if submission looks like a valid patch
        submission = result["submission"]
        if submission and "diff" in submission.lower():
            result["resolved"] = True

    # Classify failure reason
    if result["exit_status"] == "TimeoutExpired":
        result["failure_reason"] = "docker_timeout"
    elif result["exit_status"] == "CalledProcessError":
        result["failure_reason"] = "docker_error"
    elif result["exit_status"] == "cost_limit":
        result["failure_reason"] = "cost_limit"
    elif result["exit_status"] == "step_limit":
        result["failure_reason"] = "step_limit"
    elif result["exit_status"] == "submitted" and not result["resolved"]:
        result["failure_reason"] = "invalid_submission"
    elif result["resolved"]:
        result["failure_reason"] = None
    else:
        result["failure_reason"] = "other"

    return result


def summarize_results(results_dir: Path) -> dict:
    """Summarize all results in a directory."""
    results_dir = Path(results_dir)

    all_results = []

    # Find all trajectory files
    for traj_file in results_dir.rglob("*.traj.json"):
        traj = load_trajectory(traj_file)
        analysis = analyze_trajectory(traj)
        all_results.append(analysis)

    # Compute summary statistics
    summary = {
        "total_instances": len(all_results),
        "resolved": sum(1 for r in all_results if r["resolved"]),
        "failed": sum(1 for r in all_results if not r["resolved"]),
    }

    # Resolution rate
    summary["resolution_rate"] = (
        summary["resolved"] / summary["total_instances"] * 100
        if summary["total_instances"] > 0
        else 0
    )

    # Failure breakdown
    failure_counts = defaultdict(int)
    for r in all_results:
        if r["failure_reason"]:
            failure_counts[r["failure_reason"]] += 1
    summary["failure_breakdown"] = dict(failure_counts)

    # Average metrics (only for instances that actually ran)
    running_results = [r for r in all_results if r["num_turns"] > 0]
    if running_results:
        summary["avg_turns"] = sum(r["num_turns"] for r in running_results) / len(running_results)
        summary["avg_tool_calls"] = sum(r["tool_calls"] for r in running_results) / len(running_results)
        summary["avg_cost"] = sum(r["cost"] for r in running_results) / len(running_results)
    else:
        summary["avg_turns"] = 0
        summary["avg_tool_calls"] = 0
        summary["avg_cost"] = 0

    summary["total_cost"] = sum(r["cost"] for r in all_results)

    # Individual results
    summary["results"] = all_results

    return summary


def generate_markdown_report(summary: dict, output_path: Path, config_info: dict = None):
    """Generate a Markdown report from the summary."""
    lines = []

    lines.append("# Baseline Evaluation Report")
    lines.append("")
    lines.append(f"Generated: {__import__('datetime').datetime.now().isoformat()}")
    lines.append("")

    # Configuration
    if config_info:
        lines.append("## Configuration")
        lines.append("")
        lines.append("```yaml")
        for k, v in config_info.items():
            lines.append(f"{k}: {v}")
        lines.append("```")
        lines.append("")

    # Summary Statistics
    lines.append("## Summary Statistics")
    lines.append("")
    lines.append(f"| Metric | Value |")
    lines.append(f"|--------|-------|")
    lines.append(f"| Total Instances | {summary['total_instances']} |")
    lines.append(f"| Resolved | {summary['resolved']} |")
    lines.append(f"| Failed | {summary['failed']} |")
    lines.append(f"| Resolution Rate | {summary['resolution_rate']:.1f}% |")
    lines.append(f"| Avg Turns | {summary['avg_turns']:.1f} |")
    lines.append(f"| Avg Tool Calls | {summary['avg_tool_calls']:.1f} |")
    lines.append(f"| Total Cost | ${summary['total_cost']:.4f} |")
    lines.append("")

    # Failure Breakdown
    lines.append("## Failure Breakdown")
    lines.append("")
    lines.append("| Failure Type | Count | Percentage |")
    lines.append("|--------------|-------|------------|")
    total = summary['total_instances']
    for reason, count in sorted(summary['failure_breakdown'].items(), key=lambda x: -x[1]):
        pct = count / total * 100 if total > 0 else 0
        lines.append(f"| {reason} | {count} | {pct:.1f}% |")
    lines.append("")

    # Sample Failures
    lines.append("## Sample Failure Cases")
    lines.append("")

    # Group by failure reason and show examples
    by_reason = defaultdict(list)
    for r in summary['results']:
        if r['failure_reason']:
            by_reason[r['failure_reason']].append(r)

    for reason, examples in list(by_reason.items())[:3]:
        lines.append(f"### {reason}")
        for ex in examples[:2]:
            lines.append(f"\n**Instance:** `{ex['instance_id']}`")
            lines.append(f"- Exit status: `{ex['exit_status']}`")
            lines.append(f"- Turns: {ex['num_turns']}")
            lines.append(f"- Tool calls: {ex['tool_calls']}")
            if ex['traceback']:
                # Show last 3 lines of traceback
                tb_lines = ex['traceback'].strip().split('\n')[-3:]
                lines.append(f"- Traceback (last 3 lines):")
                lines.append("```")
                lines.extend(tb_lines)
                lines.append("```")
        lines.append("")

    # Instance Details Table
    lines.append("## Instance Details")
    lines.append("")
    lines.append("| Instance ID | Status | Turns | Tool Calls | Cost |")
    lines.append("|-------------|--------|-------|------------|------|")
    for r in summary['results']:
        status = "Resolved" if r['resolved'] else r['exit_status']
        lines.append(f"| {r['instance_id']} | {status} | {r['num_turns']} | {r['tool_calls']} | ${r['cost']:.4f} |")
    lines.append("")

    # Write report
    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))

    print(f"Report written to: {output_path}")


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Summarize baseline results")
    parser.add_argument("results_dir", help="Path to results directory")
    parser.add_argument("-o", "--output", help="Output markdown file", default="baseline_report.md")
    parser.add_argument("--json", help="Also output JSON summary", action="store_true")

    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"Error: {results_dir} does not exist")
        sys.exit(1)

    print(f"Analyzing results in: {results_dir}")
    summary = summarize_results(results_dir)

    # Output JSON if requested
    if args.json:
        json_path = results_dir / "summary.json"
        with open(json_path, 'w') as f:
            # Remove large fields for JSON output
            json_summary = {k: v for k, v in summary.items() if k != 'results'}
            json_summary['instance_ids'] = [r['instance_id'] for r in summary['results']]
            json.dump(json_summary, f, indent=2)
        print(f"JSON summary written to: {json_path}")

    # Generate Markdown report
    output_path = Path(args.output)
    config_info = {
        "model": "openai/gpt-5-mini",
        "subset": "lite",
        "split": "test",
        "slice": "0:20",
        "workers": 4,
        "environment": "docker",
    }
    generate_markdown_report(summary, output_path, config_info)

    # Print quick summary
    print("\n=== Quick Summary ===")
    print(f"Total: {summary['total_instances']}")
    print(f"Resolved: {summary['resolved']} ({summary['resolution_rate']:.1f}%)")
    print(f"Failed: {summary['failed']}")
    print(f"Total cost: ${summary['total_cost']:.4f}")


if __name__ == "__main__":
    main()

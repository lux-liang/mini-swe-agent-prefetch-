"""
Trajectory Parser for mini-swe-agent
Extracts step-level and instance-level metrics from trajectory files.
Supports prefetch ablation experiment metrics.
"""

import json
import re
import glob
import os
from pathlib import Path
from collections import Counter
import numpy as np
import pandas as pd


def load_traj(path: str) -> dict:
    """Load a trajectory JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# === Prefetch Ablation: nl|sed Paging Detection ===
NL_SED_PATTERN = re.compile(
    r"nl\s+(?:-\w+\s+)*['\"]?(?P<file>[^\s|'\"]+)['\"]?\s*\|\s*sed\s+-n\s*['\"]?(?P<start>\d+),(?P<end>\d+)p",
    re.IGNORECASE
)

# Alternative patterns for manual paging
HEAD_TAIL_PATTERN = re.compile(
    r"(?:head|tail)\s+(?:-n\s*)?(?P<lines>\d+)\s+['\"]?(?P<file>[^\s'\"]+)",
    re.IGNORECASE
)

SED_RANGE_PATTERN = re.compile(
    r"sed\s+-n\s*['\"]?(?P<start>\d+),(?P<end>\d+)p['\"]?\s+['\"]?(?P<file>[^\s'\"]+)",
    re.IGNORECASE
)


def extract_paging_info(cmd: str) -> dict | None:
    """Extract file path and line window from nl|sed or similar paging commands."""
    # Try nl|sed pattern first
    match = NL_SED_PATTERN.search(cmd)
    if match:
        return {
            "type": "nl_sed",
            "file": match.group("file"),
            "start": int(match.group("start")),
            "end": int(match.group("end")),
            "window_size": int(match.group("end")) - int(match.group("start")) + 1
        }

    # Try sed -n 'a,bp' file pattern
    match = SED_RANGE_PATTERN.search(cmd)
    if match:
        return {
            "type": "sed_range",
            "file": match.group("file"),
            "start": int(match.group("start")),
            "end": int(match.group("end")),
            "window_size": int(match.group("end")) - int(match.group("start")) + 1
        }

    return None


def calculate_paging_metrics(steps: list[dict]) -> dict:
    """Calculate manual paging metrics from steps."""
    paging_ops = []
    file_access_count = Counter()
    file_windows = {}  # file -> list of (start, end) windows

    for step in steps:
        cmd = step.get("command", "")
        paging = extract_paging_info(cmd)
        if paging:
            paging_ops.append(paging)
            file_path = paging["file"]
            file_access_count[file_path] += 1

            if file_path not in file_windows:
                file_windows[file_path] = []
            file_windows[file_path].append((paging["start"], paging["end"]))

    # Read duplication: files accessed >1 time via paging
    total_accesses = sum(file_access_count.values())
    duplicate_accesses = sum(c - 1 for c in file_access_count.values() if c > 1)

    # Calculate overlap ratio (how much re-reading of same lines)
    overlap_lines = 0
    total_lines_read = 0
    for file_path, windows in file_windows.items():
        if len(windows) > 1:
            # Sort windows and check overlaps
            windows_sorted = sorted(windows)
            for i in range(1, len(windows_sorted)):
                prev_end = windows_sorted[i-1][1]
                curr_start = windows_sorted[i][0]
                if curr_start <= prev_end:
                    overlap_lines += prev_end - curr_start + 1
        for start, end in windows:
            total_lines_read += end - start + 1

    return {
        "manual_paging_ops": len(paging_ops),
        "paging_files_count": len(file_access_count),
        "read_duplication_rate": duplicate_accesses / max(total_accesses, 1),
        "avg_window_size": np.mean([p["window_size"] for p in paging_ops]) if paging_ops else 0,
        "total_lines_paged": total_lines_read,
        "overlap_lines": overlap_lines,
        "overlap_ratio": overlap_lines / max(total_lines_read, 1),
    }


def calculate_explore_metrics(steps: list[dict], explore_types: set) -> dict:
    """Calculate exploration chain and front-loading metrics."""
    n_steps = len(steps)
    if n_steps == 0:
        return {
            "front_25_explore_ratio": 0.0,
            "front_50_explore_ratio": 0.0,
            "explore_chain_len": 0,
            "explore_ratio": 0.0,
        }

    # Front 25% exploration ratio
    front_25_idx = max(1, n_steps // 4)
    front_50_idx = max(1, n_steps // 2)

    front_25_steps = steps[:front_25_idx]
    front_50_steps = steps[:front_50_idx]

    front_25_explore = sum(1 for s in front_25_steps if s.get("command_type") in explore_types)
    front_50_explore = sum(1 for s in front_50_steps if s.get("command_type") in explore_types)
    total_explore = sum(1 for s in steps if s.get("command_type") in explore_types)

    # Max consecutive exploration chain
    max_chain = 0
    current_chain = 0
    for step in steps:
        if step.get("command_type") in explore_types:
            current_chain += 1
            max_chain = max(max_chain, current_chain)
        else:
            current_chain = 0

    return {
        "front_25_explore_ratio": front_25_explore / len(front_25_steps) if front_25_steps else 0,
        "front_50_explore_ratio": front_50_explore / len(front_50_steps) if front_50_steps else 0,
        "explore_chain_len": max_chain,
        "explore_ratio": total_explore / n_steps,
    }


def extract_bash_command(content: str) -> str | None:
    """Extract bash command from assistant message content."""
    # Look for ```bash ... ``` blocks
    match = re.search(r"```bash\s*\n(.*?)```", content, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


def classify_command(cmd: str) -> str:
    """Classify a bash command into categories."""
    if not cmd:
        return "other"

    cmd_lower = cmd.lower().strip()
    first_word = cmd_lower.split()[0] if cmd_lower.split() else ""

    # File reading
    if first_word in ["cat", "head", "tail", "less", "more"]:
        return "read_file"

    # Search/grep
    if first_word in ["rg", "grep", "ripgrep", "ag"]:
        return "search"

    # Directory listing
    if first_word in ["ls", "find", "tree", "fd"]:
        return "browse"

    # File editing
    if first_word in ["sed", "awk", "perl"]:
        return "edit"

    # Git operations
    if first_word == "git":
        return "git"

    # Python execution
    if first_word == "python" or first_word == "python3":
        return "python"

    # Testing
    if "pytest" in cmd_lower or "test" in cmd_lower:
        return "test"

    # Echo/create file
    if first_word == "echo" or first_word == "cat" and "<<" in cmd:
        return "write_file"

    # cd commands
    if first_word == "cd":
        return "cd"

    return "other"


def extract_files_from_command(cmd: str) -> list[str]:
    """Extract file paths mentioned in a command."""
    files = []
    # Match common file patterns
    patterns = [
        r'(?:cat|head|tail|less|more|rg|grep)\s+["\']?([^\s"\'>|]+\.\w+)',
        r'(?:python|python3)\s+["\']?([^\s"\'>]+\.py)',
    ]
    for pattern in patterns:
        matches = re.findall(pattern, cmd)
        files.extend(matches)
    return files


def parse_trajectory(traj_path: str) -> dict:
    """Parse a single trajectory file and extract metrics."""
    traj = load_traj(traj_path)

    instance_id = Path(traj_path).stem.replace(".traj", "")
    info = traj.get("info", {})
    messages = traj.get("messages", [])

    # Basic info
    result = {
        "instance_id": instance_id,
        "exit_status": info.get("exit_status", "unknown"),
        "total_messages": len(messages),
        "api_calls": info.get("model_stats", {}).get("api_calls", 0),
        "instance_cost": info.get("model_stats", {}).get("instance_cost", 0),
    }

    # Parse messages to extract steps
    steps = []
    command_counts = Counter()
    files_accessed = []

    for i, msg in enumerate(messages):
        role = msg.get("role", "")
        content = msg.get("content", "")

        if role == "assistant":
            cmd = extract_bash_command(content)
            if cmd:
                cmd_type = classify_command(cmd)
                command_counts[cmd_type] += 1
                files_accessed.extend(extract_files_from_command(cmd))

                steps.append({
                    "instance_id": instance_id,
                    "step_idx": len(steps),
                    "msg_idx": i,
                    "role": role,
                    "command": cmd,
                    "command_type": cmd_type,
                    "has_thought": "THOUGHT" in content.upper(),
                })

    result["total_steps"] = len(steps)
    result["steps"] = steps

    # Command type counts
    for cmd_type in ["read_file", "search", "browse", "edit", "git", "python", "test", "write_file", "cd", "other"]:
        result[f"{cmd_type}_count"] = command_counts.get(cmd_type, 0)

    # Files accessed
    result["unique_files"] = len(set(files_accessed))
    result["total_file_accesses"] = len(files_accessed)

    # === Prefetch Ablation Metrics ===
    # Manual paging metrics (nl|sed detection)
    paging_metrics = calculate_paging_metrics(steps)
    result.update(paging_metrics)

    # Exploration metrics (front-loading analysis)
    explore_types = {"read_file", "search", "browse", "cd"}
    explore_metrics = calculate_explore_metrics(steps, explore_types)
    result.update(explore_metrics)

    # Prefetch stats (if available in trajectory)
    prefetch_stats = info.get("prefetch_stats", {})
    result["prefetch_version"] = prefetch_stats.get("prefetch_version", "v0")
    result["prefetch_hit_rate"] = prefetch_stats.get("prefetch_hit_rate", 0)
    result["scroll_triggers"] = prefetch_stats.get("scroll_consolidation_triggers", 0)

    return result


def parse_run(run_dir: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Parse all trajectories in a run directory."""
    traj_paths = sorted(glob.glob(os.path.join(run_dir, "*", "*.traj.json")))

    all_steps = []
    all_instances = []

    for path in traj_paths:
        result = parse_trajectory(path)

        # Instance-level data
        instance_data = {k: v for k, v in result.items() if k != "steps"}
        all_instances.append(instance_data)

        # Step-level data
        all_steps.extend(result["steps"])

    steps_df = pd.DataFrame(all_steps)
    instances_df = pd.DataFrame(all_instances)

    return steps_df, instances_df


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python parse_traj.py <run_dir>")
        sys.exit(1)

    run_dir = sys.argv[1]
    steps_df, instances_df = parse_run(run_dir)

    print(f"Parsed {len(instances_df)} instances, {len(steps_df)} steps")
    print("\nInstance summary:")
    print(instances_df.describe())

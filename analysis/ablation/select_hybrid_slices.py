#!/usr/bin/env python3
"""
Repo-Aware Stratified Sampling for Prefetch Ablation Experiment

Generates 5 sampling windows:
- Sequential Blocks (2): 0:20, 20:40 - baseline efficiency
- Boundary-Crossing (1): crosses repo boundary - cold-start evaluation
- Heterogeneous Anchors (2): from index >= 260 - long-tail generalization

Background:
SWE-bench Verified is sorted alphabetically by instance_id (which starts with repo name),
so contiguous slices have very low repo diversity in early indices (0-252 is mostly astropy+django).
This protocol addresses the diversity limitation.
"""

import json
import random
from pathlib import Path
from collections import Counter
from dataclasses import dataclass, asdict, field
from typing import Optional

try:
    from datasets import load_dataset
    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False


@dataclass
class WindowInfo:
    """Information about a sampling window."""
    start: int
    end: int
    slice_str: str
    window_type: str  # "sequential", "boundary", "anchor"
    repos: list = field(default_factory=list)
    unique_repo_count: int = 0
    repo_distribution: dict = field(default_factory=dict)
    boundary_index: Optional[int] = None  # For boundary windows


def load_verified_instances() -> list[dict]:
    """Load SWE-bench Verified dataset instances."""
    if not HAS_DATASETS:
        raise RuntimeError("datasets library required. Install with: pip install datasets")

    ds = load_dataset("princeton-nlp/SWE-Bench_Verified", split="test")
    return [{"index": i, "instance_id": inst["instance_id"], "repo": inst["repo"]}
            for i, inst in enumerate(ds)]


def find_repo_boundaries(instances: list[dict]) -> list[dict]:
    """Find indices where repo changes."""
    boundaries = []
    for i in range(1, len(instances)):
        if instances[i]["repo"] != instances[i-1]["repo"]:
            boundaries.append({
                "index": i,
                "from_repo": instances[i-1]["repo"],
                "to_repo": instances[i]["repo"],
            })
    return boundaries


def create_window(instances: list[dict], start: int, end: int,
                  window_type: str, boundary_idx: int = None) -> WindowInfo:
    """Create a WindowInfo object for a slice."""
    start = max(0, start)
    end = min(len(instances), end)

    slice_instances = instances[start:end]
    repos = [inst["repo"] for inst in slice_instances]
    repo_counts = dict(Counter(repos).most_common())

    return WindowInfo(
        start=start,
        end=end,
        slice_str=f"{start}:{end}",
        window_type=window_type,
        repos=repos,
        unique_repo_count=len(set(repos)),
        repo_distribution=repo_counts,
        boundary_index=boundary_idx,
    )


def select_boundary_window(instances: list[dict], boundaries: list[dict],
                           seed: int = 42) -> WindowInfo:
    """Select a boundary-crossing window.

    Prefer boundaries that:
    1. Result in a more balanced window (both repos have similar counts)
    2. Are not in the dense django region (22-252)
    """
    random.seed(seed)

    # Score each boundary by balance and position
    candidates = []
    for b in boundaries:
        idx = b["index"]
        # Create candidate window centered on boundary
        start = max(0, idx - 10)
        end = min(len(instances), idx + 10)

        # Adjust to ensure window size = 20
        if end - start < 20:
            if start == 0:
                end = min(20, len(instances))
            else:
                start = max(0, end - 20)

        window = create_window(instances, start, end, "boundary", boundary_idx=idx)

        # Score: prefer balanced distribution and diverse repos
        if window.unique_repo_count >= 2:
            counts = list(window.repo_distribution.values())
            balance_score = min(counts) / max(counts) if max(counts) > 0 else 0
            diversity_score = window.unique_repo_count / 10.0

            # Penalize if entirely within django region
            in_django_region = (start >= 22 and end <= 252)
            region_penalty = 0.5 if in_django_region else 1.0

            total_score = (balance_score + diversity_score) * region_penalty
            candidates.append((total_score, window, b))

    if not candidates:
        # Fallback: use first boundary outside django
        for b in boundaries:
            if b["index"] > 252:
                idx = b["index"]
                return create_window(instances, idx - 10, idx + 10, "boundary", idx)
        # Ultimate fallback
        b = boundaries[0]
        return create_window(instances, b["index"] - 10, b["index"] + 10,
                            "boundary", b["index"])

    # Sort by score and pick best, or random among top candidates
    candidates.sort(key=lambda x: -x[0])
    top_candidates = candidates[:3]
    _, selected_window, _ = random.choice(top_candidates)

    return selected_window


def select_anchor_windows(instances: list[dict], min_start: int = 260,
                          seed: int = 42) -> list[WindowInfo]:
    """Select 2 heterogeneous anchor windows from index >= min_start.

    Strategy: Maximize combined unique repos using greedy selection.
    """
    random.seed(seed)

    # Generate all candidate windows
    max_start = len(instances) - 20
    candidates = []

    for start in range(min_start, max_start + 1, 5):  # Step by 5 to reduce candidates
        window = create_window(instances, start, start + 20, "anchor")
        candidates.append(window)

    if len(candidates) < 2:
        # Fallback if not enough candidates
        return [
            create_window(instances, min_start, min_start + 20, "anchor"),
            create_window(instances, min(min_start + 40, max_start),
                         min(min_start + 60, len(instances)), "anchor"),
        ]

    # Greedy selection for maximum combined diversity
    # First: pick window with most unique repos
    candidates.sort(key=lambda w: (-w.unique_repo_count, w.start))
    selected = [candidates[0]]
    selected_repos = set(candidates[0].repos)

    # Second: pick window that adds most new repos
    best_addition = None
    best_new_repos = -1
    best_distance = 0

    for w in candidates:
        if w.start == selected[0].start:
            continue
        # Check no overlap
        if not (w.end <= selected[0].start or w.start >= selected[0].end):
            continue

        new_repos = len(set(w.repos) - selected_repos)
        distance = abs(w.start - selected[0].start)

        if new_repos > best_new_repos or (new_repos == best_new_repos and distance > best_distance):
            best_new_repos = new_repos
            best_distance = distance
            best_addition = w

    if best_addition:
        selected.append(best_addition)
    else:
        # Fallback: pick furthest non-overlapping window
        for w in reversed(candidates):
            if w.start >= selected[0].end or w.end <= selected[0].start:
                selected.append(w)
                break

    # Sort by start index
    selected.sort(key=lambda w: w.start)
    return selected


def generate_hybrid_slices(output_path: str = None, seed: int = 42) -> dict:
    """Generate all 5 hybrid sampling windows."""
    print("=" * 60)
    print("Repo-Aware Stratified Sampling for Prefetch Ablation")
    print("=" * 60)

    print("\nLoading SWE-bench Verified dataset...")
    instances = load_verified_instances()
    print(f"Total instances: {len(instances)}")

    # Dataset overview
    all_repos = [inst["repo"] for inst in instances]
    print(f"Unique repos in dataset: {len(set(all_repos))}")
    print("\nRepo distribution (sorted by index range):")
    boundaries = find_repo_boundaries(instances)
    current_repo = instances[0]["repo"]
    current_start = 0
    for b in boundaries:
        print(f"  [{current_start:3d}-{b['index']-1:3d}] {current_repo}")
        current_repo = b["to_repo"]
        current_start = b["index"]
    print(f"  [{current_start:3d}-{len(instances)-1:3d}] {current_repo}")

    # Generate windows
    windows = []

    # A) Sequential Blocks (2)
    print("\n--- Sequential Blocks ---")
    for start, end in [(0, 20), (20, 40)]:
        w = create_window(instances, start, end, "sequential")
        windows.append(w)
        print(f"  {w.slice_str}: {w.unique_repo_count} repo(s) - {w.repo_distribution}")

    # B) Boundary-Crossing Window (1)
    print("\n--- Boundary-Crossing Window ---")
    print(f"  Found {len(boundaries)} repo boundaries")
    boundary_window = select_boundary_window(instances, boundaries, seed=seed)
    windows.append(boundary_window)
    print(f"  Selected: {boundary_window.slice_str} (boundary at {boundary_window.boundary_index})")
    print(f"    {boundary_window.unique_repo_count} repo(s) - {boundary_window.repo_distribution}")

    # C) Heterogeneous Anchors (2)
    print("\n--- Heterogeneous Anchor Windows ---")
    anchor_windows = select_anchor_windows(instances, min_start=260, seed=seed)
    for i, w in enumerate(anchor_windows):
        windows.append(w)
        print(f"  Anchor {i+1}: {w.slice_str} - {w.unique_repo_count} repo(s) - {w.repo_distribution}")

    # Summary
    print("\n" + "=" * 60)
    print("SAMPLING SUMMARY")
    print("=" * 60)

    all_sampled_repos = set()
    for w in windows:
        all_sampled_repos.update(w.repos)

    print(f"\nFinal 5 windows: {[w.slice_str for w in windows]}")
    print(f"Total sampled instances: {sum(w.end - w.start for w in windows)}")
    print(f"Combined unique repos: {len(all_sampled_repos)}")
    print(f"Repos covered: {sorted(all_sampled_repos)}")

    print("\nWindow breakdown by type:")
    for wtype in ["sequential", "boundary", "anchor"]:
        type_windows = [w for w in windows if w.window_type == wtype]
        type_repos = set()
        for w in type_windows:
            type_repos.update(w.repos)
        print(f"  {wtype.capitalize():12s}: {len(type_windows)} window(s), {len(type_repos)} unique repo(s)")

    # Prepare output
    result = {
        "protocol": "Repo-Aware Stratified Sampling",
        "seed": seed,
        "dataset": "princeton-nlp/SWE-Bench_Verified",
        "total_instances": len(instances),
        "total_unique_repos": len(set(all_repos)),
        "slices": [w.slice_str for w in windows],
        "windows": [asdict(w) for w in windows],
        "combined_unique_repos": len(all_sampled_repos),
        "combined_repos_list": sorted(all_sampled_repos),
        "boundaries_found": len(boundaries),
        "boundary_details": boundaries[:10],  # First 10 for reference
    }

    # Save
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nSaved to: {output_path}")

    return result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Generate hybrid sampling windows for Prefetch ablation"
    )
    parser.add_argument("--output", "-o",
                       default="analysis/ablation/slices_selected.json",
                       help="Output JSON file")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    result = generate_hybrid_slices(output_path=args.output, seed=args.seed)

    # Print for bash consumption
    print("\n# For bash script:")
    print(f'SLICES=("{" ".join(result["slices"])}")')

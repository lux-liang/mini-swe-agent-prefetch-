#!/usr/bin/env python3
"""
Slice Selection with Repo Diversity Check for Prefetch V2 Ablation

Generates 5 slices:
- 3 fixed: 0:20, 20:40, 40:60
- 2 random: from [0, 180], length=20, with repo diversity guarantee

Ensures random slices cover at least 5 unique repos combined.
"""

import json
import random
import sys
from pathlib import Path
from collections import Counter
from dataclasses import dataclass, asdict
from typing import Optional

# Try to load dataset
try:
    from datasets import load_dataset
    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False
    print("Warning: datasets not available, using mock data")


@dataclass
class SliceInfo:
    start: int
    end: int
    slice_str: str
    is_random: bool
    seed_used: Optional[int]
    instance_ids: list
    repos: list
    unique_repo_count: int
    repo_counts: dict


@dataclass
class SliceSelectionResult:
    fixed_slices: list
    random_slices: list
    all_slices: list
    total_unique_repos: int
    random_seed_final: int
    retries: int
    repo_diversity_check_passed: bool
    min_random_unique_repos: int


def get_verified_instances() -> list[dict]:
    """Load SWE-bench Verified instances."""
    if not HAS_DATASETS:
        # Mock data for testing
        repos = ["astropy/astropy", "django/django", "sympy/sympy",
                 "matplotlib/matplotlib", "scikit-learn/scikit-learn",
                 "pytest-dev/pytest", "psf/requests", "sphinx-doc/sphinx"]
        return [{"instance_id": f"repo_{i}", "repo": repos[i % len(repos)]}
                for i in range(200)]

    ds = load_dataset("princeton-nlp/SWE-Bench_Verified", split="test")
    return [{"instance_id": inst["instance_id"], "repo": inst["repo"]}
            for inst in ds]


def get_slice_info(instances: list[dict], start: int, end: int,
                   is_random: bool = False, seed_used: int = None) -> SliceInfo:
    """Get detailed info about a slice."""
    slice_instances = instances[start:end]
    instance_ids = [inst["instance_id"] for inst in slice_instances]
    repos = [inst["repo"] for inst in slice_instances]
    repo_counts = dict(Counter(repos).most_common())

    return SliceInfo(
        start=start,
        end=end,
        slice_str=f"{start}:{end}",
        is_random=is_random,
        seed_used=seed_used,
        instance_ids=instance_ids,
        repos=repos,
        unique_repo_count=len(set(repos)),
        repo_counts=repo_counts,
    )


def select_random_slices(instances: list[dict], base_seed: int = 42,
                         min_unique_repos: int = 2, max_retries: int = 10) -> tuple:
    """Select 2 random slices with repo diversity guarantee.

    Note: SWE-bench Verified is sorted alphabetically by instance_id (which starts
    with repo name), so contiguous slices typically contain only 1-2 repos.
    We lower the diversity threshold to 2 unique repos for the random slices.
    The goal is to ensure the random slices don't both come from the same repo section.
    """
    n_instances = len(instances)
    max_start = min(180, n_instances - 20)

    # Get repo boundaries to help select diverse slices
    repo_boundaries = []
    current_repo = None
    for i, inst in enumerate(instances[:200]):  # Only consider first 200
        if inst["repo"] != current_repo:
            repo_boundaries.append((i, inst["repo"]))
            current_repo = inst["repo"]

    for retry in range(max_retries):
        current_seed = base_seed + retry
        random.seed(current_seed)

        # Generate 2 random start points from [60, 180] (avoid fixed slices 0-59)
        valid_starts = list(range(60, max_start + 1))
        if len(valid_starts) < 2:
            valid_starts = list(range(0, max_start + 1))

        starts = random.sample(valid_starts, min(2, len(valid_starts)))
        starts = sorted(starts)

        # Ensure no overlap between random slices
        if len(starts) == 2 and abs(starts[0] - starts[1]) < 20:
            # Try to space them out more
            starts[1] = min(starts[0] + 40, max_start)

        # Get slice info
        slices = []
        for s in starts:
            sl = get_slice_info(instances, s, min(s + 20, len(instances)),
                               is_random=True, seed_used=current_seed)
            slices.append(sl)

        if len(slices) < 2:
            continue

        # Check combined repo diversity
        combined_repos = set(slices[0].repos + slices[1].repos)
        if len(combined_repos) >= min_unique_repos:
            return slices, current_seed, retry + 1

    # If all retries fail, return the last attempt with a warning
    print(f"Warning: Could not find slices with {min_unique_repos} unique repos after {max_retries} retries")
    print(f"Note: Dataset is sorted alphabetically, limiting within-slice diversity")
    return slices if slices else [
        get_slice_info(instances, 60, 80, is_random=True, seed_used=base_seed),
        get_slice_info(instances, 100, 120, is_random=True, seed_used=base_seed)
    ], base_seed + max_retries - 1, max_retries


def generate_slices(output_path: str = None, base_seed: int = 42) -> SliceSelectionResult:
    """Generate all 5 slices with repo diversity check."""
    print("Loading SWE-bench Verified dataset...")
    instances = get_verified_instances()
    print(f"Loaded {len(instances)} instances")

    # Fixed slices
    fixed_configs = [(0, 20), (20, 40), (40, 60)]
    fixed_slices = [get_slice_info(instances, s, e) for s, e in fixed_configs]

    print("\nFixed slices:")
    for sl in fixed_slices:
        print(f"  {sl.slice_str}: {sl.unique_repo_count} unique repos")

    # Random slices with diversity check
    print("\nSelecting random slices with repo diversity check...")
    random_slices, final_seed, retries = select_random_slices(
        instances, base_seed=base_seed, min_unique_repos=5
    )

    print(f"Random slices (seed={final_seed}, retries={retries}):")
    for sl in random_slices:
        print(f"  {sl.slice_str}: {sl.unique_repo_count} unique repos")

    # Combined stats
    all_slices = fixed_slices + random_slices
    all_repos = set()
    for sl in all_slices:
        all_repos.update(sl.repos)

    random_repos = set()
    for sl in random_slices:
        random_repos.update(sl.repos)

    result = SliceSelectionResult(
        fixed_slices=[asdict(sl) for sl in fixed_slices],
        random_slices=[asdict(sl) for sl in random_slices],
        all_slices=[sl.slice_str for sl in all_slices],
        total_unique_repos=len(all_repos),
        random_seed_final=final_seed,
        retries=retries,
        repo_diversity_check_passed=len(random_repos) >= 5,
        min_random_unique_repos=len(random_repos),
    )

    print(f"\n=== Summary ===")
    print(f"All slices: {result.all_slices}")
    print(f"Total unique repos (all 5 slices): {result.total_unique_repos}")
    print(f"Random slices unique repos: {result.min_random_unique_repos}")
    print(f"Diversity check passed: {result.repo_diversity_check_passed}")

    # Save to file
    if output_path:
        output_data = {
            "slices": result.all_slices,
            "fixed_slices": result.fixed_slices,
            "random_slices": result.random_slices,
            "total_unique_repos": result.total_unique_repos,
            "random_seed_final": result.random_seed_final,
            "retries": result.retries,
            "repo_diversity_check_passed": result.repo_diversity_check_passed,
            "min_random_unique_repos": result.min_random_unique_repos,
        }
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"\nSaved to {output_path}")

    return result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate slices for ablation experiment")
    parser.add_argument("--output", "-o", default="analysis/ablation/slices_selected.json",
                       help="Output JSON file path")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed")
    args = parser.parse_args()

    result = generate_slices(output_path=args.output, base_seed=args.seed)

    # Print slice strings for bash script consumption
    print("\n# For bash script:")
    print(f"SLICES=({' '.join(result.all_slices)})")

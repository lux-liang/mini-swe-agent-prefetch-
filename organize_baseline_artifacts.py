#!/usr/bin/env python3
"""
organize_baseline_artifacts.py

Organize mini-swe-agent baseline results to Windows D drive in WSL environment.
Copies/moves and renames files with semantic naming conventions.

Python 3.10+ | Standard library only
"""

import argparse
import shutil
import sys
from pathlib import Path


# Default paths
DEFAULT_SRC = "/home/lux_liang/work/mini-swe-agent-main/analysis/figures"
DEFAULT_DST = "/mnt/d/baseline/verified_20"
REPORT_SRC = "/home/lux_liang/work/mini-swe-agent-main/analysis/analysis_report.md"

# Rename mapping: old_prefix -> new_prefix
RENAME_MAP = {
    "fig1_step_distribution": "steps_per_instance",
    "fig2_command_breakdown": "command_type_distribution",
    "fig3_cost_vs_steps": "cost_vs_steps",
    "fig4_command_heatmap": "instance_command_heatmap",
    "fig5_trajectory_timeline": "trajectory_timelines",
    "fig6_temporal_pattern": "command_distribution_over_time",
    "fig7_api_usage": "api_calls_and_cost",
    "fig8_explore_vs_exploit": "explore_vs_exploit",
}

# Suffix for all figure files
SUFFIX = "verified_20"

# Expected files
EXPECTED_PNG = [f"{prefix}_{SUFFIX}.png" for prefix in RENAME_MAP.keys()]
EXPECTED_PDF = [f"{prefix}_{SUFFIX}.pdf" for prefix in RENAME_MAP.keys()]
EXPECTED_CSV = [f"instances_{SUFFIX}.csv", f"steps_{SUFFIX}.csv"]


def get_new_filename(old_name: str) -> str:
    """Apply rename mapping to a filename."""
    for old_prefix, new_prefix in RENAME_MAP.items():
        if old_name.startswith(old_prefix):
            return old_name.replace(old_prefix, new_prefix, 1)
    return old_name


def check_source_files(src_dir: Path, report_path: Path) -> tuple[list[Path], list[str]]:
    """
    Check if all expected source files exist.
    Returns (found_files, missing_files).
    """
    found = []
    missing = []

    # Check PNG files
    for filename in EXPECTED_PNG:
        path = src_dir / filename
        if path.exists():
            found.append(path)
        else:
            missing.append(str(path))

    # Check PDF files
    for filename in EXPECTED_PDF:
        path = src_dir / filename
        if path.exists():
            found.append(path)
        else:
            missing.append(str(path))

    # Check CSV files
    for filename in EXPECTED_CSV:
        path = src_dir / filename
        if path.exists():
            found.append(path)
        else:
            missing.append(str(path))

    # Check report (optional, don't add to missing if not found)
    if report_path.exists():
        found.append(report_path)

    return found, missing


def check_destination_conflicts(
    src_files: list[Path], dst_dir: Path, report_path: Path
) -> list[Path]:
    """Check if any destination files already exist."""
    conflicts = []

    for src_file in src_files:
        # Determine destination subdirectory and new name
        if src_file.suffix == ".png":
            dst_subdir = dst_dir / "figures" / "png"
            new_name = get_new_filename(src_file.name)
        elif src_file.suffix == ".pdf":
            dst_subdir = dst_dir / "figures" / "pdf"
            new_name = get_new_filename(src_file.name)
        elif src_file.suffix == ".csv":
            dst_subdir = dst_dir / "data"
            new_name = src_file.name
        elif src_file.suffix == ".md" and src_file == report_path:
            dst_subdir = dst_dir / "report"
            new_name = src_file.name
        else:
            continue

        dst_path = dst_subdir / new_name
        if dst_path.exists():
            conflicts.append(dst_path)

    return conflicts


def build_operation_plan(
    src_files: list[Path], dst_dir: Path, report_path: Path
) -> list[tuple[Path, Path, str]]:
    """
    Build list of (src, dst, rename_info) tuples.
    rename_info is empty string if no rename, otherwise "old -> new".
    """
    plan = []

    for src_file in src_files:
        # Determine destination subdirectory and new name
        if src_file.suffix == ".png":
            dst_subdir = dst_dir / "figures" / "png"
            new_name = get_new_filename(src_file.name)
        elif src_file.suffix == ".pdf":
            dst_subdir = dst_dir / "figures" / "pdf"
            new_name = get_new_filename(src_file.name)
        elif src_file.suffix == ".csv":
            dst_subdir = dst_dir / "data"
            new_name = src_file.name
        elif src_file.suffix == ".md" and src_file == report_path:
            dst_subdir = dst_dir / "report"
            new_name = src_file.name
        else:
            continue

        dst_path = dst_subdir / new_name

        # Build rename info
        if new_name != src_file.name:
            rename_info = f"{src_file.name} -> {new_name}"
        else:
            rename_info = ""

        plan.append((src_file, dst_path, rename_info))

    return plan


def execute_plan(
    plan: list[tuple[Path, Path, str]],
    move: bool = False,
    dry_run: bool = False,
) -> tuple[list[Path], list[tuple[Path, Path]], list[str]]:
    """
    Execute the operation plan.
    Returns (created_dirs, transferred_files, rename_mappings).
    """
    created_dirs: list[Path] = []
    transferred_files: list[tuple[Path, Path]] = []
    rename_mappings: list[str] = []

    # Collect all unique directories to create
    dirs_to_create = set()
    for _, dst_path, _ in plan:
        dirs_to_create.add(dst_path.parent)

    # Create directories
    for dir_path in sorted(dirs_to_create):
        if not dir_path.exists():
            if not dry_run:
                dir_path.mkdir(parents=True, exist_ok=True)
            created_dirs.append(dir_path)

    # Transfer files
    for src_path, dst_path, rename_info in plan:
        if not dry_run:
            if move:
                shutil.move(str(src_path), str(dst_path))
            else:
                shutil.copy2(str(src_path), str(dst_path))

        transferred_files.append((src_path, dst_path))

        if rename_info:
            rename_mappings.append(rename_info)

    return created_dirs, transferred_files, rename_mappings


def print_summary(
    created_dirs: list[Path],
    transferred_files: list[tuple[Path, Path]],
    rename_mappings: list[str],
    move: bool,
    dry_run: bool,
    dst_dir: Path,
) -> None:
    """Print execution summary."""
    action = "MOVE" if move else "COPY"
    prefix = "[DRY-RUN] " if dry_run else ""

    print(f"\n{'='*60}")
    print(f"{prefix}EXECUTION SUMMARY")
    print(f"{'='*60}")

    # Created directories
    print(f"\n{prefix}Created directories ({len(created_dirs)}):")
    if created_dirs:
        for d in created_dirs:
            print(f"  + {d}")
    else:
        print("  (none)")

    # Transferred files
    print(f"\n{prefix}{action} files ({len(transferred_files)}):")
    for src, dst in transferred_files:
        print(f"  {src.name}")
        print(f"    -> {dst}")

    # Rename mappings
    print(f"\n{prefix}Rename mappings ({len(rename_mappings)}):")
    if rename_mappings:
        for mapping in rename_mappings:
            print(f"  {mapping}")
    else:
        print("  (none)")

    print(f"\n{'='*60}")
    print(f"{prefix}Total: {len(transferred_files)} files {'moved' if move else 'copied'}")
    print(f"{'='*60}")

    # Explorer hint
    print(f"\nTo open the destination folder in Windows Explorer, run:")
    print(f"  explorer.exe $(wslpath -w '{dst_dir}')")
    print()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Organize mini-swe-agent baseline results to Windows D drive.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Dry run (preview only)
  python organize_baseline_artifacts.py --dry-run

  # Copy files (default)
  python organize_baseline_artifacts.py

  # Move files instead of copy
  python organize_baseline_artifacts.py --move

  # Overwrite existing files
  python organize_baseline_artifacts.py --overwrite

  # Custom source and destination
  python organize_baseline_artifacts.py --src /path/to/src --dst /path/to/dst
""",
    )

    parser.add_argument(
        "--src",
        type=str,
        default=DEFAULT_SRC,
        help=f"Source directory (default: {DEFAULT_SRC})",
    )
    parser.add_argument(
        "--dst",
        type=str,
        default=DEFAULT_DST,
        help=f"Destination directory (default: {DEFAULT_DST})",
    )
    parser.add_argument(
        "--move",
        action="store_true",
        help="Move files instead of copying (default: copy)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing files in destination",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print plan without executing file operations",
    )

    args = parser.parse_args()

    src_dir = Path(args.src)
    dst_dir = Path(args.dst)
    report_path = Path(REPORT_SRC)

    # Validate source directory
    if not src_dir.exists():
        print(f"ERROR: Source directory does not exist: {src_dir}", file=sys.stderr)
        return 1

    if not src_dir.is_dir():
        print(f"ERROR: Source path is not a directory: {src_dir}", file=sys.stderr)
        return 1

    # Check source files
    print(f"Checking source files in: {src_dir}")
    found_files, missing_files = check_source_files(src_dir, report_path)

    if missing_files:
        print(f"\nERROR: {len(missing_files)} required file(s) missing:", file=sys.stderr)
        for f in missing_files:
            print(f"  - {f}", file=sys.stderr)
        return 1

    print(f"Found {len(found_files)} files to process")

    # Check for report
    if report_path.exists():
        print(f"Found report: {report_path}")
    else:
        print(f"Note: Report not found at {report_path} (skipping)")

    # Check destination conflicts
    if not args.overwrite:
        conflicts = check_destination_conflicts(found_files, dst_dir, report_path)
        if conflicts:
            print(
                f"\nERROR: {len(conflicts)} file(s) already exist in destination:",
                file=sys.stderr,
            )
            for c in conflicts:
                print(f"  - {c}", file=sys.stderr)
            print("\nUse --overwrite to replace existing files.", file=sys.stderr)
            return 1

    # Build operation plan
    plan = build_operation_plan(found_files, dst_dir, report_path)

    if not plan:
        print("No files to process.")
        return 0

    # Execute plan
    created_dirs, transferred_files, rename_mappings = execute_plan(
        plan, move=args.move, dry_run=args.dry_run
    )

    # Print summary
    print_summary(
        created_dirs,
        transferred_files,
        rename_mappings,
        move=args.move,
        dry_run=args.dry_run,
        dst_dir=dst_dir,
    )

    if args.dry_run:
        print("This was a dry run. No files were actually modified.")
        print("Remove --dry-run to execute the operation.")

    return 0


if __name__ == "__main__":
    sys.exit(main())

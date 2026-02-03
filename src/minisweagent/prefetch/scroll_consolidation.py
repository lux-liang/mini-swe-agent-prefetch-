"""
Scroll Consolidation Module for Prefetch V2
Detects nl|sed manual paging patterns and provides consolidated file content.
"""

import re
import os
from dataclasses import dataclass, field
from typing import Any
from collections import defaultdict


# Pattern to detect nl|sed paging commands
NL_SED_PATTERN = re.compile(
    r"nl\s+(?:-\w+\s+)*['\"]?(?P<file>[^\s|'\"]+)['\"]?\s*\|\s*sed\s+-n\s*['\"]?(?P<start>\d+),(?P<end>\d+)p",
    re.IGNORECASE
)

# Alternative: sed -n 'a,bp' file
SED_RANGE_PATTERN = re.compile(
    r"sed\s+-n\s*['\"]?(?P<start>\d+),(?P<end>\d+)p['\"]?\s+['\"]?(?P<file>[^\s'\"]+)",
    re.IGNORECASE
)

# head/tail patterns
HEAD_PATTERN = re.compile(r"head\s+(?:-n\s*)?(?P<lines>\d+)\s+['\"]?(?P<file>[^\s'\"]+)", re.IGNORECASE)
TAIL_PATTERN = re.compile(r"tail\s+(?:-n\s*)?[+]?(?P<lines>\d+)\s+['\"]?(?P<file>[^\s'\"]+)", re.IGNORECASE)


@dataclass
class PagingRequest:
    """Represents a detected paging request."""
    file_path: str
    start_line: int
    end_line: int
    pattern_type: str  # "nl_sed", "sed_range", "head", "tail"
    original_command: str


@dataclass
class FileCache:
    """Cache for file content with line-based access."""
    content: str
    lines: list[str]
    total_lines: int
    access_count: int = 0
    last_access_step: int = 0


@dataclass
class ScrollConsolidationStats:
    """Statistics for scroll consolidation."""
    paging_requests_detected: int = 0
    consolidations_triggered: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    total_lines_prefetched: int = 0
    files_consolidated: set = field(default_factory=set)
    request_history: list = field(default_factory=list)


class ScrollConsolidator:
    """
    Detects manual paging patterns and provides consolidated file content.

    Key behaviors:
    1. Detect nl|sed patterns in commands
    2. Track consecutive accesses to same file
    3. When threshold reached, prefetch larger block
    4. Cache results to avoid re-reading
    """

    def __init__(
        self,
        lookahead_lines: int = 200,
        consolidate_threshold: int = 2,
        max_block_size: int = 1000,
        cache_ttl_steps: int = 10,
        enabled: bool = True,
    ):
        self.lookahead_lines = lookahead_lines
        self.consolidate_threshold = consolidate_threshold
        self.max_block_size = max_block_size
        self.cache_ttl_steps = cache_ttl_steps
        self.enabled = enabled

        # Internal state
        self.file_cache: dict[str, FileCache] = {}
        self.access_history: dict[str, list[PagingRequest]] = defaultdict(list)
        self.current_step = 0
        self.stats = ScrollConsolidationStats()

    def detect_paging_request(self, command: str) -> PagingRequest | None:
        """Detect if command is a manual paging operation."""
        if not self.enabled:
            return None

        # Try nl|sed pattern
        match = NL_SED_PATTERN.search(command)
        if match:
            return PagingRequest(
                file_path=match.group("file"),
                start_line=int(match.group("start")),
                end_line=int(match.group("end")),
                pattern_type="nl_sed",
                original_command=command,
            )

        # Try sed range pattern
        match = SED_RANGE_PATTERN.search(command)
        if match:
            return PagingRequest(
                file_path=match.group("file"),
                start_line=int(match.group("start")),
                end_line=int(match.group("end")),
                pattern_type="sed_range",
                original_command=command,
            )

        return None

    def should_consolidate(self, file_path: str) -> bool:
        """Check if we should consolidate reads for this file."""
        history = self.access_history.get(file_path, [])
        recent = [r for r in history if self.current_step - r.start_line <= self.cache_ttl_steps]
        return len(recent) >= self.consolidate_threshold

    def get_consolidated_range(self, request: PagingRequest) -> tuple[int, int]:
        """Calculate the consolidated range to prefetch."""
        history = self.access_history.get(request.file_path, [])

        if not history:
            # First access: prefetch from start with lookahead
            start = max(1, request.start_line)
            end = min(request.end_line + self.lookahead_lines, request.start_line + self.max_block_size)
            return start, end

        # Find the max line accessed so far
        max_accessed = max(r.end_line for r in history)

        # Prefetch from current position with lookahead
        start = request.start_line
        end = min(
            max(request.end_line, max_accessed) + self.lookahead_lines,
            start + self.max_block_size
        )

        return start, end

    def process_command(self, command: str, step: int) -> dict[str, Any]:
        """
        Process a command and return prefetch suggestion if applicable.

        Returns:
            dict with keys:
            - detected: bool - whether paging was detected
            - should_prefetch: bool - whether we should inject prefetched content
            - prefetch_suggestion: str | None - suggested content to inject
            - file_path: str | None - file being accessed
            - range: tuple[int, int] | None - line range to prefetch
        """
        self.current_step = step

        result = {
            "detected": False,
            "should_prefetch": False,
            "prefetch_suggestion": None,
            "file_path": None,
            "range": None,
            "stats": None,
        }

        if not self.enabled:
            return result

        request = self.detect_paging_request(command)
        if not request:
            return result

        result["detected"] = True
        result["file_path"] = request.file_path
        self.stats.paging_requests_detected += 1

        # Record access
        self.access_history[request.file_path].append(request)
        self.stats.request_history.append({
            "step": step,
            "file": request.file_path,
            "start": request.start_line,
            "end": request.end_line,
            "type": request.pattern_type,
        })

        # Check if consolidation should happen
        if self.should_consolidate(request.file_path):
            start, end = self.get_consolidated_range(request)
            result["should_prefetch"] = True
            result["range"] = (start, end)
            self.stats.consolidations_triggered += 1
            self.stats.files_consolidated.add(request.file_path)
            self.stats.total_lines_prefetched += end - start

        return result

    def read_file_range(self, file_path: str, start: int, end: int) -> str | None:
        """Read a range of lines from file (1-indexed)."""
        try:
            # Check cache first
            if file_path in self.file_cache:
                cache = self.file_cache[file_path]
                cache.access_count += 1
                cache.last_access_step = self.current_step
                self.stats.cache_hits += 1

                # Return requested range
                start_idx = max(0, start - 1)
                end_idx = min(end, cache.total_lines)
                lines = cache.lines[start_idx:end_idx]
                return self._format_lines(lines, start)

            # Read file and cache
            self.stats.cache_misses += 1

            if not os.path.exists(file_path):
                return None

            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                content = f.read()

            lines = content.splitlines()
            self.file_cache[file_path] = FileCache(
                content=content,
                lines=lines,
                total_lines=len(lines),
                access_count=1,
                last_access_step=self.current_step,
            )

            # Return requested range
            start_idx = max(0, start - 1)
            end_idx = min(end, len(lines))
            return self._format_lines(lines[start_idx:end_idx], start)

        except Exception as e:
            return f"[Error reading file: {e}]"

    def _format_lines(self, lines: list[str], start_line: int) -> str:
        """Format lines with line numbers (nl-style)."""
        formatted = []
        for i, line in enumerate(lines):
            line_num = start_line + i
            formatted.append(f"{line_num:6d}\t{line}")
        return "\n".join(formatted)

    def get_prefetch_content(self, file_path: str, start: int, end: int) -> str:
        """Get prefetched content for injection into agent context."""
        content = self.read_file_range(file_path, start, end)
        if content is None:
            return ""

        header = f"\n[Prefetch: {file_path} lines {start}-{end}]\n"
        footer = f"\n[End prefetch - {end - start + 1} lines]\n"

        return header + content + footer

    def get_stats(self) -> dict[str, Any]:
        """Get current statistics."""
        return {
            "paging_requests_detected": self.stats.paging_requests_detected,
            "consolidations_triggered": self.stats.consolidations_triggered,
            "cache_hits": self.stats.cache_hits,
            "cache_misses": self.stats.cache_misses,
            "total_lines_prefetched": self.stats.total_lines_prefetched,
            "files_consolidated": list(self.stats.files_consolidated),
            "unique_files_accessed": len(self.access_history),
        }

    def cleanup_stale_cache(self):
        """Remove stale cache entries."""
        stale_files = [
            f for f, cache in self.file_cache.items()
            if self.current_step - cache.last_access_step > self.cache_ttl_steps
        ]
        for f in stale_files:
            del self.file_cache[f]


def create_scroll_consolidator(config: dict) -> ScrollConsolidator:
    """Factory function to create ScrollConsolidator from config dict."""
    scroll_config = config.get("scroll_consolidation", {})
    return ScrollConsolidator(
        lookahead_lines=scroll_config.get("lookahead_lines", 200),
        consolidate_threshold=scroll_config.get("consolidate_threshold", 2),
        max_block_size=scroll_config.get("max_block_size", 1000),
        cache_ttl_steps=scroll_config.get("cache_ttl_steps", 10),
        enabled=config.get("enable_scroll_consolidation", False),
    )


# === Test/Demo ===
if __name__ == "__main__":
    # Demo usage
    consolidator = ScrollConsolidator(
        lookahead_lines=100,
        consolidate_threshold=2,
        enabled=True,
    )

    # Simulate agent commands
    test_commands = [
        "nl -ba /testbed/django/db/models/query.py | sed -n '1,50p'",
        "nl -ba /testbed/django/db/models/query.py | sed -n '51,100p'",
        "nl -ba /testbed/django/db/models/query.py | sed -n '101,150p'",
        "grep -n 'def filter' /testbed/django/db/models/query.py",
        "nl -ba /testbed/django/db/models/query.py | sed -n '200,250p'",
    ]

    for i, cmd in enumerate(test_commands):
        result = consolidator.process_command(cmd, step=i)
        if result["detected"]:
            print(f"Step {i}: Detected paging -> {result['file_path']}")
            if result["should_prefetch"]:
                print(f"  -> Consolidation triggered! Prefetch range: {result['range']}")

    print("\n=== Stats ===")
    print(consolidator.get_stats())

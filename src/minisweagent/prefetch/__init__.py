"""
Prefetch Configuration for mini-swe-agent Ablation Experiments
Defines feature flags and parameters for V0-V4 prefetch versions.
"""

from dataclasses import dataclass, field
from typing import Literal, Any
import os


@dataclass
class PrefetchConfig:
    """Prefetch feature flags and parameters."""

    # === Master Switch ===
    enabled: bool = False
    version: Literal["v0", "v1", "v2", "v3", "v4"] = "v0"

    # === Module Flags ===
    enable_anchor_sink: bool = False       # V1+: PR keyword anchoring
    enable_evidence: bool = False          # V1+: Import/reference expansion
    enable_scroll_consolidation: bool = False  # V2+: nl|sed pattern detection
    enable_one_hop: bool = False           # V3+: Call graph traversal
    enable_gate: bool = False              # V3+: Stagnation-triggered hints
    enable_jitter: bool = False            # Randomized prefetch (Lite only)
    enable_shadow_feedback: bool = False   # V4: Background patch validation

    # === Budget Parameters ===
    max_files: int = 3
    max_lines_per_file: int = 500
    max_total_tokens: int = 4000

    # === Scroll Consolidation (V2 core) ===
    scroll_detect_pattern: str = r"nl\s+(?:-\w+\s+)*.*\|\s*sed\s+-n"
    scroll_lookahead_lines: int = 200
    scroll_consolidate_threshold: int = 2
    scroll_max_block_size: int = 1000
    scroll_cache_ttl_steps: int = 10

    # === Anchor Parameters ===
    anchor_keyword_sources: list[str] = field(default_factory=lambda: ["pr_title", "pr_body", "error_message"])
    anchor_min_score: float = 0.3

    # === Evidence Parameters ===
    evidence_follow_imports: bool = True
    evidence_max_depth: int = 1

    # === One-Hop Parameters ===
    one_hop_track_calls: bool = True
    one_hop_track_imports: bool = True
    one_hop_max_hops: int = 1

    # === Gate Parameters ===
    gate_stagnation_threshold: int = 2
    gate_flatness_threshold: float = 0.7
    gate_stack_coverage_min: float = 0.3
    gate_max_hints: int = 3

    # === Jitter (Lite only) ===
    jitter_probability: float = 0.0  # Verified: 0.0, Lite: 0.05

    # === Logging ===
    log_prefetch_decisions: bool = True
    log_cache_hits: bool = True


# === Profile Presets ===
def get_v0_config() -> PrefetchConfig:
    """V0 Baseline - no prefetch."""
    return PrefetchConfig(enabled=False, version="v0")


def get_v1_config(profile: str = "verified") -> PrefetchConfig:
    """V1 Anchor-Sink + Evidence."""
    is_lite = profile == "lite"
    return PrefetchConfig(
        enabled=True,
        version="v1",
        enable_anchor_sink=True,
        enable_evidence=True,
        enable_scroll_consolidation=False,
        enable_one_hop=False,
        enable_gate=False,
        enable_jitter=False,
        max_files=5 if is_lite else 3,
        max_lines_per_file=600 if is_lite else 500,
        max_total_tokens=5000 if is_lite else 4000,
        jitter_probability=0.0,
    )


def get_v2_config(profile: str = "verified") -> PrefetchConfig:
    """V2 Scroll Consolidation (key experiment)."""
    is_lite = profile == "lite"
    return PrefetchConfig(
        enabled=True,
        version="v2",
        enable_anchor_sink=True,
        enable_evidence=True,
        enable_scroll_consolidation=True,  # Core feature
        enable_one_hop=False,
        enable_gate=False,
        enable_jitter=False,
        max_files=8 if is_lite else 5,
        max_lines_per_file=1000 if is_lite else 800,
        max_total_tokens=10000 if is_lite else 6000,
        scroll_lookahead_lines=300 if is_lite else 200,
        scroll_consolidate_threshold=2,
        scroll_max_block_size=1200 if is_lite else 1000,
        jitter_probability=0.0,
    )


def get_v3_config(profile: str = "verified") -> PrefetchConfig:
    """V3 Full - One-Hop + Gate."""
    is_lite = profile == "lite"
    return PrefetchConfig(
        enabled=True,
        version="v3",
        enable_anchor_sink=True,
        enable_evidence=True,
        enable_scroll_consolidation=True,
        enable_one_hop=True,
        enable_gate=True,
        enable_jitter=is_lite,
        max_files=12 if is_lite else 8,
        max_lines_per_file=1200 if is_lite else 1000,
        max_total_tokens=12000 if is_lite else 10000,
        scroll_lookahead_lines=300 if is_lite else 200,
        scroll_consolidate_threshold=2,
        gate_stagnation_threshold=2,
        gate_max_hints=3,
        jitter_probability=0.05 if is_lite else 0.0,
    )


def get_config_by_version(version: str, profile: str = "verified") -> PrefetchConfig:
    """Get prefetch config by version string."""
    configs = {
        "v0": get_v0_config,
        "v1": lambda: get_v1_config(profile),
        "v2": lambda: get_v2_config(profile),
        "v3": lambda: get_v3_config(profile),
    }
    factory = configs.get(version, get_v0_config)
    return factory() if callable(factory) else factory


def config_to_dict(config: PrefetchConfig) -> dict[str, Any]:
    """Convert config to dictionary for YAML/JSON serialization."""
    return {
        "enabled": config.enabled,
        "version": config.version,
        "enable_anchor_sink": config.enable_anchor_sink,
        "enable_evidence": config.enable_evidence,
        "enable_scroll_consolidation": config.enable_scroll_consolidation,
        "enable_one_hop": config.enable_one_hop,
        "enable_gate": config.enable_gate,
        "enable_jitter": config.enable_jitter,
        "budget": {
            "max_files": config.max_files,
            "max_lines_per_file": config.max_lines_per_file,
            "max_total_tokens": config.max_total_tokens,
        },
        "scroll_consolidation": {
            "detect_pattern": config.scroll_detect_pattern,
            "lookahead_lines": config.scroll_lookahead_lines,
            "consolidate_threshold": config.scroll_consolidate_threshold,
            "max_block_size": config.scroll_max_block_size,
            "cache_ttl_steps": config.scroll_cache_ttl_steps,
        },
        "gate": {
            "stagnation_threshold": config.gate_stagnation_threshold,
            "flatness_threshold": config.gate_flatness_threshold,
            "stack_coverage_min": config.gate_stack_coverage_min,
            "max_hints": config.gate_max_hints,
        },
        "jitter_probability": config.jitter_probability,
    }


# === Profile Summary ===
VERIFIED_VS_LITE = """
Verified vs Lite Profile Differences:

| Parameter               | Verified | Lite   | Reason                    |
|-------------------------|----------|--------|---------------------------|
| max_files               | 5        | 8      | Lite problems more spread |
| max_total_tokens        | 6000     | 10000  | Lite needs more context   |
| enable_one_hop          | False    | True   | Verified single-file focus|
| enable_gate             | False    | True   | Lite more prone to stall  |
| enable_jitter           | False    | True   | Verified needs determinism|
| jitter_probability      | 0.0      | 0.05   | Lite more fault tolerant  |
| scroll_lookahead_lines  | 200      | 300    | Lite files are longer     |
"""

if __name__ == "__main__":
    import json

    print("=== Prefetch Configuration Presets ===\n")

    for version in ["v0", "v1", "v2", "v3"]:
        config = get_config_by_version(version, "verified")
        print(f"\n--- {version.upper()} (Verified Profile) ---")
        print(json.dumps(config_to_dict(config), indent=2))

    print(VERIFIED_VS_LITE)

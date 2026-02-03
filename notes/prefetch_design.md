# Prefetch Module Design Document

**Version:** 1.0 (Draft)
**Author:** Research Team
**Date:** 2026-01-27

## 1. Motivation and Hypothesis

### 1.1 Observations from Baseline (Projected)

Based on literature review of SWE-bench agent behaviors and the mini-swe-agent architecture, we hypothesize the following inefficiencies exist in baseline agent runs:

1. **Repository Navigation Overhead**
   - Agents spend significant turns (estimated 30-50%) exploring repository structure
   - Common patterns: `ls`, `find`, `grep` sequences before locating relevant files
   - Hypothesis: Pre-computing file relevance can reduce navigation turns by 40-60%

2. **File Localization Delay**
   - Agents often visit 5-15 files before finding the correct modification target
   - Error messages and stack traces contain strong signals for file relevance
   - Hypothesis: Parsing problem statements can provide file hints with 70%+ accuracy

3. **Common Failure Patterns**
   - Step limit exceeded (agent stuck in exploration loops)
   - Cost limit exceeded (excessive file reading)
   - Invalid submissions (agent didn't understand codebase structure)
   - Hypothesis: Upfront context can reduce these failure modes by 25-40%

### 1.2 Core Hypothesis

> **Prefetching relevant files and repository structure before agent execution can improve resolution rate by 15-30% while reducing average turns by 40-60%.**

Key assumptions:
- Problem statements contain extractable signals (file names, class names, error messages)
- Repository structure is predictable from project type and naming conventions
- Providing focused context is better than letting agents explore freely

## 2. Interface Definition

### 2.1 Input Schema

```python
@dataclass
class PrefetchInput:
    """Input to the prefetch module."""

    instance_id: str              # SWE-bench instance ID
    problem_statement: str        # Full problem/issue description
    repo_name: str                # e.g., "django/django"
    base_commit: str              # Git commit hash

    # Optional hints
    test_patch: str | None        # Test file changes (if available)
    hints_text: str | None        # Additional hints from dataset

    # Configuration
    max_files: int = 10           # Maximum files to prefetch
    max_depth: int = 3            # Maximum directory depth to scan
    include_tests: bool = False   # Whether to include test files
```

### 2.2 Output Schema

```python
@dataclass
class PrefetchOutput:
    """Output from the prefetch module."""

    instance_id: str

    # Core outputs
    relevant_files: list[FileInfo]     # Ranked list of relevant files
    repo_structure: dict               # Directory tree summary
    entry_points: list[str]            # Suggested starting points

    # Metadata
    confidence_scores: dict[str, float]  # Per-file confidence
    extraction_method: str              # How files were identified
    processing_time_ms: float

    # For debugging/analysis
    raw_signals: dict                   # Extracted signals from input
    rejected_candidates: list[str]      # Files considered but rejected

@dataclass
class FileInfo:
    """Information about a prefetched file."""
    path: str
    content: str | None          # File content (if fetched)
    relevance_score: float       # 0.0 to 1.0
    reason: str                  # Why this file is relevant
    line_hints: list[int] | None # Specific lines of interest
```

### 2.3 Hook Point

The prefetch module integrates before agent initialization:

```
┌─────────────────┐
│  Load Instance  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│    PREFETCH     │  ◄── NEW MODULE
│  (async/batch)  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Format Prompt  │  (inject prefetch results)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Agent Loop    │
└─────────────────┘
```

### 2.4 Integration with Agent Prompt

Prefetch results are injected into the instance template:

```jinja2
{% if prefetch_results %}
<prefetch_context>
## Repository Structure
{{ prefetch_results.repo_structure | format_tree }}

## Relevant Files (ranked by relevance)
{% for file in prefetch_results.relevant_files[:5] %}
### {{ file.path }} (confidence: {{ file.relevance_score | round(2) }})
Reason: {{ file.reason }}
{% if file.content %}
```{{ file.path | get_extension }}
{{ file.content | truncate(2000) }}
```
{% endif %}
{% endfor %}

## Suggested Starting Points
{% for entry in prefetch_results.entry_points %}
- {{ entry }}
{% endfor %}
</prefetch_context>
{% endif %}
```

## 3. Algorithm Roadmap

### 3.1 MVP (v0.1) - Signal Extraction

**Goal:** Extract file/class/function mentions from problem statement.

**Algorithm:**
```
1. Parse problem_statement for:
   - File paths (regex: `[\w/]+\.(py|js|java|...)`)
   - Class names (regex: `class\s+(\w+)`)
   - Function names (regex: `def\s+(\w+)|function\s+(\w+)`)
   - Error locations (regex: `File "([^"]+)", line (\d+)`)

2. Parse test_patch (if available) for:
   - Test file paths → infer source file paths
   - Test class/function names → infer source names

3. Return ranked list based on:
   - Exact mentions (score: 1.0)
   - Inferred from tests (score: 0.8)
   - Partial matches (score: 0.5)
```

**Complexity:** O(n) where n = length of problem statement
**Expected Impact:** 10-15% reduction in navigation turns

### 3.2 v1.0 - Repository-Aware Prefetch

**Goal:** Combine signal extraction with repository structure analysis.

**Algorithm:**
```
1. Run MVP signal extraction

2. Clone/fetch repository at base_commit

3. Build file index:
   - Parse directory structure
   - Extract imports/dependencies
   - Identify entry points (setup.py, __main__.py, etc.)

4. Expand candidates:
   - For each extracted signal, find matching files
   - Include files that import/are imported by matches
   - Include test files that test matched files

5. Rank by:
   - Direct mention score
   - Import graph distance
   - File modification frequency (from git history)
   - Code complexity metrics
```

**Complexity:** O(n * m) where m = number of files in repo
**Expected Impact:** 25-35% reduction in navigation turns

### 3.3 v2.0 - ML-Enhanced Prefetch

**Goal:** Use embeddings/LLM for semantic matching.

**Algorithm:**
```
1. Run v1.0 prefetch

2. Embed problem statement using code embedding model
   (e.g., CodeBERT, StarCoder embeddings)

3. Embed top-k candidate files

4. Compute semantic similarity

5. Re-rank candidates using combined score:
   score = α * structural_score + β * semantic_score + γ * historical_score

6. Optional: Use small LLM to generate file relevance explanations
```

**Complexity:** O(k * embedding_time)
**Expected Impact:** 40-50% reduction in navigation turns, 15-20% resolution rate improvement

## 4. Logging Schema

### 4.1 Per-Step Log Format

File: `prefetch_step_{instance_id}.json`

```json
{
  "instance_id": "django__django-11039",
  "timestamp": "2026-01-27T15:30:00Z",
  "version": "0.1.0",

  "input": {
    "problem_statement_length": 1234,
    "has_test_patch": true,
    "has_hints": false
  },

  "extraction": {
    "method": "regex_v1",
    "signals": {
      "file_mentions": ["django/core/management/commands/sqlmigrate.py"],
      "class_mentions": ["Command"],
      "function_mentions": ["handle"],
      "error_locations": []
    },
    "processing_time_ms": 12.5
  },

  "candidates": {
    "initial_count": 15,
    "after_filtering": 8,
    "final_count": 5
  },

  "output": {
    "files": [
      {
        "path": "django/core/management/commands/sqlmigrate.py",
        "relevance_score": 0.95,
        "reason": "directly mentioned in problem statement",
        "content_included": true,
        "content_length": 2456
      }
    ],
    "repo_structure_depth": 3,
    "entry_points": ["django/core/management/"]
  },

  "metrics": {
    "total_time_ms": 156.7,
    "files_scanned": 1234,
    "memory_mb": 45.2
  }
}
```

### 4.2 Aggregate Metrics

For A/B analysis:

```json
{
  "experiment_id": "prefetch_ab_001",
  "group": "treatment",  // or "control"
  "instance_id": "django__django-11039",

  "prefetch_metrics": {
    "files_suggested": 5,
    "files_actually_visited": 3,
    "suggestion_hit_rate": 0.6,
    "first_relevant_file_turn": 2,  // Turn when agent first visited relevant file
    "wasted_navigation_turns": 1
  },

  "outcome_metrics": {
    "resolved": true,
    "total_turns": 15,
    "total_cost": 0.0234,
    "exit_status": "submitted"
  }
}
```

## 5. Experiment Design

### 5.1 A/B Test Setup

**Hypothesis:** Prefetch (treatment) improves resolution rate compared to baseline (control).

**Design:**
- Same instance set for both groups (paired design)
- Same model (gpt-5-mini)
- Same prompts except prefetch injection
- Random assignment at run time (controlled by seed)

**Sample Size Calculation:**
- Baseline resolution rate (estimated): 15-20%
- Minimum detectable effect: 10% absolute improvement
- Power: 0.80, α: 0.05
- Required: ~100 instances per group

**Proposed Splits:**
```
# Group A (Control - no prefetch)
--slice 0:50 --env PREFETCH=0

# Group B (Treatment - with prefetch)
--slice 0:50 --env PREFETCH=1
```

### 5.2 Ablation Studies

1. **Signal Source Ablation**
   - Problem statement only vs. problem + test patch
   - Expected: test patch adds 5-10% accuracy

2. **Context Size Ablation**
   - 1, 3, 5, 10 files prefetched
   - Expected: diminishing returns after 5 files

3. **Content Inclusion Ablation**
   - File paths only vs. paths + content
   - Expected: content adds 10-15% accuracy but increases cost

### 5.3 Metrics

**Primary:**
- Resolution rate (% of instances resolved)
- Average turns to resolution

**Secondary:**
- Cost per instance
- First-file-visit accuracy (was first visited file relevant?)
- Navigation efficiency (relevant files visited / total files visited)
- Prefetch hit rate (suggested files that were actually used)

## 6. Risks and Mitigations

### 6.1 Risk: Prefetch Bias

**Description:** Prefetch might bias agent toward suggested files even when they're wrong.

**Mitigation:**
- Include confidence scores and reasons
- Instruct agent that suggestions are hints, not requirements
- Monitor "false positive" rate (suggested files that weren't relevant)
- Add ablation: random file suggestions to measure bias effect

### 6.2 Risk: Budget Inflation

**Description:** Prefetch processing adds cost/time before agent runs.

**Mitigation:**
- Set strict time budget (< 5s per instance)
- Cache repository clones across instances
- Use lightweight extraction methods first
- Monitor prefetch cost as % of total cost

### 6.3 Risk: Stale Evidence

**Description:** Prefetch based on problem statement might not reflect actual fix location.

**Mitigation:**
- Use multiple signal sources (problem + test + hints)
- Include confidence scores to indicate uncertainty
- Validate against ground truth patches in development
- Monitor accuracy metrics over time

### 6.4 Risk: Context Window Overflow

**Description:** Too much prefetch content might exceed context limits or dilute focus.

**Mitigation:**
- Set max_files limit
- Truncate file contents intelligently
- Prioritize file paths over content for large repos
- Test with different context sizes

## 7. Implementation Plan

### Phase 1: MVP (Week 1)
- [ ] Implement signal extraction (regex-based)
- [ ] Add prefetch output schema
- [ ] Integrate with prompt template
- [ ] Add logging infrastructure
- [ ] Run sanity tests on 5 instances

### Phase 2: Baseline Comparison (Week 2)
- [ ] Run control group (20 instances, no prefetch)
- [ ] Run treatment group (20 instances, with prefetch)
- [ ] Compute metrics and compare
- [ ] Document findings

### Phase 3: Iteration (Week 3-4)
- [ ] Implement v1.0 (repo-aware)
- [ ] Run ablation studies
- [ ] Tune parameters based on results
- [ ] Scale to 100+ instances

## 8. API Reference (Proposed)

```python
# Main entry point
def prefetch(
    instance: SWEBenchInstance,
    config: PrefetchConfig | None = None,
) -> PrefetchOutput:
    """
    Prefetch relevant files for a SWE-bench instance.

    Args:
        instance: SWE-bench instance with problem statement, repo, etc.
        config: Optional configuration overrides.

    Returns:
        PrefetchOutput with relevant files and metadata.
    """
    pass

# Configuration
@dataclass
class PrefetchConfig:
    max_files: int = 10
    max_depth: int = 3
    include_content: bool = True
    max_content_length: int = 5000
    extraction_methods: list[str] = field(default_factory=lambda: ["regex", "ast"])
    cache_dir: Path | None = None
    timeout_seconds: float = 30.0

# Environment variable control
# PREFETCH=0  -> disabled (control group)
# PREFETCH=1  -> enabled (treatment group)
# PREFETCH=2  -> enabled + verbose logging
```

## 9. References

1. SWE-bench paper and evaluation methodology
2. mini-swe-agent architecture documentation
3. Code search and retrieval literature
4. A/B testing best practices for ML systems

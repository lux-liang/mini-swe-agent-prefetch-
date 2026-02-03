"""
LSR-Engine v2 — Logic Super-Resolution Prefetch
================================================
Single-file, zero-external-dependency prefetch coordinator for SWE-bench agents.

Design pillars:
  1. Pager-first: reduce nl|sed scroll thrashing before anything else.
  2. Atlas-optional: full functionality without call-graph data.
  3. No runtime graph compute: read offline node_stats only.
  4. Context hygiene & budget governance.
  5. Shadow metrics: per-round hit@topM with shrink state machine.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    runtime_checkable,
)

logger = logging.getLogger("lsr_engine_v2")

# ---------------------------------------------------------------------------
# Constants & defaults
# ---------------------------------------------------------------------------
CHARS_PER_TOKEN = 4
DEFAULT_PREFETCH_BUDGET_TOKENS = 6000
DEFAULT_MAX_PAGER_BLOCKS = 2
DEFAULT_MERGE_GAP = 5
DEFAULT_CONTEXT_LINES = 10
DEFAULT_MAX_CHARS_PER_BLOCK = 2400
DEFAULT_MAX_LINES_PER_BLOCK = 80
DEFAULT_EVIDENCE_TOP_K = 2
DEFAULT_EVIDENCE_MAX_LINES = 30
DEFAULT_SHADOW_TOP_M = 5
DEFAULT_SHRINK_WINDOW = 3
DEFAULT_SHRINK_FACTOR = 0.7
DEFAULT_MAX_FANOUT_PER_NODE = 4
DEFAULT_MAX_FUNCS_TOTAL = 8
DEFAULT_HUB_SCORE_HIGH = 0.8
DEFAULT_HUB_FILTER_SIM_THRESHOLD = 0.15
DEFAULT_JITTER_RATIO = 0.0
DEFAULT_JITTER_TOKEN_CAP_RATIO = 0.25
DEFAULT_JITTER_TEMPERATURE = 1.0


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class CodeSpan:
    """Represents a contiguous range of lines in a source file."""

    file_path: str
    start_line: int
    end_line: int

    @property
    def length(self) -> int:
        return max(0, self.end_line - self.start_line + 1)


@dataclass
class PagerBlock:
    """A consolidated pager block ready for rendering."""

    span: CodeSpan
    content: str = ""
    source: str = "pager"


@dataclass
class EvidenceBlock:
    """A single evidence snippet from retrieval results."""

    symbol: str
    score: float
    span: Optional[CodeSpan]
    content: str = ""
    source: str = "evidence"


@dataclass
class InfillerBlock:
    """A block produced by deterministic infilling (callee expansion)."""

    func_id: str
    span: Optional[CodeSpan]
    content: str = ""
    hop: int = 1
    source: str = "det"


@dataclass
class JitterBlock:
    """A block produced by controlled jitter sampling."""

    func_id: str
    span: Optional[CodeSpan]
    content: str = ""
    source: str = "jit"


@dataclass
class BudgetState:
    """Snapshot of the current budget allocation."""

    uncertainty_score: float = 0.0
    flatness: float = 0.0
    coverage: float = 0.0
    stagnation: float = 0.0
    repeat: float = 0.0
    aggression_level: int = 0
    token_budget_prefetch: int = DEFAULT_PREFETCH_BUDGET_TOKENS
    jitter_ratio: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Serialise for logging."""
        return {
            "uncertainty": round(self.uncertainty_score, 4),
            "flatness": round(self.flatness, 4),
            "coverage": round(self.coverage, 4),
            "stagnation": round(self.stagnation, 4),
            "repeat": round(self.repeat, 4),
            "aggression_level": self.aggression_level,
            "token_budget_prefetch": self.token_budget_prefetch,
            "jitter_ratio": round(self.jitter_ratio, 4),
        }


@dataclass
class PrefetchPlan:
    """The full prefetch plan for one agent round."""

    pager_blocks: List[PagerBlock] = field(default_factory=list)
    evidence_blocks: List[EvidenceBlock] = field(default_factory=list)
    det_blocks: List[InfillerBlock] = field(default_factory=list)
    jitter_blocks: List[JitterBlock] = field(default_factory=list)
    budget_state: Optional[BudgetState] = None
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def pager_spans(self) -> List[CodeSpan]:
        return [b.span for b in self.pager_blocks]

    @property
    def det_func_ids(self) -> List[str]:
        return [b.func_id for b in self.det_blocks]

    @property
    def jitter_func_ids(self) -> List[str]:
        return [b.func_id for b in self.jitter_blocks]


@dataclass
class ShadowRecord:
    """One round of shadow evaluation data."""

    step_id: int
    predictions_top_m: List[str] = field(default_factory=list)
    det_count: int = 0
    jit_count: int = 0
    actual_accesses: List[str] = field(default_factory=list)
    hit_at_top_m: bool = False
    unique_hits: int = 0


# ---------------------------------------------------------------------------
# A. Atlas Protocol (optional)
# ---------------------------------------------------------------------------
@runtime_checkable
class Atlas(Protocol):
    """Optional offline code-graph interface.

    Implementations must provide pre-computed node_stats — runtime graph
    algorithms (PageRank, betweenness, etc.) are strictly forbidden.
    """

    def callees(self, func_id: str) -> List[str]:
        """Return direct callees of *func_id*."""
        ...

    def data_neighbors(self, func_id: str) -> List[str]:
        """Return data-flow neighbours of *func_id*."""
        ...

    def func_span(self, func_id: str) -> Optional[CodeSpan]:
        """Return the source span for *func_id*, or None."""
        ...

    def node_stats(self, func_id: str) -> Dict[str, Any]:
        """Return **offline-cached** statistics for *func_id*.

        Must include at minimum:
            hub_score (float), centrality (float),
            structural_entropy (float), is_hub (bool).
        """
        ...


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _estimate_tokens(text: str) -> int:
    """Rough token estimate: ceil(len / CHARS_PER_TOKEN)."""
    return math.ceil(len(text) / CHARS_PER_TOKEN) if text else 0


def _stable_hash(*parts: Any) -> int:
    """Deterministic integer hash from arbitrary parts."""
    raw = "|".join(str(p) for p in parts)
    return int(hashlib.sha256(raw.encode()).hexdigest()[:16], 16)


def _truncate_lines(text: str, max_lines: int) -> str:
    """Keep at most *max_lines* lines."""
    lines = text.splitlines(keepends=True)
    if len(lines) <= max_lines:
        return text
    return "".join(lines[:max_lines]) + f"\n... ({len(lines) - max_lines} lines truncated)\n"


def _truncate_chars(text: str, max_chars: int) -> str:
    """Keep at most *max_chars* characters."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + f"\n... ({len(text) - max_chars} chars truncated)\n"


def _similarity(a: str, b: str) -> float:
    """Quick similarity via SequenceMatcher (0..1)."""
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a[:500].lower(), b[:500].lower()).ratio()


# ===================================================================
# 1) CommandPagerParser
# ===================================================================
class CommandPagerParser:
    """Parse manual paging commands like ``nl -ba file | sed -n '240,520p'``.

    Supports common shell variations:
      - single/double quotes around the sed range
      - optional flags on ``nl``
      - whitespace variations around the pipe
    """

    # Regex: nl [flags] <path> | sed -n '<start>,<end>p'
    _PATTERN = re.compile(
        r"""nl\s+(?:-\w+\s+)*"""        # nl with optional flags
        r"""(?P<file>[^\s|]+)\s*"""      # file path (no spaces, stop at pipe)
        r"""\|\s*sed\s+-n\s*"""          # pipe into sed -n
        r"""['"]?"""                     # optional opening quote
        r"""(?P<start>\d+)"""            # start line
        r""","""
        r"""(?P<end>\d+)"""             # end line
        r"""p"""
        r"""['"]?""",                   # optional closing quote
        re.VERBOSE,
    )

    @classmethod
    def parse(cls, cmd: str) -> Optional[CodeSpan]:
        """Try to extract a CodeSpan from a raw shell command string.

        Returns None when the command does not match the nl|sed pattern.
        """
        m = cls._PATTERN.search(cmd)
        if not m:
            return None
        return CodeSpan(
            file_path=m.group("file"),
            start_line=int(m.group("start")),
            end_line=int(m.group("end")),
        )

    @classmethod
    def parse_many(cls, cmds: Sequence[str]) -> List[CodeSpan]:
        """Parse a batch of commands, returning only successful matches."""
        spans: List[CodeSpan] = []
        for cmd in cmds:
            span = cls.parse(cmd)
            if span is not None:
                spans.append(span)
        return spans


# ===================================================================
# 2) PagerConsolidator
# ===================================================================
class PagerConsolidator:
    """Merge overlapping / nearby paging windows per file and emit top-K blocks.

    This is the **pager-first** core: its output gets priority budget allocation.
    """

    def __init__(
        self,
        max_blocks: int = DEFAULT_MAX_PAGER_BLOCKS,
        merge_gap: int = DEFAULT_MERGE_GAP,
        context_lines: int = DEFAULT_CONTEXT_LINES,
    ) -> None:
        self.max_blocks = max_blocks
        self.merge_gap = merge_gap
        self.context_lines = context_lines

    def consolidate(
        self,
        spans: List[CodeSpan],
        fetch_text: Callable[[CodeSpan], str],
        token_budget: int,
    ) -> List[PagerBlock]:
        """Merge *spans*, pick top files, fetch text within *token_budget*.

        Returns at most ``self.max_blocks`` :class:`PagerBlock` instances.
        """
        if not spans:
            return []

        # Group by file, count frequency for ranking
        by_file: Dict[str, List[CodeSpan]] = defaultdict(list)
        freq: Counter[str] = Counter()
        for sp in spans:
            by_file[sp.file_path].append(sp)
            freq[sp.file_path] += 1

        # Rank files: most-paged first
        ranked_files = [f for f, _ in freq.most_common()]

        blocks: List[PagerBlock] = []
        budget_remaining = token_budget

        for fpath in ranked_files:
            if len(blocks) >= self.max_blocks:
                break
            if budget_remaining <= 0:
                break

            merged = self._merge_spans(by_file[fpath])
            # Pick the largest merged span for this file
            merged.sort(key=lambda s: s.length, reverse=True)
            best = merged[0]

            # Expand with context
            expanded = CodeSpan(
                file_path=best.file_path,
                start_line=max(1, best.start_line - self.context_lines),
                end_line=best.end_line + self.context_lines,
            )

            content = fetch_text(expanded)
            tok = _estimate_tokens(content)

            if tok > budget_remaining:
                # Truncate to fit budget
                max_chars = budget_remaining * CHARS_PER_TOKEN
                content = _truncate_chars(content, max_chars)
                tok = _estimate_tokens(content)

            blocks.append(PagerBlock(span=expanded, content=content, source="pager"))
            budget_remaining -= tok

        return blocks

    def _merge_spans(self, spans: List[CodeSpan]) -> List[CodeSpan]:
        """Union-merge spans for a single file, closing gaps <= merge_gap."""
        if not spans:
            return []
        fpath = spans[0].file_path
        intervals = sorted((s.start_line, s.end_line) for s in spans)
        merged: List[Tuple[int, int]] = [intervals[0]]
        for s, e in intervals[1:]:
            prev_s, prev_e = merged[-1]
            if s <= prev_e + self.merge_gap + 1:
                merged[-1] = (prev_s, max(prev_e, e))
            else:
                merged.append((s, e))
        return [CodeSpan(fpath, s, e) for s, e in merged]


# ===================================================================
# 3) AttentionSink
# ===================================================================
class AttentionSink:
    """Maintains a structured anchor rendered at the top of every observation.

    Content is append-only for revision_deltas; core fields may be updated.
    """

    def __init__(
        self,
        bug_anchor: Optional[str] = None,
        core_focus: Optional[str] = None,
    ) -> None:
        self.bug_anchor: str = bug_anchor or "(no bug anchor yet)"
        self.core_focus: str = core_focus or "(no focus)"
        self.invariants: List[str] = []
        self.revision_deltas: List[str] = []

    def update_focus(self, core_focus: str) -> None:
        """Update the current core focus identifier."""
        self.core_focus = core_focus

    def add_invariant(self, inv: str) -> None:
        """Append an invariant (idempotent by value)."""
        if inv not in self.invariants:
            self.invariants.append(inv)

    def add_revision(self, rev: str) -> None:
        """Append a revision delta (hypothesis correction). Never removed."""
        self.revision_deltas.append(rev)

    def render(self) -> str:
        """Produce the fixed-format attention-sink text block."""
        parts = [
            "┌─ ATTENTION SINK ──────────────────────────────────",
            f"│ bug_anchor : {self.bug_anchor}",
            f"│ core_focus : {self.core_focus}",
        ]
        if self.invariants:
            parts.append("│ invariants :")
            for inv in self.invariants:
                parts.append(f"│   • {inv}")
        if self.revision_deltas:
            parts.append("│ revisions  :")
            for i, rev in enumerate(self.revision_deltas, 1):
                parts.append(f"│   [{i}] {rev}")
        parts.append("└──────────────────────────────────────────────────")
        return "\n".join(parts)


# ===================================================================
# 4) EvidenceBuilder
# ===================================================================
class EvidenceBuilder:
    """Extract top-K evidence blocks from retrieval results."""

    def __init__(
        self,
        top_k: int = DEFAULT_EVIDENCE_TOP_K,
        max_lines: int = DEFAULT_EVIDENCE_MAX_LINES,
        max_chars: int = DEFAULT_MAX_CHARS_PER_BLOCK,
    ) -> None:
        self.top_k = top_k
        self.max_lines = max_lines
        self.max_chars = max_chars

    def build(
        self,
        retrieval_items: List[Dict[str, Any]],
        fetch_text: Callable[[CodeSpan], str],
    ) -> List[EvidenceBlock]:
        """Return at most *top_k* evidence blocks, truncated."""
        # Sort descending by score
        ranked = sorted(retrieval_items, key=lambda x: x.get("score", 0.0), reverse=True)
        blocks: List[EvidenceBlock] = []

        for item in ranked[: self.top_k]:
            symbol = item.get("symbol", "?")
            score = float(item.get("score", 0.0))
            span_raw = item.get("span")
            snippet = item.get("text_snippet", "")
            span: Optional[CodeSpan] = None

            if span_raw and isinstance(span_raw, dict):
                span = CodeSpan(
                    file_path=span_raw["file_path"],
                    start_line=span_raw.get("start_line", 1),
                    end_line=span_raw.get("end_line", 1),
                )

            content = snippet
            if not content and span:
                content = fetch_text(span)

            content = _truncate_lines(content, self.max_lines)
            content = _truncate_chars(content, self.max_chars)

            blocks.append(EvidenceBlock(symbol=symbol, score=score, span=span, content=content))

        return blocks


# ===================================================================
# 5–6) DeterministicInfiller (Atlas-optional)
# ===================================================================
class DeterministicInfiller:
    """2-hop deterministic callee expansion.  Requires Atlas; no-ops without it."""

    def __init__(
        self,
        max_fanout: int = DEFAULT_MAX_FANOUT_PER_NODE,
        max_funcs: int = DEFAULT_MAX_FUNCS_TOTAL,
        hub_score_high: float = DEFAULT_HUB_SCORE_HIGH,
        hub_sim_threshold: float = DEFAULT_HUB_FILTER_SIM_THRESHOLD,
    ) -> None:
        self.max_fanout = max_fanout
        self.max_funcs = max_funcs
        self.hub_score_high = hub_score_high
        self.hub_sim_threshold = hub_sim_threshold

    def expand(
        self,
        seeds: List[str],
        atlas: Optional[Atlas],
        query: str,
        bug_anchor: str,
        fetch_text: Callable[[CodeSpan], str],
        token_budget: int,
        high_uncertainty: bool = False,
    ) -> List[InfillerBlock]:
        """Expand seed func_ids via 1-hop (and optionally 2-hop) callees.

        Returns :class:`InfillerBlock` list, trimmed to budget.
        """
        if atlas is None or not seeds:
            return []

        context_text = f"{query} {bug_anchor}".strip()
        collected: List[Tuple[str, int]] = []  # (func_id, hop)
        seen: set[str] = set(seeds)

        for seed in seeds:
            callees_1 = atlas.callees(seed)[:self.max_fanout]
            for c in callees_1:
                if c not in seen:
                    if not self._hub_filter(c, atlas, context_text):
                        seen.add(c)
                        collected.append((c, 1))

            if high_uncertainty:
                for c in callees_1:
                    callees_2 = atlas.callees(c)[:self.max_fanout]
                    for c2 in callees_2:
                        if c2 not in seen:
                            if not self._hub_filter(c2, atlas, context_text):
                                seen.add(c2)
                                collected.append((c2, 2))

        # Sort: non-hub first, then by centrality desc, entropy desc
        def _sort_key(item: Tuple[str, int]) -> Tuple[float, float, float]:
            fid, _ = item
            ns = atlas.node_stats(fid)
            is_hub = 1.0 if ns.get("is_hub", False) else 0.0
            centrality = ns.get("centrality", 0.0)
            entropy = ns.get("structural_entropy", 0.0)
            sim = _similarity(fid, context_text)
            return (is_hub, -centrality - entropy, -sim)

        collected.sort(key=_sort_key)
        collected = collected[: self.max_funcs]

        # Build blocks within budget
        blocks: List[InfillerBlock] = []
        budget_remaining = token_budget
        for func_id, hop in collected:
            if budget_remaining <= 0:
                break
            span = atlas.func_span(func_id)
            content = ""
            if span:
                content = fetch_text(span)
                content = _truncate_lines(content, DEFAULT_MAX_LINES_PER_BLOCK)
                tok = _estimate_tokens(content)
                if tok > budget_remaining:
                    content = _truncate_chars(content, budget_remaining * CHARS_PER_TOKEN)
                    tok = _estimate_tokens(content)
                budget_remaining -= tok
            blocks.append(InfillerBlock(func_id=func_id, span=span, content=content, hop=hop))

        return blocks

    def _hub_filter(self, func_id: str, atlas: Atlas, context_text: str) -> bool:
        """Return True if *func_id* should be **skipped** (hub with low relevance)."""
        ns = atlas.node_stats(func_id)
        is_hub = ns.get("is_hub", False)
        hs = ns.get("hub_score", 0.0)
        if is_hub or hs >= self.hub_score_high:
            sim = _similarity(func_id, context_text)
            if sim < self.hub_sim_threshold:
                return True
        return False


# ===================================================================
# 7) ControlledJitter
# ===================================================================
class ControlledJitter:
    """Strictly controlled jitter sampling from side candidates only.

    Jitter is deterministic given (instance_id, step_id, query, core_focus_id).
    """

    def __init__(
        self,
        temperature: float = DEFAULT_JITTER_TEMPERATURE,
        token_cap_ratio: float = DEFAULT_JITTER_TOKEN_CAP_RATIO,
    ) -> None:
        self.temperature = temperature
        self.token_cap_ratio = token_cap_ratio

    def sample(
        self,
        seeds: List[str],
        atlas: Optional[Atlas],
        instance_id: str,
        step_id: int,
        query: str,
        core_focus_id: Optional[str],
        jitter_ratio: float,
        prefetch_budget: int,
        fetch_text: Callable[[CodeSpan], str],
        already_used: set[str],
    ) -> List[JitterBlock]:
        """Sample side candidates with softmax weighting.

        Returns :class:`JitterBlock` list within the jitter token cap.
        """
        if atlas is None or jitter_ratio <= 0 or not seeds:
            return []

        jitter_token_cap = int(prefetch_budget * min(self.token_cap_ratio, 0.25))
        if jitter_token_cap <= 0:
            return []

        # Collect side candidates: data_neighbors + 2-hop non-callees
        candidates: List[str] = []
        for seed in seeds:
            for dn in atlas.data_neighbors(seed):
                if dn not in already_used:
                    candidates.append(dn)
            for c in atlas.callees(seed):
                for c2 in atlas.callees(c):
                    if c2 not in already_used and c2 not in seeds:
                        candidates.append(c2)

        if not candidates:
            return []

        # Deduplicate while preserving order
        seen: set[str] = set()
        unique: List[str] = []
        for c in candidates:
            if c not in seen:
                seen.add(c)
                unique.append(c)
        candidates = unique

        # Deterministic seed
        rng_seed = _stable_hash(instance_id, step_id, query, core_focus_id or "")
        rng = random.Random(rng_seed)

        # Softmax scoring
        context = f"{query} {core_focus_id or ''}".strip()
        raw_scores = [_similarity(c, context) for c in candidates]
        if self.temperature > 0:
            max_s = max(raw_scores) if raw_scores else 0.0
            exp_scores = [math.exp((s - max_s) / self.temperature) for s in raw_scores]
            total = sum(exp_scores)
            if total > 0:
                weights = [e / total for e in exp_scores]
            else:
                weights = [1.0 / len(candidates)] * len(candidates)
        else:
            weights = [1.0 / len(candidates)] * len(candidates)

        # How many to sample
        n_sample = max(1, int(len(candidates) * jitter_ratio))
        n_sample = min(n_sample, len(candidates))

        # Weighted sampling without replacement
        chosen: List[str] = []
        pool = list(range(len(candidates)))
        pool_weights = list(weights)
        for _ in range(n_sample):
            if not pool:
                break
            total_w = sum(pool_weights)
            if total_w <= 0:
                break
            r = rng.random() * total_w
            cumulative = 0.0
            idx_in_pool = 0
            for j, w in enumerate(pool_weights):
                cumulative += w
                if cumulative >= r:
                    idx_in_pool = j
                    break
            chosen.append(candidates[pool[idx_in_pool]])
            pool.pop(idx_in_pool)
            pool_weights.pop(idx_in_pool)

        # Build blocks within cap
        blocks: List[JitterBlock] = []
        budget_remaining = jitter_token_cap
        for func_id in chosen:
            if budget_remaining <= 0:
                break
            span = atlas.func_span(func_id)
            content = ""
            if span:
                content = fetch_text(span)
                content = _truncate_lines(content, DEFAULT_MAX_LINES_PER_BLOCK)
                tok = _estimate_tokens(content)
                if tok > budget_remaining:
                    content = _truncate_chars(content, budget_remaining * CHARS_PER_TOKEN)
                    tok = _estimate_tokens(content)
                budget_remaining -= tok
            blocks.append(JitterBlock(func_id=func_id, span=span, content=content))

        return blocks


# ===================================================================
# 8) GlimpGate / BudgetScheduler
# ===================================================================
class GlimpGate:
    """Adaptive budget & jitter scheduler based on uncertainty estimation.

    Computes an aggression_level (0–3) that governs how much prefetch content
    and jitter to inject.
    """

    def __init__(
        self,
        base_budget: int = DEFAULT_PREFETCH_BUDGET_TOKENS,
        w_flatness: float = 0.35,
        w_coverage: float = 0.25,
        w_stagnation: float = 0.25,
        w_repeat: float = 0.15,
        verified_mode: bool = True,
    ) -> None:
        self.base_budget = base_budget
        self.w_flatness = w_flatness
        self.w_coverage = w_coverage
        self.w_stagnation = w_stagnation
        self.w_repeat = w_repeat
        self.verified_mode = verified_mode

    def decide(
        self,
        retrieval_items: List[Dict[str, Any]],
        stack_coverage: float = 0.0,
        stagnation_rounds: int = 0,
        query_repeat_rate: float = 0.0,
    ) -> BudgetState:
        """Compute budget allocation for this round.

        Returns a :class:`BudgetState` with logs.
        """
        # Flatness
        if not retrieval_items:
            flatness = 1.0
        else:
            scores = sorted(
                [float(it.get("score", 0.0)) for it in retrieval_items], reverse=True
            )
            if len(scores) >= 5 and scores[4] > 0:
                flatness = 1.0 - min(1.0, scores[0] / scores[4] - 1.0)
            elif len(scores) >= 2 and scores[1] > 0:
                flatness = 1.0 - min(1.0, scores[0] / scores[1] - 1.0)
            else:
                flatness = 0.5

        flatness = max(0.0, min(1.0, flatness))
        stagnation = min(1.0, stagnation_rounds / 5.0)
        repeat = min(1.0, query_repeat_rate)
        coverage = max(0.0, min(1.0, stack_coverage))

        uncertainty = (
            self.w_flatness * flatness
            + self.w_coverage * (1.0 - coverage)
            + self.w_stagnation * stagnation
            + self.w_repeat * repeat
        )
        uncertainty = max(0.0, min(1.0, uncertainty))

        # Aggression level (0=conservative, 3=aggressive)
        if uncertainty < 0.25:
            agg = 0
        elif uncertainty < 0.50:
            agg = 1
        elif uncertainty < 0.75:
            agg = 2
        else:
            agg = 3

        # Budget scaling
        budget_multipliers = {0: 0.6, 1: 0.8, 2: 1.0, 3: 1.2}
        budget = int(self.base_budget * budget_multipliers[agg])

        # Jitter ratio (Verified = conservative)
        if self.verified_mode:
            jitter_map = {0: 0.0, 1: 0.0, 2: 0.05, 3: 0.10}
        else:
            jitter_map = {0: 0.0, 1: 0.05, 2: 0.15, 3: 0.25}
        jitter = jitter_map[agg]

        return BudgetState(
            uncertainty_score=uncertainty,
            flatness=flatness,
            coverage=coverage,
            stagnation=stagnation,
            repeat=repeat,
            aggression_level=agg,
            token_budget_prefetch=budget,
            jitter_ratio=jitter,
        )


# ===================================================================
# 9) ShadowEvaluator
# ===================================================================
class ShadowEvaluator:
    """Per-round hit@topM shadow evaluation with shrink state machine.

    Tracks predictions vs actual accesses each round.  After *shrink_window*
    consecutive zero-hit rounds the budget is shrunk and jitter disabled.
    """

    def __init__(
        self,
        top_m: int = DEFAULT_SHADOW_TOP_M,
        shrink_window: int = DEFAULT_SHRINK_WINDOW,
        shrink_factor: float = DEFAULT_SHRINK_FACTOR,
    ) -> None:
        self.top_m = top_m
        self.shrink_window = shrink_window
        self.shrink_factor = shrink_factor
        self.history: List[ShadowRecord] = []
        self.consecutive_misses: int = 0
        self.budget_multiplier: float = 1.0
        self.jitter_suppressed_rounds: int = 0

    def register_predictions(
        self,
        step_id: int,
        plan: PrefetchPlan,
    ) -> None:
        """Record predictions for this round (before observing actuals)."""
        preds: List[str] = []

        # Collect identifiers from plan blocks
        for b in plan.pager_blocks:
            preds.append(b.span.file_path)
        for b in plan.evidence_blocks:
            preds.append(b.symbol)
            if b.span:
                preds.append(b.span.file_path)
        for b in plan.det_blocks:
            preds.append(b.func_id)
            if b.span:
                preds.append(b.span.file_path)
        for b in plan.jitter_blocks:
            preds.append(b.func_id)
            if b.span:
                preds.append(b.span.file_path)

        # Deduplicate, keep top_m
        seen: set[str] = set()
        unique: List[str] = []
        for p in preds:
            if p not in seen:
                seen.add(p)
                unique.append(p)
        top_m_preds = unique[: self.top_m]

        rec = ShadowRecord(
            step_id=step_id,
            predictions_top_m=top_m_preds,
            det_count=len(plan.det_blocks),
            jit_count=len(plan.jitter_blocks),
        )
        self.history.append(rec)

    def evaluate(
        self,
        actual_accesses: List[str],
        hit_modes: Sequence[str] = ("symbol", "file", "substring"),
    ) -> Dict[str, Any]:
        """Score the latest round against *actual_accesses*.

        Updates shrink state machine. Returns logs dict.
        """
        if not self.history:
            return {"error": "no predictions registered"}

        rec = self.history[-1]
        rec.actual_accesses = list(actual_accesses)

        # Compute hits
        hit_set: set[str] = set()
        preds_lower = {p.lower() for p in rec.predictions_top_m}
        actuals_lower = {a.lower() for a in actual_accesses}

        for pred in rec.predictions_top_m:
            pl = pred.lower()
            for actual in actual_accesses:
                al = actual.lower()
                matched = False
                if "symbol" in hit_modes and pl == al:
                    matched = True
                if "file" in hit_modes and (
                    pl.endswith(al) or al.endswith(pl)
                    or pl.split("/")[-1] == al.split("/")[-1]
                ):
                    matched = True
                if "substring" in hit_modes and (pl in al or al in pl):
                    matched = True
                if matched:
                    hit_set.add(pred)

        rec.hit_at_top_m = len(hit_set) > 0
        rec.unique_hits = len(hit_set)

        # Shrink state machine
        if rec.hit_at_top_m:
            self.consecutive_misses = 0
            # Gradual recovery
            if self.budget_multiplier < 1.0:
                self.budget_multiplier = min(1.0, self.budget_multiplier + 0.15)
            self.jitter_suppressed_rounds = 0
        else:
            self.consecutive_misses += 1
            if self.consecutive_misses >= self.shrink_window:
                self.budget_multiplier = max(0.3, self.budget_multiplier * self.shrink_factor)
                self.jitter_suppressed_rounds = 2

        if self.jitter_suppressed_rounds > 0:
            self.jitter_suppressed_rounds -= 1

        return {
            "step_id": rec.step_id,
            "predictions_top_m": rec.predictions_top_m,
            "actual_accesses": rec.actual_accesses,
            "hit_at_top_m": rec.hit_at_top_m,
            "unique_hits": rec.unique_hits,
            "consecutive_misses": self.consecutive_misses,
            "budget_multiplier": round(self.budget_multiplier, 4),
            "jitter_suppressed": self.jitter_suppressed_rounds > 0,
        }

    def apply_shrink(self, budget_state: BudgetState) -> BudgetState:
        """Apply shrink adjustments to a budget state in-place and return it."""
        budget_state.token_budget_prefetch = int(
            budget_state.token_budget_prefetch * self.budget_multiplier
        )
        if self.jitter_suppressed_rounds > 0:
            budget_state.jitter_ratio = 0.0
        return budget_state


# ===================================================================
# 10) ContextRenderer
# ===================================================================
class ContextRenderer:
    """Assemble the final observation string with deduplication and budget cap."""

    def __init__(
        self,
        max_chars_per_block: int = DEFAULT_MAX_CHARS_PER_BLOCK,
        max_lines_per_block: int = DEFAULT_MAX_LINES_PER_BLOCK,
    ) -> None:
        self.max_chars_per_block = max_chars_per_block
        self.max_lines_per_block = max_lines_per_block

    def render(
        self,
        attention_sink: str,
        original_result: str,
        plan: PrefetchPlan,
        token_budget: int,
    ) -> Tuple[str, Dict[str, Any]]:
        """Build the full observation and return (text, render_logs).

        Section order:
          [ATTENTION_SINK] → [CURRENT_RESULT] → [PREFETCH_PAGER]
          → [PREFETCH_LOGIC_FLOW][DET] → [PREFETCH_LOGIC_FLOW][JIT]

        Evidence blocks are folded into CURRENT_RESULT.
        """
        seen_keys: set[str] = set()  # dedup across sections
        sections: List[str] = []
        token_counts: Dict[str, int] = {
            "pager": 0, "evidence": 0, "det": 0, "jit": 0,
        }
        budget_remaining = token_budget

        # --- Attention Sink (always first, not counted against prefetch budget) ---
        sections.append(f"[ATTENTION_SINK]\n{attention_sink}")

        # --- Current Result + Evidence ---
        result_section = f"[CURRENT_RESULT]\n{original_result}"
        if plan.evidence_blocks:
            ev_parts: List[str] = []
            for eb in plan.evidence_blocks:
                key = eb.symbol
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                content = _truncate_lines(eb.content, self.max_lines_per_block)
                content = _truncate_chars(content, self.max_chars_per_block)
                loc = ""
                if eb.span:
                    loc = f" ({eb.span.file_path}:{eb.span.start_line}-{eb.span.end_line})"
                ev_parts.append(
                    f"  ── evidence: {eb.symbol} (score={eb.score:.3f}){loc}\n{content}"
                )
                token_counts["evidence"] += _estimate_tokens(content)
            if ev_parts:
                result_section += "\n" + "\n".join(ev_parts)
        sections.append(result_section)

        # --- Pager blocks (pager-first: allocated first) ---
        if plan.pager_blocks:
            pager_parts: List[str] = []
            for pb in plan.pager_blocks:
                key = f"pager:{pb.span.file_path}:{pb.span.start_line}"
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                content = _truncate_lines(pb.content, self.max_lines_per_block)
                content = _truncate_chars(content, self.max_chars_per_block)
                tok = _estimate_tokens(content)
                if tok > budget_remaining:
                    content = _truncate_chars(content, budget_remaining * CHARS_PER_TOKEN)
                    tok = _estimate_tokens(content)
                budget_remaining -= tok
                token_counts["pager"] += tok
                loc = f"{pb.span.file_path}:{pb.span.start_line}-{pb.span.end_line}"
                pager_parts.append(f"  ── pager block: {loc}\n{content}")
            if pager_parts:
                sections.append("[PREFETCH_PAGER]\n" + "\n".join(pager_parts))

        # --- Deterministic infiller ---
        if plan.det_blocks:
            det_parts: List[str] = []
            for db in plan.det_blocks:
                if db.func_id in seen_keys:
                    continue
                seen_keys.add(db.func_id)
                content = _truncate_lines(db.content, self.max_lines_per_block)
                content = _truncate_chars(content, self.max_chars_per_block)
                tok = _estimate_tokens(content)
                if tok > budget_remaining:
                    content = _truncate_chars(content, budget_remaining * CHARS_PER_TOKEN)
                    tok = _estimate_tokens(content)
                budget_remaining -= tok
                token_counts["det"] += tok
                loc = ""
                if db.span:
                    loc = f" ({db.span.file_path}:{db.span.start_line}-{db.span.end_line})"
                det_parts.append(
                    f"  ── det[hop{db.hop}]: {db.func_id}{loc}\n{content}"
                )
            if det_parts:
                sections.append("[PREFETCH_LOGIC_FLOW][DET]\n" + "\n".join(det_parts))

        # --- Jitter ---
        if plan.jitter_blocks:
            jit_parts: List[str] = []
            for jb in plan.jitter_blocks:
                if jb.func_id in seen_keys:
                    continue
                seen_keys.add(jb.func_id)
                content = _truncate_lines(jb.content, self.max_lines_per_block)
                content = _truncate_chars(content, self.max_chars_per_block)
                tok = _estimate_tokens(content)
                if tok > budget_remaining:
                    content = _truncate_chars(content, budget_remaining * CHARS_PER_TOKEN)
                    tok = _estimate_tokens(content)
                budget_remaining -= tok
                token_counts["jit"] += tok
                loc = ""
                if jb.span:
                    loc = f" ({jb.span.file_path}:{jb.span.start_line}-{jb.span.end_line})"
                jit_parts.append(f"  ── jit: {jb.func_id}{loc}\n{content}")
            if jit_parts:
                sections.append("[PREFETCH_LOGIC_FLOW][JIT]\n" + "\n".join(jit_parts))

        total_tokens = sum(token_counts.values())
        render_logs = {
            "pager_blocks_count": len(plan.pager_blocks),
            "evidence_blocks_count": len(plan.evidence_blocks),
            "det_blocks_count": len(plan.det_blocks),
            "jit_blocks_count": len(plan.jitter_blocks),
            "estimated_tokens_total": total_tokens,
            "estimated_tokens_pager": token_counts["pager"],
            "estimated_tokens_evidence": token_counts["evidence"],
            "estimated_tokens_det": token_counts["det"],
            "estimated_tokens_jit": token_counts["jit"],
        }

        return "\n\n".join(sections), render_logs


# ===================================================================
# 11) PrefetchCoordinator — top-level API
# ===================================================================
class PrefetchCoordinator:
    """Orchestrates all prefetch sub-modules for one agent session.

    Public API
    ----------
    - ``decide_budget(...)``      → (BudgetState, logs)
    - ``build_plan(...)``         → PrefetchPlan
    - ``wrap_observation(...)``   → str
    - ``update_shadow(...)``      → shadow_logs
    """

    def __init__(
        self,
        atlas: Optional[Atlas] = None,
        base_budget: int = DEFAULT_PREFETCH_BUDGET_TOKENS,
        verified_mode: bool = True,
        max_pager_blocks: int = DEFAULT_MAX_PAGER_BLOCKS,
        merge_gap: int = DEFAULT_MERGE_GAP,
        context_lines: int = DEFAULT_CONTEXT_LINES,
        evidence_top_k: int = DEFAULT_EVIDENCE_TOP_K,
        shadow_top_m: int = DEFAULT_SHADOW_TOP_M,
    ) -> None:
        self.atlas = atlas
        self.gate = GlimpGate(base_budget=base_budget, verified_mode=verified_mode)
        self.pager_parser = CommandPagerParser()
        self.pager_consolidator = PagerConsolidator(
            max_blocks=max_pager_blocks,
            merge_gap=merge_gap,
            context_lines=context_lines,
        )
        self.evidence_builder = EvidenceBuilder(top_k=evidence_top_k)
        self.infiller = DeterministicInfiller()
        self.jitter = ControlledJitter()
        self.shadow = ShadowEvaluator(top_m=shadow_top_m)
        self.renderer = ContextRenderer()
        self.attention_sink = AttentionSink()
        self._last_budget: Optional[BudgetState] = None

    # ---- 1. decide_budget ----
    def decide_budget(
        self,
        retrieval_items: List[Dict[str, Any]],
        stack_coverage: float = 0.0,
        stagnation_rounds: int = 0,
        query_repeat_rate: float = 0.0,
    ) -> Tuple[BudgetState, Dict[str, Any]]:
        """Compute budget for this round, incorporating shadow shrink.

        Returns (BudgetState, logs_dict).
        """
        bs = self.gate.decide(
            retrieval_items=retrieval_items,
            stack_coverage=stack_coverage,
            stagnation_rounds=stagnation_rounds,
            query_repeat_rate=query_repeat_rate,
        )
        bs = self.shadow.apply_shrink(bs)
        self._last_budget = bs
        return bs, bs.to_dict()

    # ---- 2. build_plan ----
    def build_plan(
        self,
        query: str,
        instance_id: str,
        step_id: int,
        retrieval_items: List[Dict[str, Any]],
        bug_anchor_text: Optional[str],
        core_focus_id: Optional[str],
        paging_history_cmds: List[str],
        fetch_text: Callable[[CodeSpan], str],
        budget_state: Optional[BudgetState] = None,
    ) -> PrefetchPlan:
        """Build the complete prefetch plan for one agent round.

        Uses pager-first priority: pager blocks get budget before det/jit.
        """
        bs = budget_state or self._last_budget or BudgetState()
        total_budget = bs.token_budget_prefetch

        # Update attention sink
        if bug_anchor_text:
            self.attention_sink.bug_anchor = bug_anchor_text
        if core_focus_id:
            self.attention_sink.update_focus(core_focus_id)

        # 1. Parse paging history → pager blocks (PRIORITY)
        pager_spans = self.pager_parser.parse_many(paging_history_cmds)
        pager_budget = int(total_budget * 0.5)  # reserve up to 50% for pager
        pager_blocks = self.pager_consolidator.consolidate(
            pager_spans, fetch_text, pager_budget,
        )
        pager_tokens_used = sum(_estimate_tokens(b.content) for b in pager_blocks)

        # 2. Evidence
        evidence_blocks = self.evidence_builder.build(retrieval_items, fetch_text)
        evidence_tokens = sum(_estimate_tokens(b.content) for b in evidence_blocks)

        # 3. Remaining budget for det + jit
        remaining = max(0, total_budget - pager_tokens_used - evidence_tokens)

        # Collect already-used identifiers for dedup
        used: set[str] = set()
        for pb in pager_blocks:
            used.add(pb.span.file_path)
        for eb in evidence_blocks:
            used.add(eb.symbol)

        # 4. Deterministic infiller
        seeds = [core_focus_id] if core_focus_id else []
        jitter_ratio = bs.jitter_ratio
        jitter_budget = int(remaining * min(jitter_ratio, 0.3)) if jitter_ratio > 0 else 0
        det_budget = remaining - jitter_budget

        high_uncertainty = bs.uncertainty_score >= 0.6
        det_blocks = self.infiller.expand(
            seeds=seeds,
            atlas=self.atlas,
            query=query,
            bug_anchor=bug_anchor_text or "",
            fetch_text=fetch_text,
            token_budget=det_budget,
            high_uncertainty=high_uncertainty,
        )
        det_tokens = sum(_estimate_tokens(b.content) for b in det_blocks)

        for db in det_blocks:
            used.add(db.func_id)

        # 5. Jitter
        jitter_blocks = self.jitter.sample(
            seeds=seeds,
            atlas=self.atlas,
            instance_id=instance_id,
            step_id=step_id,
            query=query,
            core_focus_id=core_focus_id,
            jitter_ratio=jitter_ratio,
            prefetch_budget=total_budget,
            fetch_text=fetch_text,
            already_used=used,
        )

        plan = PrefetchPlan(
            pager_blocks=pager_blocks,
            evidence_blocks=evidence_blocks,
            det_blocks=det_blocks,
            jitter_blocks=jitter_blocks,
            budget_state=bs,
            meta={
                "pager_spans_parsed": len(pager_spans),
                "pager_tokens": pager_tokens_used,
                "evidence_tokens": evidence_tokens,
                "det_tokens": det_tokens,
                "jitter_tokens": sum(_estimate_tokens(b.content) for b in jitter_blocks),
                "total_budget": total_budget,
                "high_uncertainty": high_uncertainty,
            },
        )

        # Register predictions for shadow
        self.shadow.register_predictions(step_id, plan)
        return plan

    # ---- 3. wrap_observation ----
    def wrap_observation(
        self,
        original_result: str,
        plan: PrefetchPlan,
    ) -> str:
        """Assemble the full agent observation with prefetch context.

        Returns the rendered observation string.
        """
        bs = plan.budget_state or BudgetState()
        sink_text = self.attention_sink.render()
        obs, render_logs = self.renderer.render(
            attention_sink=sink_text,
            original_result=original_result,
            plan=plan,
            token_budget=bs.token_budget_prefetch,
        )
        logger.debug("render_logs=%s", json.dumps(render_logs))
        return obs

    # ---- 4. update_shadow ----
    def update_shadow(
        self,
        actual_accesses: List[str],
    ) -> Dict[str, Any]:
        """Evaluate predictions against actual accesses. May trigger shrink.

        Returns shadow evaluation logs.
        """
        return self.shadow.evaluate(actual_accesses)


# ===================================================================
# DEMO
# ===================================================================
def _demo() -> None:
    """Minimal 3-round demo exercising all components.

    Constructs in-memory file contents, fake retrieval items,
    paging history commands, and an optional FakeAtlas.
    """
    logging.basicConfig(level=logging.DEBUG, format="%(name)s %(levelname)s: %(message)s")

    # ---- In-memory file store ----
    FILE_STORE: Dict[str, List[str]] = {
        "django/db/models/query.py": [
            f"# line {i}: QuerySet implementation" for i in range(1, 601)
        ],
        "django/db/models/sql/compiler.py": [
            f"# line {i}: SQL compiler logic" for i in range(1, 401)
        ],
        "django/db/models/fields/__init__.py": [
            f"# line {i}: Field base classes" for i in range(1, 301)
        ],
        "tests/queries/test_qs.py": [
            f"# line {i}: test case" for i in range(1, 201)
        ],
    }

    def fetch_text(span: CodeSpan) -> str:
        lines = FILE_STORE.get(span.file_path, [])
        start = max(0, span.start_line - 1)
        end = min(len(lines), span.end_line)
        return "\n".join(lines[start:end])

    # ---- Fake Atlas ----
    class FakeAtlas:
        _graph: Dict[str, List[str]] = {
            "QuerySet.filter": ["QuerySet._filter_or_exclude", "Q.__init__"],
            "QuerySet._filter_or_exclude": ["SQLCompiler.compile", "WhereNode.add"],
            "SQLCompiler.compile": ["SQLCompiler.as_sql"],
            "SQLCompiler.as_sql": [],
            "Q.__init__": [],
            "WhereNode.add": ["WhereNode.resolve_expression"],
            "WhereNode.resolve_expression": [],
        }
        _data_neighbors: Dict[str, List[str]] = {
            "QuerySet.filter": ["QuerySet.exclude", "Manager.get_queryset"],
            "SQLCompiler.compile": ["SQLCompiler.execute_sql"],
        }
        _spans: Dict[str, CodeSpan] = {
            "QuerySet.filter": CodeSpan("django/db/models/query.py", 200, 240),
            "QuerySet._filter_or_exclude": CodeSpan("django/db/models/query.py", 250, 310),
            "SQLCompiler.compile": CodeSpan("django/db/models/sql/compiler.py", 50, 110),
            "SQLCompiler.as_sql": CodeSpan("django/db/models/sql/compiler.py", 120, 180),
            "Q.__init__": CodeSpan("django/db/models/query.py", 10, 30),
            "WhereNode.add": CodeSpan("django/db/models/sql/compiler.py", 200, 240),
            "WhereNode.resolve_expression": CodeSpan("django/db/models/sql/compiler.py", 250, 300),
            "QuerySet.exclude": CodeSpan("django/db/models/query.py", 245, 250),
            "Manager.get_queryset": CodeSpan("django/db/models/query.py", 500, 520),
            "SQLCompiler.execute_sql": CodeSpan("django/db/models/sql/compiler.py", 300, 350),
        }
        _stats: Dict[str, Dict[str, Any]] = {
            "QuerySet.filter": {
                "hub_score": 0.9, "centrality": 0.8,
                "structural_entropy": 0.6, "is_hub": True,
            },
            "QuerySet._filter_or_exclude": {
                "hub_score": 0.3, "centrality": 0.5,
                "structural_entropy": 0.7, "is_hub": False,
            },
            "SQLCompiler.compile": {
                "hub_score": 0.4, "centrality": 0.6,
                "structural_entropy": 0.5, "is_hub": False,
            },
            "SQLCompiler.as_sql": {
                "hub_score": 0.2, "centrality": 0.3,
                "structural_entropy": 0.4, "is_hub": False,
            },
            "Q.__init__": {
                "hub_score": 0.1, "centrality": 0.2,
                "structural_entropy": 0.3, "is_hub": False,
            },
            "WhereNode.add": {
                "hub_score": 0.3, "centrality": 0.4,
                "structural_entropy": 0.5, "is_hub": False,
            },
        }
        _default_stats: Dict[str, Any] = {
            "hub_score": 0.0, "centrality": 0.0,
            "structural_entropy": 0.0, "is_hub": False,
        }

        def callees(self, func_id: str) -> List[str]:
            return self._graph.get(func_id, [])

        def data_neighbors(self, func_id: str) -> List[str]:
            return self._data_neighbors.get(func_id, [])

        def func_span(self, func_id: str) -> Optional[CodeSpan]:
            return self._spans.get(func_id)

        def node_stats(self, func_id: str) -> Dict[str, Any]:
            return self._stats.get(func_id, dict(self._default_stats))

    atlas = FakeAtlas()

    # ---- Round definitions ----
    rounds = [
        {
            "step_id": 1,
            "query": "filter queryset duplicate results",
            "retrieval_items": [
                {
                    "symbol": "QuerySet.filter",
                    "score": 0.92,
                    "span": {"file_path": "django/db/models/query.py",
                             "start_line": 200, "end_line": 240},
                    "text_snippet": "def filter(self, *args, **kwargs):\n    return self._filter_or_exclude(False, *args, **kwargs)\n",
                },
                {
                    "symbol": "QuerySet.distinct",
                    "score": 0.78,
                    "span": {"file_path": "django/db/models/query.py",
                             "start_line": 350, "end_line": 380},
                    "text_snippet": "def distinct(self, *field_names):\n    ...\n",
                },
                {
                    "symbol": "Q.__init__",
                    "score": 0.55,
                    "span": {"file_path": "django/db/models/query.py",
                             "start_line": 10, "end_line": 30},
                },
            ],
            "bug_anchor_text": "AssertionError in test_filter_duplicates: expected 3, got 6",
            "core_focus_id": "QuerySet.filter",
            "paging_history_cmds": [
                "nl -ba django/db/models/query.py | sed -n '200,260p'",
                "nl -ba django/db/models/query.py | sed -n '250,320p'",
                "nl -ba django/db/models/sql/compiler.py | sed -n '50,120p'",
            ],
            "actual_accesses": [
                "QuerySet._filter_or_exclude",
                "django/db/models/query.py",
            ],
            "stack_coverage": 0.3,
            "stagnation": 0,
            "repeat": 0.0,
        },
        {
            "step_id": 2,
            "query": "SQL compiler compile method",
            "retrieval_items": [
                {
                    "symbol": "SQLCompiler.compile",
                    "score": 0.88,
                    "span": {"file_path": "django/db/models/sql/compiler.py",
                             "start_line": 50, "end_line": 110},
                    "text_snippet": "def compile(self, node):\n    ...\n",
                },
            ],
            "bug_anchor_text": "AssertionError in test_filter_duplicates: expected 3, got 6",
            "core_focus_id": "SQLCompiler.compile",
            "paging_history_cmds": [
                "nl -ba django/db/models/sql/compiler.py | sed -n '100,200p'",
            ],
            "actual_accesses": [
                "some_unrelated_file.py",  # deliberate miss
            ],
            "stack_coverage": 0.5,
            "stagnation": 1,
            "repeat": 0.2,
        },
        {
            "step_id": 3,
            "query": "SQL compiler compile method",  # repeated query
            "retrieval_items": [],  # empty retrieval → high flatness
            "bug_anchor_text": "AssertionError in test_filter_duplicates: expected 3, got 6",
            "core_focus_id": "SQLCompiler.compile",
            "paging_history_cmds": [],
            "actual_accesses": [
                "another_miss.py",  # miss again → triggers shrink
            ],
            "stack_coverage": 0.5,
            "stagnation": 2,
            "repeat": 0.8,
        },
    ]

    # ---- Run coordinator ----
    coord = PrefetchCoordinator(atlas=atlas, base_budget=4000, verified_mode=True)

    print("=" * 72)
    print("  LSR-Engine v2 — 3-Round Demo")
    print("=" * 72)

    for rnd in rounds:
        step = rnd["step_id"]
        print(f"\n{'─' * 72}")
        print(f"  ROUND {step}  query={rnd['query']!r}")
        print(f"{'─' * 72}")

        # 1. Budget
        bs, budget_logs = coord.decide_budget(
            retrieval_items=rnd["retrieval_items"],
            stack_coverage=rnd["stack_coverage"],
            stagnation_rounds=rnd["stagnation"],
            query_repeat_rate=rnd["repeat"],
        )
        print(f"\n  [Budget] {json.dumps(budget_logs, indent=4)}")

        # 2. Plan
        plan = coord.build_plan(
            query=rnd["query"],
            instance_id="django__django-12345",
            step_id=step,
            retrieval_items=rnd["retrieval_items"],
            bug_anchor_text=rnd["bug_anchor_text"],
            core_focus_id=rnd["core_focus_id"],
            paging_history_cmds=rnd["paging_history_cmds"],
            fetch_text=fetch_text,
            budget_state=bs,
        )
        print(f"\n  [Plan meta] {json.dumps(plan.meta, indent=4)}")
        print(f"  [Plan] pager_spans  = {plan.pager_spans}")
        print(f"  [Plan] det_func_ids = {plan.det_func_ids}")
        print(f"  [Plan] jitter_ids   = {plan.jitter_func_ids}")
        print(f"  [Plan] evidence     = {[e.symbol for e in plan.evidence_blocks]}")

        # 3. Observation (show first 600 chars)
        obs = coord.wrap_observation(
            original_result=f"Search results for: {rnd['query']}",
            plan=plan,
        )
        print(f"\n  [Observation preview] ({len(obs)} chars total)")
        for line in obs[:800].splitlines():
            print(f"    {line}")
        if len(obs) > 800:
            print(f"    ... ({len(obs) - 800} more chars)")

        # 4. Shadow evaluation
        shadow_logs = coord.update_shadow(actual_accesses=rnd["actual_accesses"])
        print(f"\n  [Shadow] {json.dumps(shadow_logs, indent=4)}")

    # ---- No-Atlas demo (1 round) ----
    print(f"\n{'=' * 72}")
    print("  Atlas-None fallback (1 round)")
    print(f"{'=' * 72}")

    coord_no_atlas = PrefetchCoordinator(atlas=None, base_budget=3000)
    bs2, logs2 = coord_no_atlas.decide_budget(
        retrieval_items=[{"symbol": "foo", "score": 0.7}],
    )
    plan2 = coord_no_atlas.build_plan(
        query="find bug",
        instance_id="test__test-999",
        step_id=1,
        retrieval_items=[{
            "symbol": "foo",
            "score": 0.7,
            "text_snippet": "def foo():\n    return bar() + 1\n",
        }],
        bug_anchor_text="IndexError at line 42",
        core_focus_id=None,
        paging_history_cmds=[
            "nl -ba src/module.py | sed -n '30,80p'",
        ],
        fetch_text=fetch_text,
        budget_state=bs2,
    )
    obs2 = coord_no_atlas.wrap_observation("(raw result)", plan2)
    print(f"\n  [Budget] {json.dumps(logs2, indent=4)}")
    print(f"  [Plan] pager_spans={plan2.pager_spans}, det={plan2.det_func_ids}, "
          f"jit={plan2.jitter_func_ids}, evidence={[e.symbol for e in plan2.evidence_blocks]}")
    print(f"  [Obs preview] ({len(obs2)} chars)")
    for line in obs2[:600].splitlines():
        print(f"    {line}")

    print(f"\n{'=' * 72}")
    print("  Demo complete.")
    print(f"{'=' * 72}")


if __name__ == "__main__":
    _demo()

"""Prefetch-enabled agent using LSR-Engine v2.

This module provides PrefetchAgent, a DefaultAgent subclass that integrates
the LSR-Engine v2 prefetch system for reduced paging operations and improved
context management in SWE-bench style tasks.

Key features:
- Pager-first: Detects and consolidates nl|sed manual paging patterns
- Atlas-optional: Full functionality without call-graph data
- Per-instance isolation: No state leakage between instances
- Shadow evaluation: Tracks prediction accuracy with shrink state machine
"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING

from pydantic import BaseModel

from minisweagent.agents.default import DefaultAgent, AgentConfig
from minisweagent.prefetch.lsr_engine_v2 import (
    PrefetchCoordinator,
    CodeSpan,
    CommandPagerParser,
    BudgetState,
    PrefetchPlan,
)


class PrefetchAgentConfig(AgentConfig):
    """Extended configuration for prefetch-enabled agent."""

    prefetch_enabled: bool = True
    prefetch_base_budget: int = 4000
    prefetch_verified_mode: bool = True
    prefetch_max_pager_blocks: int = 2
    prefetch_evidence_top_k: int = 2
    prefetch_shadow_top_m: int = 5


class PrefetchAgent(DefaultAgent):
    """Agent with LSR-Engine v2 prefetch capabilities.

    Overrides get_observation() to inject prefetch context based on:
    - Detected manual paging patterns (nl|sed)
    - Retrieval-like output from grep/find commands
    - Progress state (stagnation, query repetition)

    All prefetch state is per-instance to prevent cross-instance leakage.
    """

    def __init__(
        self,
        model,
        env,
        *,
        instance_id: str = "",
        bug_anchor_text: str = "",
        file_content_store: Optional[Dict[str, str]] = None,
        progress_manager: Optional[Any] = None,  # P4: 兼容 batch mode
        config_class: type = PrefetchAgentConfig,
        **kwargs,
    ):
        """Initialize prefetch agent.

        Args:
            model: The LLM model instance.
            env: The execution environment.
            instance_id: Unique identifier for this SWE-bench instance.
            bug_anchor_text: Bug description/stacktrace for attention sink.
            file_content_store: Optional pre-loaded file contents for fetch_text.
            progress_manager: Optional progress manager for batch mode (accepted but unused).
            config_class: Configuration class (default: PrefetchAgentConfig).
            **kwargs: Additional config parameters.
        """
        super().__init__(model, env, config_class=config_class, **kwargs)

        # P4: 启动日志（验证用）
        import logging
        _logger = logging.getLogger("minisweagent.prefetch")
        _logger.info(f"[PrefetchAgent] ENABLED instance_id={instance_id}")
        print(f"[PrefetchAgent] ENABLED instance_id={instance_id}")

        # Per-instance identifiers
        self.instance_id = instance_id
        self.bug_anchor_text = bug_anchor_text
        self.file_content_store = file_content_store or {}
        self._progress_manager = progress_manager  # 存储但不使用

        # Command history for paging detection (last N commands)
        self.command_history: List[str] = []

        # Step-level metrics for trajectory logging
        self.step_metrics: List[Dict[str, Any]] = []

        # Query history for repeat detection
        self.query_history: List[str] = []

        # Stagnation tracking (step of last progress)
        self.last_progress_step = 0

        # Paging span tracking for unique lines computation
        self.paging_spans_by_file: Dict[str, List[tuple]] = defaultdict(list)

        # Initialize prefetch coordinator (per-instance, Atlas=None default)
        self.coordinator: Optional[PrefetchCoordinator] = None
        if self.config.prefetch_enabled:
            self.coordinator = PrefetchCoordinator(
                atlas=None,  # Atlas optional, disabled by default
                base_budget=self.config.prefetch_base_budget,
                verified_mode=self.config.prefetch_verified_mode,
                max_pager_blocks=self.config.prefetch_max_pager_blocks,
                evidence_top_k=self.config.prefetch_evidence_top_k,
                shadow_top_m=self.config.prefetch_shadow_top_m,
            )
            # Initialize attention sink with bug anchor
            if self.bug_anchor_text:
                self.coordinator.attention_sink.bug_anchor = self.bug_anchor_text[:500]

    def _fetch_text(self, span: CodeSpan) -> str:
        """Fetch text content for a code span.

        Tries file_content_store first, falls back to environment execution.
        """
        content = self.file_content_store.get(span.file_path, "")
        if content:
            lines = content.splitlines()
            start = max(0, span.start_line - 1)
            end = min(len(lines), span.end_line)
            return "\n".join(lines[start:end])

        # Fallback: execute sed in environment
        try:
            result = self.env.execute(
                f"sed -n '{span.start_line},{span.end_line}p' '{span.file_path}'"
            )
            return result.get("output", "")
        except Exception:
            return f"(content unavailable for {span.file_path}:{span.start_line}-{span.end_line})"

    def _extract_query(self, command: str) -> str:
        """Extract search query from grep/rg/find commands."""
        patterns = [
            r"grep\s+(?:-[^\s]+\s+)*['\"]?([^'\"|\n]+)['\"]?",
            r"rg\s+(?:-[^\s]+\s+)*['\"]?([^'\"|\n]+)['\"]?",
            r"find\s+.*-name\s+['\"]?([^'\"]+)['\"]?",
            r"ag\s+(?:-[^\s]+\s+)*['\"]?([^'\"|\n]+)['\"]?",
        ]
        for p in patterns:
            m = re.search(p, command)
            if m:
                return m.group(1).strip()
        # Fallback: return truncated command
        return command[:100]

    def _extract_retrieval_items(self, output: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract retrieval-like items from grep/rg output.

        Parses file:line:content format common in search results.
        """
        items = []
        text = output.get("output", "")

        # Pattern: file:line:content (grep/rg style)
        file_line_pattern = r"^([^\s:]+):(\d+):(.*)$"

        for i, line in enumerate(text.splitlines()[:50]):  # Limit parsing
            m = re.match(file_line_pattern, line)
            if m:
                fpath, lineno, snippet = m.groups()
                items.append({
                    "symbol": f"{fpath}:{lineno}",
                    "score": 1.0 / (i + 1),  # Rank-based score
                    "span": {
                        "file_path": fpath,
                        "start_line": int(lineno),
                        "end_line": int(lineno) + 5,
                    },
                    "text_snippet": snippet[:200],
                })

        return items

    def _compute_progress_state(self) -> Dict[str, float]:
        """Compute progress indicators for adaptive budget scheduling."""
        step_idx = len(self.messages) // 2

        # Query repeat rate (how often recent queries repeat)
        if len(self.query_history) >= 2:
            recent = self.query_history[-5:]
            unique = len(set(recent))
            repeat_rate = 1.0 - (unique / len(recent))
        else:
            repeat_rate = 0.0

        # Stagnation (steps since last progress indicator)
        stagnation = step_idx - self.last_progress_step

        # Stack coverage (placeholder - would need actual stack parsing)
        # In production, this could be computed from test output or stacktrace matching
        stack_coverage = 0.3

        return {
            "stagnation_steps": stagnation,
            "stack_coverage": stack_coverage,
            "query_repeat_rate": repeat_rate,
            "step_index": step_idx,
        }

    def _infer_focus(self, output: Dict[str, Any]) -> Optional[str]:
        """Infer current focus file/function from command or output."""
        cmd = output.get("action", "")

        # Extract file being read
        file_patterns = [
            r"(?:cat|head|tail|less|more)\s+([^\s|>]+\.py)",
            r"(?:sed|nl)\s+(?:-[^\s]+\s+)*([^\s|]+\.py)",
            r"vim?\s+([^\s]+\.py)",
        ]
        for p in file_patterns:
            m = re.search(p, cmd)
            if m:
                return m.group(1)

        return None

    def _parse_actual_accesses(self, output: Dict[str, Any]) -> List[str]:
        """Parse symbols/files actually accessed this step for shadow evaluation."""
        accesses = []
        cmd = output.get("action", "")

        # File read operations
        file_pattern = r"(?:cat|head|tail|sed|nl|less|more|vim?)\s+(?:-[^\s]+\s+)*([^\s|>]+)"
        for m in re.finditer(file_pattern, cmd):
            accesses.append(m.group(1))

        # Grep/search targets
        search_pattern = r"(?:grep|rg|ag)\s+.*?([^\s]+)$"
        m = re.search(search_pattern, cmd)
        if m:
            accesses.append(m.group(1))

        return accesses

    def _count_manual_paging_ops(self, cmd: str) -> int:
        """Count nl|sed style manual paging operations in command."""
        spans = CommandPagerParser.parse_many([cmd])
        return len(spans)

    def _compute_paging_metrics(self, cmd: str) -> Dict[str, Any]:
        """Compute paging-specific metrics for a command."""
        span = CommandPagerParser.parse(cmd)
        if span:
            # Track for unique lines computation
            self.paging_spans_by_file[span.file_path].append(
                (span.start_line, span.end_line)
            )
            return {
                "is_paging_op": True,
                "paging_file": span.file_path,
                "paging_start": span.start_line,
                "paging_end": span.end_line,
                "paging_lines": span.end_line - span.start_line + 1,
            }
        return {"is_paging_op": False, "paging_lines": 0}

    def _record_step_metrics(
        self,
        output: Dict[str, Any],
        plan: Optional[PrefetchPlan],
        budget_logs: Dict[str, Any],
    ) -> None:
        """Record comprehensive step-level metrics for analysis."""
        cmd = output.get("action", "")
        step_idx = len(self.step_metrics)

        paging = self._compute_paging_metrics(cmd)

        metrics = {
            "step_id": step_idx,
            "instance_id": self.instance_id,
            "prefetch_enabled": self.config.prefetch_enabled,

            # Budget state
            "token_budget_prefetch": budget_logs.get("token_budget_prefetch", 0),
            "aggression_level": budget_logs.get("aggression_level", 0),
            "jitter_ratio": budget_logs.get("jitter_ratio", 0.0),
            "uncertainty": budget_logs.get("uncertainty", 0.0),
            "flatness": budget_logs.get("flatness", 0.0),
            "coverage": budget_logs.get("coverage", 0.0),
            "stagnation": budget_logs.get("stagnation", 0.0),
            "repeat": budget_logs.get("repeat", 0.0),

            # Plan composition
            "pager_blocks_count": len(plan.pager_blocks) if plan else 0,
            "det_blocks_count": len(plan.det_blocks) if plan else 0,
            "jit_blocks_count": len(plan.jitter_blocks) if plan else 0,
            "evidence_blocks_count": len(plan.evidence_blocks) if plan else 0,

            # Token estimates
            "estimated_tokens_total": (
                plan.meta.get("pager_tokens", 0) +
                plan.meta.get("evidence_tokens", 0) +
                plan.meta.get("det_tokens", 0) +
                plan.meta.get("jitter_tokens", 0)
            ) if plan else 0,
            "estimated_tokens_pager": plan.meta.get("pager_tokens", 0) if plan else 0,
            "estimated_tokens_evidence": plan.meta.get("evidence_tokens", 0) if plan else 0,
            "estimated_tokens_det": plan.meta.get("det_tokens", 0) if plan else 0,
            "estimated_tokens_jit": plan.meta.get("jitter_tokens", 0) if plan else 0,

            # Paging metrics
            "manual_paging_ops_step": self._count_manual_paging_ops(cmd),
            "paging_span_lines_step": paging.get("paging_lines", 0),
            "is_paging_op": paging.get("is_paging_op", False),

            # Command (truncated)
            "command": cmd[:500],
        }

        self.step_metrics.append(metrics)

    def _compute_unique_paging_lines(self) -> int:
        """Compute total unique lines across all paging operations (union rule)."""
        total_unique = 0

        for fpath, ranges in self.paging_spans_by_file.items():
            if not ranges:
                continue

            # Sort and merge overlapping ranges
            sorted_ranges = sorted(ranges)
            merged = [sorted_ranges[0]]

            for s, e in sorted_ranges[1:]:
                prev_s, prev_e = merged[-1]
                if s <= prev_e + 1:  # Overlapping or adjacent
                    merged[-1] = (prev_s, max(prev_e, e))
                else:
                    merged.append((s, e))

            # Sum merged ranges
            for s, e in merged:
                total_unique += (e - s + 1)

        return total_unique

    def get_observation(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """Execute action and build observation with prefetch context.

        This is the main integration point for LSR-Engine v2.
        """
        output = self.execute_action(self.parse_action(response))

        # Track command history
        cmd = output.get("action", "")
        self.command_history.append(cmd)

        # Track query for repeat detection
        query = self._extract_query(cmd)
        self.query_history.append(query)

        # Detect progress indicators for stagnation tracking
        out_text = output.get("output", "")
        progress_indicators = ["PASS", "OK", "modified", "created", "success"]
        if any(ind.lower() in out_text.lower() for ind in progress_indicators):
            self.last_progress_step = len(self.messages) // 2

        if self.coordinator and self.config.prefetch_enabled:
            # Extract retrieval-like items from output
            retrieval_items = self._extract_retrieval_items(output)

            # Compute progress state for adaptive budgeting
            progress = self._compute_progress_state()

            # 1. Decide budget
            budget_state, budget_logs = self.coordinator.decide_budget(
                retrieval_items=retrieval_items,
                stack_coverage=progress["stack_coverage"],
                stagnation_rounds=progress["stagnation_steps"],
                query_repeat_rate=progress["query_repeat_rate"],
            )

            # 2. Build prefetch plan
            step_id = len(self.messages) // 2
            plan = self.coordinator.build_plan(
                query=query,
                instance_id=self.instance_id,
                step_id=step_id,
                retrieval_items=retrieval_items,
                bug_anchor_text=self.bug_anchor_text,
                core_focus_id=self._infer_focus(output),
                paging_history_cmds=self.command_history[-10:],
                fetch_text=self._fetch_text,
                budget_state=budget_state,
            )

            # 3. Wrap observation with prefetch context
            raw_obs = self.render_template(
                self.config.action_observation_template, output=output
            )
            observation = self.coordinator.wrap_observation(raw_obs, plan)

            # 4. Record step metrics
            self._record_step_metrics(output, plan, budget_logs)

            # 5. Update shadow evaluation
            actual = self._parse_actual_accesses(output)
            shadow_logs = self.coordinator.update_shadow(actual)

            # Add shadow metrics to step record
            self.step_metrics[-1].update({
                "shadow_hit_at5": shadow_logs.get("hit_at_top_m", False),
                "shadow_unique_hits": shadow_logs.get("unique_hits", 0),
                "shadow_consecutive_misses": shadow_logs.get("consecutive_misses", 0),
                "shrink_applied": shadow_logs.get("budget_multiplier", 1.0) < 1.0,
            })

        else:
            # Prefetch disabled: use standard observation
            observation = self.render_template(
                self.config.action_observation_template, output=output
            )
            self._record_step_metrics(output, None, {})

        self.add_message("user", observation)
        return output

    def get_step_metrics(self) -> List[Dict[str, Any]]:
        """Return all collected step metrics for trajectory logging."""
        return self.step_metrics

    def get_aggregate_metrics(self) -> Dict[str, Any]:
        """Compute aggregate metrics across all steps."""
        if not self.step_metrics:
            return {}

        total_paging_ops = sum(m.get("manual_paging_ops_step", 0) for m in self.step_metrics)
        total_paging_lines = sum(m.get("paging_span_lines_step", 0) for m in self.step_metrics)
        unique_paging_lines = self._compute_unique_paging_lines()

        paging_steps = [m for m in self.step_metrics if m.get("is_paging_op")]
        n_steps = len(self.step_metrics)

        # Compute read duplication rate
        read_dup_rate = (
            1.0 - (unique_paging_lines / total_paging_lines)
            if total_paging_lines > 0 else 0.0
        )

        # Shadow hit rate
        shadow_hits = sum(1 for m in self.step_metrics if m.get("shadow_hit_at5"))

        return {
            "total_steps": n_steps,
            "total_manual_paging_ops": total_paging_ops,
            "paging_span_total_lines": total_paging_lines,
            "paging_span_unique_lines": unique_paging_lines,
            "read_duplication_rate": round(read_dup_rate, 4),
            "paging_steps_count": len(paging_steps),
            "avg_pager_blocks": round(
                sum(m.get("pager_blocks_count", 0) for m in self.step_metrics) / max(1, n_steps),
                2
            ),
            "avg_det_blocks": round(
                sum(m.get("det_blocks_count", 0) for m in self.step_metrics) / max(1, n_steps),
                2
            ),
            "avg_jit_blocks": round(
                sum(m.get("jit_blocks_count", 0) for m in self.step_metrics) / max(1, n_steps),
                2
            ),
            "shadow_hit_rate": round(shadow_hits / max(1, n_steps), 4),
            "total_estimated_tokens": sum(m.get("estimated_tokens_total", 0) for m in self.step_metrics),
        }

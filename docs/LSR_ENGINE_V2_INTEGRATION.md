# LSR-Engine v2 Integration Plan

## 1) Integration Plan (Call Sites & Flow)

### 1.1 Agent Loop Call Flow (Current)

```
DefaultAgent.run(task)
  └─> add_message("system", system_template)
  └─> add_message("user", instance_template)
  └─> while True:
        └─> step()
              └─> query()           # LLM call
              └─> parse_action()    # Extract bash block
              └─> execute_action()  # Run command
              └─> get_observation() # Render output
                    └─> render_template(action_observation_template)
                    └─> add_message("user", observation)
```

### 1.2 Integration Points for LSR-Engine v2

We create a **PrefetchAgent** subclass of `DefaultAgent` that overrides `get_observation()`:

```
PrefetchAgent(DefaultAgent)
  ├─ __init__():
  │     └─> self.coordinator = PrefetchCoordinator(atlas=None, ...)  # per-instance
  │     └─> self.command_history: List[str] = []
  │     └─> self.step_metrics: List[dict] = []
  │
  └─> get_observation(response):
        # BEFORE: Normal execution
        output = self.execute_action(self.parse_action(response))

        # A) Collect paging history (last N commands)
        self.command_history.append(output["action"])
        paging_cmds = self.command_history[-10:]

        # B) Build retrieval_items from output (if search-like)
        retrieval_items = self._extract_retrieval_items(output)

        # C) Build progress state
        progress = self._compute_progress_state()

        # D) Call LSR-Engine:
        #    1. decide_budget
        budget_state, budget_logs = self.coordinator.decide_budget(
            retrieval_items=retrieval_items,
            stack_coverage=progress.stack_coverage,
            stagnation_rounds=progress.stagnation_steps,
            query_repeat_rate=progress.query_repeat_rate,
        )

        #    2. build_plan
        plan = self.coordinator.build_plan(
            query=self._extract_query(output["action"]),
            instance_id=self.instance_id,
            step_id=len(self.messages) // 2,
            retrieval_items=retrieval_items,
            bug_anchor_text=self.bug_anchor_text,
            core_focus_id=self._infer_focus(output),
            paging_history_cmds=paging_cmds,
            fetch_text=self._fetch_text,
            budget_state=budget_state,
        )

        #    3. wrap_observation (instead of direct render_template)
        raw_observation = self.render_template(self.config.action_observation_template, output=output)
        observation = self.coordinator.wrap_observation(raw_observation, plan)

        # E) Record step metrics
        self._record_step_metrics(output, plan, budget_logs)

        self.add_message("user", observation)
        return output

  └─> _post_step_hook(output):
        # Called after message is added; compute actual_accesses
        actual_accesses = self._parse_actual_accesses(output)
        shadow_logs = self.coordinator.update_shadow(actual_accesses)
        self.step_metrics[-1]["shadow"] = shadow_logs
```

### 1.3 File → Module Mapping

| Integration Point | File | Function/Class |
|-------------------|------|----------------|
| Agent subclass | `src/minisweagent/agents/prefetch_agent.py` (NEW) | `PrefetchAgent` |
| Coordinator init | `PrefetchAgent.__init__` | Per-instance isolation |
| decide_budget | `PrefetchAgent.get_observation` | Before build_plan |
| build_plan | `PrefetchAgent.get_observation` | After budget |
| wrap_observation | `PrefetchAgent.get_observation` | Before add_message |
| update_shadow | `PrefetchAgent._post_step_hook` | After step completes |
| Trajectory save | `src/minisweagent/run/utils/save.py` | `save_traj()` + extra_info |
| SWE-bench runner | `src/minisweagent/run/extra/swebench_single.py` | Use PrefetchAgent |

### 1.4 Atlas Paths

```python
# Atlas=None path (default, always works)
if self.atlas is None:
    # Only pager + evidence + attention_sink + gate + shadow
    # DeterministicInfiller.expand() returns []
    # ControlledJitter.sample() returns []

# Atlas present path (future, requires offline cache)
if self.atlas is not None:
    # Full functionality with det + jitter
    # node_stats read from Atlas.node_stats() cache
    # NO runtime graph compute (forbidden)
```

---

## 2) Patches (Files to Change + Key Code Snippets)

### 2.1 NEW FILE: `src/minisweagent/agents/prefetch_agent.py`

```python
"""Prefetch-enabled agent using LSR-Engine v2."""

import re
from typing import Any, Callable, Dict, List, Optional

from minisweagent.agents.default import DefaultAgent, AgentConfig
from minisweagent.prefetch.lsr_engine_v2 import (
    PrefetchCoordinator,
    CodeSpan,
    CommandPagerParser,
    BudgetState,
)


class PrefetchAgentConfig(AgentConfig):
    """Extended config for prefetch agent."""
    prefetch_enabled: bool = True
    prefetch_base_budget: int = 4000
    prefetch_verified_mode: bool = True
    prefetch_max_pager_blocks: int = 2
    prefetch_evidence_top_k: int = 2


class PrefetchAgent(DefaultAgent):
    """Agent with LSR-Engine v2 prefetch capabilities."""

    def __init__(
        self,
        model,
        env,
        *,
        instance_id: str = "",
        bug_anchor_text: str = "",
        file_content_store: Optional[Dict[str, str]] = None,
        config_class: type = PrefetchAgentConfig,
        **kwargs,
    ):
        super().__init__(model, env, config_class=config_class, **kwargs)

        # Per-instance state (CRITICAL: no cross-instance leakage)
        self.instance_id = instance_id
        self.bug_anchor_text = bug_anchor_text
        self.file_content_store = file_content_store or {}

        # Command history for paging detection
        self.command_history: List[str] = []

        # Step-level metrics
        self.step_metrics: List[Dict[str, Any]] = []

        # Query history for repeat detection
        self.query_history: List[str] = []

        # Stagnation tracking
        self.last_progress_step = 0

        # Initialize coordinator (per-instance, Atlas=None default)
        self.coordinator: Optional[PrefetchCoordinator] = None
        if self.config.prefetch_enabled:
            self.coordinator = PrefetchCoordinator(
                atlas=None,  # Atlas optional, default off
                base_budget=self.config.prefetch_base_budget,
                verified_mode=self.config.prefetch_verified_mode,
                max_pager_blocks=self.config.prefetch_max_pager_blocks,
                evidence_top_k=self.config.prefetch_evidence_top_k,
            )
            # Set bug anchor
            if self.bug_anchor_text:
                self.coordinator.attention_sink.bug_anchor = self.bug_anchor_text

    def _fetch_text(self, span: CodeSpan) -> str:
        """Fetch text for a code span from file store or environment."""
        content = self.file_content_store.get(span.file_path, "")
        if not content:
            # Fallback: try to read from environment (if available)
            try:
                result = self.env.execute(f"sed -n '{span.start_line},{span.end_line}p' {span.file_path}")
                content = result.get("output", "")
            except Exception:
                content = f"(content unavailable for {span.file_path})"
        else:
            lines = content.splitlines()
            start = max(0, span.start_line - 1)
            end = min(len(lines), span.end_line)
            content = "\n".join(lines[start:end])
        return content

    def _extract_query(self, command: str) -> str:
        """Extract search query from command if it's a search operation."""
        # Pattern: grep/rg/find/ag with query
        patterns = [
            r"grep\s+(?:-[^\s]+\s+)*['\"]?([^'\"]+)['\"]?",
            r"rg\s+(?:-[^\s]+\s+)*['\"]?([^'\"]+)['\"]?",
            r"find\s+.*-name\s+['\"]?([^'\"]+)['\"]?",
        ]
        for p in patterns:
            m = re.search(p, command)
            if m:
                return m.group(1)
        return command[:100]  # fallback: truncated command

    def _extract_retrieval_items(self, output: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract retrieval-like items from command output."""
        items = []
        text = output.get("output", "")

        # Pattern: file:line matches (grep/rg style)
        file_line_pattern = r"^([^\s:]+):(\d+):(.*)$"
        for line in text.splitlines()[:50]:  # limit parsing
            m = re.match(file_line_pattern, line)
            if m:
                fpath, lineno, snippet = m.groups()
                items.append({
                    "symbol": f"{fpath}:{lineno}",
                    "score": 1.0 / (len(items) + 1),  # rank-based score
                    "span": {
                        "file_path": fpath,
                        "start_line": int(lineno),
                        "end_line": int(lineno) + 5,
                    },
                    "text_snippet": snippet[:200],
                })
        return items

    def _compute_progress_state(self) -> Dict[str, float]:
        """Compute progress indicators for budget scheduling."""
        step_idx = len(self.messages) // 2

        # Query repeat rate
        if len(self.query_history) >= 2:
            recent = self.query_history[-5:]
            unique = len(set(recent))
            repeat_rate = 1.0 - (unique / len(recent))
        else:
            repeat_rate = 0.0

        # Stagnation (steps since last "progress" - file edit, test pass, etc.)
        stagnation = step_idx - self.last_progress_step

        # Stack coverage (placeholder - would need actual stack parsing)
        stack_coverage = 0.3  # default conservative estimate

        return {
            "stagnation_steps": stagnation,
            "stack_coverage": stack_coverage,
            "query_repeat_rate": repeat_rate,
            "step_index": step_idx,
        }

    def _infer_focus(self, output: Dict[str, Any]) -> Optional[str]:
        """Infer current focus function from output or command."""
        cmd = output.get("action", "")
        # If reading a specific file, that's the focus
        m = re.search(r"(?:cat|head|tail|sed|nl)\s+[^\s|]+?([a-zA-Z_][a-zA-Z0-9_]*\.py)", cmd)
        if m:
            return m.group(0)
        return None

    def _parse_actual_accesses(self, output: Dict[str, Any]) -> List[str]:
        """Parse what the agent actually accessed this step."""
        accesses = []
        cmd = output.get("action", "")

        # File reads
        file_pattern = r"(?:cat|head|tail|sed|nl|less|more)\s+(?:-[^\s]+\s+)*([^\s|]+)"
        for m in re.finditer(file_pattern, cmd):
            accesses.append(m.group(1))

        # Grep targets
        grep_pattern = r"(?:grep|rg)\s+.*?([^\s]+)$"
        m = re.search(grep_pattern, cmd)
        if m:
            accesses.append(m.group(1))

        return accesses

    def _count_manual_paging_ops(self, cmd: str) -> int:
        """Count nl|sed style manual paging operations in command."""
        spans = CommandPagerParser.parse_many([cmd])
        return len(spans)

    def _compute_paging_metrics(self, cmd: str) -> Dict[str, Any]:
        """Compute paging-specific metrics for this command."""
        span = CommandPagerParser.parse(cmd)
        if span:
            return {
                "is_paging_op": True,
                "paging_file": span.file_path,
                "paging_start": span.start_line,
                "paging_end": span.end_line,
                "paging_lines": span.end_line - span.start_line + 1,
            }
        return {"is_paging_op": False}

    def _record_step_metrics(
        self,
        output: Dict[str, Any],
        plan,
        budget_logs: Dict[str, Any],
    ) -> None:
        """Record comprehensive step-level metrics."""
        cmd = output.get("action", "")
        step_idx = len(self.step_metrics)

        paging = self._compute_paging_metrics(cmd)

        metrics = {
            "step_id": step_idx,
            "instance_id": self.instance_id,
            "prefetch_enabled": self.config.prefetch_enabled,

            # Budget
            "token_budget_prefetch": budget_logs.get("token_budget_prefetch", 0),
            "aggression_level": budget_logs.get("aggression_level", 0),
            "jitter_ratio": budget_logs.get("jitter_ratio", 0.0),
            "uncertainty": budget_logs.get("uncertainty", 0.0),

            # Plan composition
            "pager_blocks_count": len(plan.pager_blocks) if plan else 0,
            "det_blocks_count": len(plan.det_blocks) if plan else 0,
            "jit_blocks_count": len(plan.jitter_blocks) if plan else 0,
            "evidence_blocks_count": len(plan.evidence_blocks) if plan else 0,

            # Token estimates
            "estimated_tokens_total": plan.meta.get("pager_tokens", 0) + plan.meta.get("det_tokens", 0) + plan.meta.get("jitter_tokens", 0) if plan else 0,
            "estimated_tokens_pager": plan.meta.get("pager_tokens", 0) if plan else 0,
            "estimated_tokens_det": plan.meta.get("det_tokens", 0) if plan else 0,
            "estimated_tokens_jit": plan.meta.get("jitter_tokens", 0) if plan else 0,

            # Paging metrics
            "manual_paging_ops_step": self._count_manual_paging_ops(cmd),
            "paging_span_lines_step": paging.get("paging_lines", 0),
            "is_paging_op": paging.get("is_paging_op", False),

            # Command info
            "command": cmd[:500],
        }

        self.step_metrics.append(metrics)

    def get_observation(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """Override to inject prefetch context."""
        output = self.execute_action(self.parse_action(response))

        # Track command history
        cmd = output.get("action", "")
        self.command_history.append(cmd)

        # Track query for repeat detection
        query = self._extract_query(cmd)
        self.query_history.append(query)

        # Check for progress indicators
        out_text = output.get("output", "")
        if "PASS" in out_text or "OK" in out_text or "modified" in out_text.lower():
            self.last_progress_step = len(self.messages) // 2

        if self.coordinator and self.config.prefetch_enabled:
            # Extract retrieval items
            retrieval_items = self._extract_retrieval_items(output)

            # Compute progress state
            progress = self._compute_progress_state()

            # 1. decide_budget
            budget_state, budget_logs = self.coordinator.decide_budget(
                retrieval_items=retrieval_items,
                stack_coverage=progress["stack_coverage"],
                stagnation_rounds=progress["stagnation_steps"],
                query_repeat_rate=progress["query_repeat_rate"],
            )

            # 2. build_plan
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

            # 3. wrap_observation
            raw_obs = self.render_template(self.config.action_observation_template, output=output)
            observation = self.coordinator.wrap_observation(raw_obs, plan)

            # 4. Record metrics
            self._record_step_metrics(output, plan, budget_logs)

            # 5. Update shadow (with actual accesses from this step)
            actual = self._parse_actual_accesses(output)
            shadow_logs = self.coordinator.update_shadow(actual)
            self.step_metrics[-1]["shadow_hit_at5"] = shadow_logs.get("hit_at_top_m", False)
            self.step_metrics[-1]["shadow_unique_hits"] = shadow_logs.get("unique_hits", 0)
            self.step_metrics[-1]["shadow_consecutive_misses"] = shadow_logs.get("consecutive_misses", 0)
            self.step_metrics[-1]["shrink_applied"] = shadow_logs.get("budget_multiplier", 1.0) < 1.0

        else:
            # Prefetch disabled: normal observation
            observation = self.render_template(self.config.action_observation_template, output=output)
            self._record_step_metrics(output, None, {})

        self.add_message("user", observation)
        return output

    def get_step_metrics(self) -> List[Dict[str, Any]]:
        """Return all collected step metrics for trajectory saving."""
        return self.step_metrics

    def get_aggregate_metrics(self) -> Dict[str, Any]:
        """Compute aggregate metrics across all steps."""
        if not self.step_metrics:
            return {}

        total_paging_ops = sum(m.get("manual_paging_ops_step", 0) for m in self.step_metrics)
        total_paging_lines = sum(m.get("paging_span_lines_step", 0) for m in self.step_metrics)

        # Unique lines (union of all paging spans)
        seen_ranges = {}  # file -> set of line numbers
        for m in self.step_metrics:
            if m.get("is_paging_op"):
                # Would need to track per-span, simplified here
                pass

        paging_steps = [m for m in self.step_metrics if m.get("is_paging_op")]

        return {
            "total_steps": len(self.step_metrics),
            "total_manual_paging_ops": total_paging_ops,
            "total_paging_lines": total_paging_lines,
            "paging_steps_count": len(paging_steps),
            "avg_pager_blocks": sum(m.get("pager_blocks_count", 0) for m in self.step_metrics) / max(1, len(self.step_metrics)),
            "shadow_hit_rate": sum(1 for m in self.step_metrics if m.get("shadow_hit_at5")) / max(1, len(self.step_metrics)),
        }
```

### 2.2 MODIFY: `src/minisweagent/run/utils/save.py`

Add prefetch metrics to trajectory:

```python
# In save_traj(), after line 63:

    # Add prefetch metrics if available
    if hasattr(agent, 'get_step_metrics'):
        data["prefetch_step_metrics"] = agent.get_step_metrics()
    if hasattr(agent, 'get_aggregate_metrics'):
        data["info"]["prefetch_aggregate"] = agent.get_aggregate_metrics()
```

### 2.3 MODIFY: `src/minisweagent/run/extra/swebench_single.py`

Use PrefetchAgent:

```python
# Add import at top:
from minisweagent.agents.prefetch_agent import PrefetchAgent, PrefetchAgentConfig

# Replace InteractiveAgent with PrefetchAgent (around line 61):

    # Extract bug anchor from problem statement
    bug_anchor = instance.get("problem_statement", "")[:500]

    # Determine if prefetch enabled from config
    prefetch_config = config.get("prefetch", {})
    prefetch_enabled = prefetch_config.get("enabled", False)

    agent_cls = PrefetchAgent if prefetch_enabled else InteractiveAgent
    agent_kwargs = {"mode": "yolo"} | config.get("agent", {})

    if prefetch_enabled:
        agent_kwargs.update({
            "instance_id": instance["instance_id"],
            "bug_anchor_text": bug_anchor,
            "prefetch_enabled": True,
            "prefetch_base_budget": prefetch_config.get("base_budget", 4000),
            "prefetch_verified_mode": prefetch_config.get("verified_mode", True),
        })

    agent = agent_cls(
        get_model(model_name, config.get("model", {})),
        env,
        **agent_kwargs,
    )
```

### 2.4 Config Schema Addition

Add to YAML configs (e.g., `swebench_prefetch_v2.yaml`):

```yaml
# Prefetch configuration
prefetch:
  enabled: true
  base_budget: 4000
  verified_mode: true
  max_pager_blocks: 2
  evidence_top_k: 2
  # Future Atlas settings (disabled by default)
  atlas_enabled: false
  atlas_cache_path: null
```

---

## 3) Logging Schema (Field Definitions & Sources)

### 3.1 Per-Step Fields (in `prefetch_step_metrics[]`)

| Field | Type | Source | Description |
|-------|------|--------|-------------|
| `prefetch_enabled` | bool | Config | Whether prefetch is active |
| `instance_id` | str | Init param | SWE-bench instance ID |
| `step_id` | int | Message count | Current step index |
| `token_budget_prefetch` | int | `BudgetState` | Allocated prefetch token budget |
| `aggression_level` | int | `BudgetState` | 0-3 scale of prefetch intensity |
| `jitter_ratio` | float | `BudgetState` | Jitter sampling ratio (0-0.3) |
| `uncertainty` | float | `BudgetState` | Computed uncertainty score |
| `pager_blocks_count` | int | `PrefetchPlan` | Number of pager blocks emitted |
| `det_blocks_count` | int | `PrefetchPlan` | Number of deterministic callee blocks |
| `jit_blocks_count` | int | `PrefetchPlan` | Number of jitter blocks |
| `evidence_blocks_count` | int | `PrefetchPlan` | Number of evidence blocks |
| `estimated_tokens_total` | int | `plan.meta` | Total estimated tokens in prefetch |
| `estimated_tokens_pager` | int | `plan.meta` | Tokens from pager blocks |
| `estimated_tokens_det` | int | `plan.meta` | Tokens from det blocks |
| `estimated_tokens_jit` | int | `plan.meta` | Tokens from jitter blocks |
| `manual_paging_ops_step` | int | `CommandPagerParser` | Count of nl\|sed ops in this step |
| `paging_span_lines_step` | int | Parsed span | Lines covered by paging this step |
| `is_paging_op` | bool | Parser | Whether this command is a paging op |
| `shadow_hit_at5` | bool | `ShadowEvaluator` | Did any top-5 prediction hit? |
| `shadow_unique_hits` | int | `ShadowEvaluator` | Count of unique hits |
| `shadow_consecutive_misses` | int | `ShadowEvaluator` | Consecutive miss count |
| `shrink_applied` | bool | `ShadowEvaluator` | Was budget shrunk? |
| `command` | str | Execution | Truncated command text |

### 3.2 Aggregate Fields (in `info.prefetch_aggregate`)

| Field | Type | Computation |
|-------|------|-------------|
| `total_steps` | int | `len(step_metrics)` |
| `total_manual_paging_ops` | int | `sum(manual_paging_ops_step)` |
| `total_paging_lines` | int | `sum(paging_span_lines_step)` |
| `paging_span_unique_lines` | int | Union of all paging spans per file |
| `paging_steps_count` | int | Count of steps with `is_paging_op=True` |
| `avg_pager_blocks` | float | Mean of `pager_blocks_count` |
| `shadow_hit_rate` | float | `sum(shadow_hit_at5) / total_steps` |
| `read_duplication_rate` | float | `1 - (unique_lines / total_lines)` |

### 3.3 Unique Lines Computation (Union Rule)

```python
def compute_unique_paging_lines(step_metrics: List[dict]) -> int:
    """
    Union rule: For each file, collect all (start, end) ranges,
    merge overlapping, then sum line counts.
    """
    from collections import defaultdict

    by_file = defaultdict(list)
    for m in step_metrics:
        if m.get("is_paging_op") and "paging_file" in m:
            by_file[m["paging_file"]].append(
                (m["paging_start"], m["paging_end"])
            )

    total_unique = 0
    for fpath, ranges in by_file.items():
        # Sort and merge
        ranges.sort()
        merged = []
        for s, e in ranges:
            if merged and s <= merged[-1][1] + 1:
                merged[-1] = (merged[-1][0], max(merged[-1][1], e))
            else:
                merged.append((s, e))
        # Sum
        for s, e in merged:
            total_unique += (e - s + 1)

    return total_unique
```

### 3.4 Cache-Specific Fields (for `prefetch_v2_cache` group)

| Field | Type | Source |
|-------|------|--------|
| `cache_hits` | int | ReadCache module |
| `cache_misses` | int | ReadCache module |
| `cache_hit_rate` | float | `hits / (hits + misses)` |
| `cache_evictions` | int | LRU eviction count |

---

## 4) Modal Runbook (3-Slice Plan + Smoke→Full + Retry)

### 4.1 Why Previous Runs Failed

**Root Cause**: `SandboxTimeoutError: Some tunnels failed to open`

1. **Cold start amplification**: Modal's swe-rex nested sandbox requires:
   - Outer container boot (~30s)
   - Inner sandbox tunnel establishment (~20s)
   - Python environment setup (~15s)

2. **Concurrent overload**: Running 20 instances × 4 groups = 80 concurrent sandbox requests overwhelmed Modal's tunnel allocation.

3. **No retry logic**: Single failure = permanent failure for that instance.

### 4.2 Recommended 3-Slice Selection

| Slice | Range | Repos | Rationale |
|-------|-------|-------|-----------|
| Slice A | 0:20 | astropy (20) | Single-repo baseline |
| Slice B | 20:40 | django (18) + astropy (2) | Cross-repo boundary |
| Slice C | 280:300 | 5 repos (matplotlib, seaborn, flask, requests, xarray) | Long-tail diversity |

### 4.3 Smoke → Full Expansion Strategy

**Phase 1: Smoke Test (2 instances per slice)**

```bash
# Smoke test: 2 instances, 1 worker, verify tunnel stability
for SLICE in "0:2" "20:22" "280:282"; do
    START="${SLICE%%:*}"
    END="${SLICE##*:}"

    echo "=== Smoke: baseline $SLICE ==="
    uv run modal run -e main modal_run.py \
        --config v0 \
        --subset verified \
        --slice-spec "$SLICE" \
        --workers 1 \
        --run-tag "smoke_baseline_${START}_${END}"

    echo "=== Smoke: prefetch_v2_all $SLICE ==="
    uv run modal run -e main modal_run.py \
        --config prefetch_v2_all \
        --subset verified \
        --slice-spec "$SLICE" \
        --workers 1 \
        --run-tag "smoke_prefetch_${START}_${END}"
done
```

**Phase 2: Full Run (20 instances, 2 workers)**

```bash
# Only proceed if smoke passes (>80% success)
for SLICE in "0:20" "20:40" "280:300"; do
    START="${SLICE%%:*}"
    END="${SLICE##*:}"

    # Stagger launches by 60s to avoid cold-start storm
    for CONFIG in "v0" "prefetch_v2_all"; do
        echo "=== Full: $CONFIG $SLICE ==="
        uv run modal run --detach -e main modal_run.py \
            --config "$CONFIG" \
            --subset verified \
            --slice-spec "$SLICE" \
            --workers 2 \
            --run-tag "${CONFIG}_verified_${START}_${END}"

        sleep 60  # Stagger to reduce concurrent sandbox load
    done
done
```

### 4.4 Experiment Matrix

**Phase A: Baseline vs Full Prefetch (6 runs)**

| Group | Slice 0:20 | Slice 20:40 | Slice 280:300 |
|-------|------------|-------------|---------------|
| `v0` (baseline) | Run 1 | Run 2 | Run 3 |
| `prefetch_v2_all` | Run 4 | Run 5 | Run 6 |

**Phase B: Ablation (4 runs, slice 20:40 only)**

| Group | Slice 20:40 |
|-------|-------------|
| `prefetch_v2` (anchor) | Run 7 |
| `prefetch_v2_cache` | Run 8 |
| `prefetch_v2_pager` | Run 9 |
| `prefetch_v2_all` | (from Phase A) |

### 4.5 Retry Logic Patch for `modal_run.py`

```python
# Add to modal_run.py, replace the subprocess.run call in run_swebench_instance

import time

MAX_RETRIES = 2
RETRY_BASE_DELAY = 30  # seconds
CONSECUTIVE_FAILURE_THRESHOLD = 5

def run_with_retry(cmd: list, cwd: str, env: dict, timeout: int) -> dict:
    """Run command with exponential backoff retry for sandbox errors."""
    last_error = None

    for attempt in range(MAX_RETRIES + 1):
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=cwd,
                env=env,
            )

            # Check for sandbox tunnel error in stderr
            if "SandboxTimeoutError" in proc.stderr or "tunnels failed" in proc.stderr:
                if attempt < MAX_RETRIES:
                    delay = RETRY_BASE_DELAY * (2 ** attempt)
                    print(f"[Retry] Sandbox tunnel error, waiting {delay}s (attempt {attempt + 1})")
                    time.sleep(delay)
                    continue

            return {
                "returncode": proc.returncode,
                "stdout": proc.stdout,
                "stderr": proc.stderr,
                "attempts": attempt + 1,
            }

        except subprocess.TimeoutExpired as e:
            last_error = e
            if attempt < MAX_RETRIES:
                delay = RETRY_BASE_DELAY * (2 ** attempt)
                print(f"[Retry] Timeout, waiting {delay}s (attempt {attempt + 1})")
                time.sleep(delay)

    return {
        "returncode": -1,
        "stdout": "",
        "stderr": str(last_error),
        "attempts": MAX_RETRIES + 1,
        "final_error": "max_retries_exceeded",
    }


# Circuit breaker state (module-level)
_consecutive_failures = 0

def check_circuit_breaker() -> bool:
    """Return True if should proceed, False if circuit is open."""
    global _consecutive_failures
    if _consecutive_failures >= CONSECUTIVE_FAILURE_THRESHOLD:
        print(f"[CircuitBreaker] OPEN - {_consecutive_failures} consecutive failures")
        return False
    return True

def record_result(success: bool):
    """Update circuit breaker state."""
    global _consecutive_failures
    if success:
        _consecutive_failures = 0
    else:
        _consecutive_failures += 1
```

### 4.6 Run Commands Template

```bash
# === PHASE A: Smoke Test ===
./scripts/run_smoke_test.sh

# === PHASE A: Full Baseline vs Prefetch (if smoke passes) ===
./scripts/run_phase_a.sh

# === PHASE B: Ablation on slice 20:40 ===
./scripts/run_phase_b_ablation.sh

# === Download & Analyze ===
uv run modal volume ls research-data | grep -E "(v0|prefetch)" > results_list.txt
./scripts/download_results.sh
uv run python analysis/ablation/summarize_hybrid_ablation.py --results-dir results/
```

---

## 5) Summarizer Changes (Paired Metrics + Paging Parsing + Plots)

### 5.1 Key Changes to `summarize_hybrid_ablation.py`

```python
"""
summarize_hybrid_ablation.py - Updated for LSR-Engine v2 integration
"""

import json
import re
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Any, Tuple
import statistics

# Reuse LSR parser
from minisweagent.prefetch.lsr_engine_v2 import CommandPagerParser


class HybridAblationAnalyzer:
    """Analyze ablation results with paired comparison support."""

    def __init__(self, results_dir: Path):
        self.results_dir = results_dir
        self.trajectories: Dict[str, Dict[str, Any]] = {}  # (group, instance_id) -> traj
        self.pager_parser = CommandPagerParser()

    def load_all_trajectories(self):
        """Load all trajectory files from results directory."""
        for traj_file in self.results_dir.rglob("*.traj.json"):
            with open(traj_file) as f:
                data = json.load(f)

            # Extract group and instance_id from path
            # e.g., prefetch_v2_verified_20_40/django__django-11039/...
            parts = traj_file.parts
            group = None
            instance_id = None
            for p in parts:
                if p.startswith("prefetch_") or p.startswith("v0_"):
                    group = p.split("_verified")[0]
                if "__" in p and p.endswith(".traj.json"):
                    instance_id = p.replace(".traj.json", "")

            if group and instance_id:
                key = (group, instance_id)
                self.trajectories[key] = data

    def parse_paging_from_messages(self, messages: List[dict]) -> Dict[str, Any]:
        """Extract paging metrics from message history."""
        paging_ops = 0
        total_lines = 0
        spans_by_file = defaultdict(list)

        for msg in messages:
            if msg.get("role") != "assistant":
                continue
            content = msg.get("content", "")

            # Extract bash commands
            bash_blocks = re.findall(r"```bash\s*\n(.*?)\n```", content, re.DOTALL)
            for block in bash_blocks:
                span = self.pager_parser.parse(block)
                if span:
                    paging_ops += 1
                    lines = span.end_line - span.start_line + 1
                    total_lines += lines
                    spans_by_file[span.file_path].append((span.start_line, span.end_line))

        # Compute unique lines (union per file)
        unique_lines = 0
        for fpath, ranges in spans_by_file.items():
            ranges.sort()
            merged = []
            for s, e in ranges:
                if merged and s <= merged[-1][1] + 1:
                    merged[-1] = (merged[-1][0], max(merged[-1][1], e))
                else:
                    merged.append((s, e))
            for s, e in merged:
                unique_lines += (e - s + 1)

        read_dup_rate = 1.0 - (unique_lines / total_lines) if total_lines > 0 else 0.0

        return {
            "manual_paging_ops": paging_ops,
            "paging_span_total_lines": total_lines,
            "paging_span_unique_lines": unique_lines,
            "read_duplication_rate": read_dup_rate,
        }

    def compute_front_25_explore(self, messages: List[dict]) -> float:
        """Compute fraction of first 25% steps spent exploring (grep/find/ls)."""
        assistant_msgs = [m for m in messages if m.get("role") == "assistant"]
        n_steps = len(assistant_msgs)
        front_25_count = max(1, n_steps // 4)

        explore_patterns = [r"\bgrep\b", r"\bfind\b", r"\bls\b", r"\brg\b", r"\bag\b"]
        explore_count = 0

        for msg in assistant_msgs[:front_25_count]:
            content = msg.get("content", "")
            if any(re.search(p, content) for p in explore_patterns):
                explore_count += 1

        return explore_count / front_25_count

    def compute_instance_metrics(self, traj: dict) -> Dict[str, Any]:
        """Compute all metrics for a single trajectory."""
        messages = traj.get("messages", [])
        info = traj.get("info", {})

        # Basic metrics
        n_steps = sum(1 for m in messages if m.get("role") == "assistant")
        cost = info.get("model_stats", {}).get("instance_cost", 0.0)
        exit_status = info.get("exit_status", "Unknown")
        success = exit_status == "Submitted"

        # Paging metrics
        paging = self.parse_paging_from_messages(messages)

        # Explore ratio
        front_25 = self.compute_front_25_explore(messages)

        # Prefetch metrics (if available)
        prefetch_agg = info.get("prefetch_aggregate", {})

        return {
            "n_steps": n_steps,
            "cost": cost,
            "success": success,
            "exit_status": exit_status,
            **paging,
            "front_25_explore_ratio": front_25,
            **prefetch_agg,
        }

    def compute_paired_deltas(
        self,
        anchor_group: str,
        variant_group: str,
    ) -> Dict[str, List[float]]:
        """Compute paired deltas for instances present in both groups."""
        deltas = defaultdict(list)

        anchor_instances = {k[1]: v for k, v in self.trajectories.items() if k[0] == anchor_group}
        variant_instances = {k[1]: v for k, v in self.trajectories.items() if k[0] == variant_group}

        common_ids = set(anchor_instances.keys()) & set(variant_instances.keys())

        for iid in common_ids:
            anchor_m = self.compute_instance_metrics(anchor_instances[iid])
            variant_m = self.compute_instance_metrics(variant_instances[iid])

            # Only compare if both succeeded
            if anchor_m["success"] and variant_m["success"]:
                deltas["steps_delta"].append(variant_m["n_steps"] - anchor_m["n_steps"])
                deltas["cost_delta"].append(variant_m["cost"] - anchor_m["cost"])
                deltas["paging_ops_delta"].append(
                    variant_m["manual_paging_ops"] - anchor_m["manual_paging_ops"]
                )
                deltas["unique_lines_delta"].append(
                    variant_m["paging_span_unique_lines"] - anchor_m["paging_span_unique_lines"]
                )

        return dict(deltas)

    def compute_group_summary(self, group: str) -> Dict[str, Any]:
        """Compute summary statistics for a single group."""
        group_trajs = {k[1]: v for k, v in self.trajectories.items() if k[0] == group}

        if not group_trajs:
            return {"error": f"No data for group {group}"}

        metrics_list = [self.compute_instance_metrics(t) for t in group_trajs.values()]

        successes = [m for m in metrics_list if m["success"]]

        return {
            "n_instances": len(metrics_list),
            "n_success": len(successes),
            "success_rate": len(successes) / len(metrics_list),
            "avg_steps": statistics.mean([m["n_steps"] for m in successes]) if successes else 0,
            "median_steps": statistics.median([m["n_steps"] for m in successes]) if successes else 0,
            "avg_cost": statistics.mean([m["cost"] for m in successes]) if successes else 0,
            "avg_paging_ops": statistics.mean([m["manual_paging_ops"] for m in successes]) if successes else 0,
            "avg_unique_lines": statistics.mean([m["paging_span_unique_lines"] for m in successes]) if successes else 0,
            "avg_read_dup_rate": statistics.mean([m["read_duplication_rate"] for m in successes]) if successes else 0,
            "avg_front_25_explore": statistics.mean([m["front_25_explore_ratio"] for m in successes]) if successes else 0,
        }

    def generate_report(self, output_path: Path):
        """Generate full ablation report."""
        self.load_all_trajectories()

        # Identify groups
        groups = sorted(set(k[0] for k in self.trajectories.keys()))

        report_lines = [
            "# Hybrid Ablation Experiment Report",
            "",
            f"**Groups**: {', '.join(groups)}",
            f"**Total Trajectories**: {len(self.trajectories)}",
            "",
            "## Group Summaries",
            "",
        ]

        # Per-group summaries
        group_summaries = {}
        for group in groups:
            summary = self.compute_group_summary(group)
            group_summaries[group] = summary

            report_lines.extend([
                f"### {group}",
                f"- Instances: {summary['n_instances']}",
                f"- Success Rate: {summary['success_rate']:.1%}",
                f"- Avg Steps: {summary['avg_steps']:.1f}",
                f"- Median Steps: {summary['median_steps']:.1f}",
                f"- Avg Cost: ${summary['avg_cost']:.4f}",
                f"- Avg Paging Ops: {summary['avg_paging_ops']:.1f}",
                f"- Avg Unique Lines: {summary['avg_unique_lines']:.0f}",
                f"- Read Duplication Rate: {summary['avg_read_dup_rate']:.1%}",
                f"- Front 25% Explore Ratio: {summary['avg_front_25_explore']:.1%}",
                "",
            ])

        # Paired comparisons (if baseline exists)
        if "v0" in groups:
            report_lines.extend([
                "## Paired Comparisons (vs v0 baseline)",
                "",
            ])

            for variant in groups:
                if variant == "v0":
                    continue

                deltas = self.compute_paired_deltas("v0", variant)
                if deltas:
                    n_pairs = len(deltas.get("steps_delta", []))
                    report_lines.extend([
                        f"### {variant} vs v0 (n={n_pairs} paired instances)",
                        f"- Steps Δ: {statistics.mean(deltas['steps_delta']):+.1f} (median: {statistics.median(deltas['steps_delta']):+.1f})",
                        f"- Cost Δ: ${statistics.mean(deltas['cost_delta']):+.4f}",
                        f"- Paging Ops Δ: {statistics.mean(deltas['paging_ops_delta']):+.1f}",
                        "",
                    ])

        output_path.write_text("\n".join(report_lines))
        print(f"Report saved to {output_path}")

        # Also save JSON for plotting
        json_path = output_path.with_suffix(".json")
        with open(json_path, "w") as f:
            json.dump({
                "groups": groups,
                "summaries": group_summaries,
            }, f, indent=2)


# === PLOTTING FUNCTIONS ===

def plot_steps_distribution(analyzer: HybridAblationAnalyzer, output_dir: Path):
    """Plot steps distribution per group."""
    import matplotlib.pyplot as plt

    groups = sorted(set(k[0] for k in analyzer.trajectories.keys()))

    fig, ax = plt.subplots(figsize=(10, 6))

    data = []
    labels = []
    for group in groups:
        trajs = [v for k, v in analyzer.trajectories.items() if k[0] == group]
        steps = [
            sum(1 for m in t.get("messages", []) if m.get("role") == "assistant")
            for t in trajs
        ]
        data.append(steps)
        labels.append(group)

    ax.boxplot(data, labels=labels)
    ax.set_ylabel("Steps")
    ax.set_title("Steps Distribution by Group")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(output_dir / "steps_distribution.png", dpi=150)
    plt.close()


def plot_cost_vs_steps(analyzer: HybridAblationAnalyzer, output_dir: Path):
    """Scatter plot of cost vs steps."""
    import matplotlib.pyplot as plt

    groups = sorted(set(k[0] for k in analyzer.trajectories.keys()))
    colors = plt.cm.tab10.colors

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, group in enumerate(groups):
        trajs = [v for k, v in analyzer.trajectories.items() if k[0] == group]
        steps = []
        costs = []
        for t in trajs:
            s = sum(1 for m in t.get("messages", []) if m.get("role") == "assistant")
            c = t.get("info", {}).get("model_stats", {}).get("instance_cost", 0)
            steps.append(s)
            costs.append(c)
        ax.scatter(steps, costs, label=group, alpha=0.6, color=colors[i % len(colors)])

    ax.set_xlabel("Steps")
    ax.set_ylabel("Cost ($)")
    ax.set_title("Cost vs Steps by Group")
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "cost_vs_steps.png", dpi=150)
    plt.close()


def plot_paging_ops_distribution(analyzer: HybridAblationAnalyzer, output_dir: Path):
    """Plot manual paging ops distribution."""
    import matplotlib.pyplot as plt

    groups = sorted(set(k[0] for k in analyzer.trajectories.keys()))

    fig, ax = plt.subplots(figsize=(10, 6))

    data = []
    labels = []
    for group in groups:
        trajs = [v for k, v in analyzer.trajectories.items() if k[0] == group]
        paging_ops = [
            analyzer.parse_paging_from_messages(t.get("messages", []))["manual_paging_ops"]
            for t in trajs
        ]
        data.append(paging_ops)
        labels.append(group)

    ax.boxplot(data, labels=labels)
    ax.set_ylabel("Manual Paging Ops")
    ax.set_title("Paging Operations by Group")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(output_dir / "paging_ops_distribution.png", dpi=150)
    plt.close()


def plot_unique_lines_distribution(analyzer: HybridAblationAnalyzer, output_dir: Path):
    """Plot unique paging lines distribution."""
    import matplotlib.pyplot as plt

    groups = sorted(set(k[0] for k in analyzer.trajectories.keys()))

    fig, ax = plt.subplots(figsize=(10, 6))

    data = []
    labels = []
    for group in groups:
        trajs = [v for k, v in analyzer.trajectories.items() if k[0] == group]
        unique_lines = [
            analyzer.parse_paging_from_messages(t.get("messages", []))["paging_span_unique_lines"]
            for t in trajs
        ]
        data.append(unique_lines)
        labels.append(group)

    ax.boxplot(data, labels=labels)
    ax.set_ylabel("Unique Paging Lines")
    ax.set_title("Unique Lines Read by Group")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(output_dir / "unique_lines_distribution.png", dpi=150)
    plt.close()


def plot_front_25_explore(analyzer: HybridAblationAnalyzer, output_dir: Path):
    """Plot front-25% exploration ratio."""
    import matplotlib.pyplot as plt

    groups = sorted(set(k[0] for k in analyzer.trajectories.keys()))

    fig, ax = plt.subplots(figsize=(10, 6))

    ratios = []
    labels = []
    for group in groups:
        trajs = [v for k, v in analyzer.trajectories.items() if k[0] == group]
        group_ratios = [
            analyzer.compute_front_25_explore(t.get("messages", []))
            for t in trajs
        ]
        ratios.append(sum(group_ratios) / len(group_ratios) if group_ratios else 0)
        labels.append(group)

    ax.bar(labels, ratios)
    ax.set_ylabel("Explore Ratio")
    ax.set_title("Front 25% Steps: Exploration Ratio by Group")
    ax.set_ylim(0, 1)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(output_dir / "front_25_explore.png", dpi=150)
    plt.close()


def generate_all_plots(results_dir: Path, output_dir: Path):
    """Generate all plots for the ablation analysis."""
    analyzer = HybridAblationAnalyzer(results_dir)
    analyzer.load_all_trajectories()

    output_dir.mkdir(parents=True, exist_ok=True)

    plot_steps_distribution(analyzer, output_dir)
    plot_cost_vs_steps(analyzer, output_dir)
    plot_paging_ops_distribution(analyzer, output_dir)
    plot_unique_lines_distribution(analyzer, output_dir)
    plot_front_25_explore(analyzer, output_dir)

    print(f"Plots saved to {output_dir}")
```

### 5.2 Required Plots Checklist

| Plot | Filename | Description |
|------|----------|-------------|
| Steps Distribution | `steps_distribution.png` | Boxplot per group |
| Cost vs Steps | `cost_vs_steps.png` | Scatter with group colors |
| Paging Ops | `paging_ops_distribution.png` | Boxplot of nl\|sed counts |
| Unique Lines | `unique_lines_distribution.png` | Boxplot of union-merged lines |
| Front 25% Explore | `front_25_explore.png` | Bar chart of explore ratio |

### 5.3 Metrics Table Output Format

```
| Group | N | Success% | Avg Steps | Median | Avg Cost | Paging Ops | Unique Lines | Read Dup% | Front25 Explore |
|-------|---|----------|-----------|--------|----------|------------|--------------|-----------|-----------------|
| v0 | 60 | 45.0% | 32.5 | 28 | $0.042 | 8.3 | 420 | 23.5% | 38.2% |
| prefetch_v2_all | 60 | 48.3% | 28.1 | 24 | $0.035 | 4.1 | 380 | 15.2% | 32.1% |
```

### 5.4 CLI Entry Point

```python
# Add to summarize_hybrid_ablation.py

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("analysis/ablation"))
    parser.add_argument("--plots", action="store_true", help="Generate plots")
    args = parser.parse_args()

    analyzer = HybridAblationAnalyzer(args.results_dir)
    analyzer.generate_report(args.output_dir / "ABLATION_REPORT.md")

    if args.plots:
        generate_all_plots(args.results_dir, args.output_dir / "plots")
```

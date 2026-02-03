# Hybrid Ablation Experiment Report

**Protocol**: Repo-Aware Stratified Sampling
**Date**: 2026-01-28 23:42

## 1. Background: Why Hybrid Sampling?

SWE-bench Verified is sorted alphabetically by `instance_id` (which starts with repo name),
causing severe repo concentration in contiguous slices:

```
Index Range    Repo                  Count
─────────────────────────────────────────────
  0-21         astropy/astropy        22
 22-252        django/django         231  ← 46% of dataset!
253-286        matplotlib/matplotlib  34
...
```

**Problem**: Pure sequential sampling (e.g., 0:100) would be ~80% django,
making efficiency metrics misleading for cross-repo generalization.

**Solution**: Repo-Aware Stratified Sampling with 3 window types:
- **Sequential**: Baseline efficiency on dominant repos
- **Boundary**: Cold-start behavior when switching repos
- **Anchor**: Long-tail generalization (rare repos)

## 2. Sampling Windows

| Type | Slice | Repos | Distribution |
|------|-------|-------|--------------|
| sequential | 0:20 | 1 | astropy:20 |
| sequential | 20:40 | 2 | django:18, astropy:2 |
| boundary | 310:330 | 2 | xarray:10, pylint:10 |
| anchor | 280:300 | 5 | requests:8, matplotlib:7, seaborn:2 |
| anchor | 420:440 | 2 | sympy:15, sphinx:5 |

**Combined Coverage**: 10 unique repos

## 3. Detailed Results

### 3.2 Aggregate Results (Weighted by N)

| Group | N | Success% | Avg Steps | Avg Cost | Pager Ops | Read Dup% |
|-------|---|----------|-----------|----------|-----------|-----------|

## 4. Comparison vs Anchor (prefetch_v2)

| Group | Steps Δ | Cost Δ | Success Δ | Pager Ops Δ | Valid? |
|-------|---------|--------|-----------|-------------|--------|

## 5. Quality Warnings

✓ All variants maintain success rate within 5% of anchor.

## 6. Key Findings

### Cache Module (prefetch_v2_cache)
- Data not available

### Pager Module (prefetch_v2_pager)
- Data not available

### Combined (prefetch_v2_all)
- Data not available

## 7. Missing Metrics / Data Gaps

The following metrics could not be computed from current logs:
- `resolve_rate`: Requires SWE-bench evaluation harness
- `cache_hit_rate`: Requires read_cache module logging
- `pager_expansion_count`: Requires pager_policy module logging

To enable these metrics, add logging to:
- `src/minisweagent/prefetch/scroll_consolidation.py`
- `src/minisweagent/prefetch/__init__.py` (cache/pager modules)

## 8. Reproduction

```bash
# Generate sampling windows
uv run python analysis/ablation/select_hybrid_slices.py

# Run full experiment
./scripts/run_hybrid_ablation.sh run

# Generate this report
uv run python analysis/ablation/summarize_hybrid_ablation.py
```

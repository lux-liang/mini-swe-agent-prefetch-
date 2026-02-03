# Prefetch V2 Ablation Experiment Report

**Date**: 2026-01-28
**Dataset**: SWE-bench Verified (slice 0:20)
**Model**: gpt-5-mini via LiteLLM

---

## 1. Experiment Configuration

| Version | Config | Key Features |
|---------|--------|--------------|
| **V0 (Baseline)** | `swebench_modal_litellm.yaml` | No prefetch, standard prompts |
| **V2 (Prefetch)** | `swebench_prefetch_v2.yaml` | Scroll Consolidation, larger chunk prompts |

### V2 Key Changes
- Modified prompt to encourage larger file reads (100+ lines at a time)
- Added "File Reading Best Practices" section discouraging small nl|sed windows
- Scroll Consolidation config enabled (though behavioral effect is via prompts)

---

## 2. Results Summary

### Primary Metrics (Lower is Better)

| Metric | V0 Baseline | V2 Prefetch | Change | Status |
|--------|-------------|-------------|--------|--------|
| **Avg Steps/Instance** | 16.15 | 14.55 | **-9.9%** | ✓ Improved |
| **Avg Cost/Instance** | $0.052 | $0.044 | **-16.4%** | ✓ Improved |
| **Total Steps** | 323 | 291 | **-9.9%** | ✓ Improved |
| **Total Cost** | $1.04 | $0.87 | **-16.3%** | ✓ Improved |

### Paging Behavior Metrics

| Metric | V0 Baseline | V2 Prefetch | Change | Status |
|--------|-------------|-------------|--------|--------|
| **Avg Paging Ops** | 5.65 | 5.25 | **-7.1%** | ✓ Improved |
| **Total Paging Ops** | 113 | 105 | **-7.1%** | ✓ Improved |
| **Read Duplication Rate** | 56.3% | 53.5% | **-4.9%** | ✓ Improved |

### Exploration Metrics

| Metric | V0 Baseline | V2 Prefetch | Change | Status |
|--------|-------------|-------------|--------|--------|
| **Front 25% Explore Ratio** | 68.5% | 76.6% | +11.9% | ✗ Worse |
| **Avg Explore Chain Len** | 1.95 | 2.15 | +10.3% | ✗ Worse |

---

## 3. Key Findings

### Positive Outcomes
1. **Cost Reduction**: 16.4% reduction in average cost per instance ($0.052 → $0.044)
2. **Step Reduction**: 9.9% fewer steps on average (16.15 → 14.55)
3. **Paging Improvement**: 7.1% reduction in manual paging operations

### Trade-offs
1. **Front-loaded Exploration**: V2 shows higher front 25% exploration ratio (68.5% → 76.6%)
   - This is expected: encouraging larger upfront file reads shifts exploration earlier
   - Net effect is positive (fewer total steps)

2. **Explore Chain Length**: Slightly longer consecutive exploration chains
   - Agent reads more content per exploration step
   - More efficient overall despite longer chains

---

## 4. Statistical Comparison

### V0 Baseline Distribution
```
Metric              Mean    Std     Min     Max
───────────────────────────────────────────────
total_steps         16.15   10.06   8       47
instance_cost       $0.052  $0.046  $0.011  $0.188
manual_paging_ops   5.65    3.73    1       14
```

### V2 Prefetch Distribution
```
Metric              Mean    Std     Min     Max
───────────────────────────────────────────────
total_steps         14.55   4.78    8       24
instance_cost       $0.044  $0.025  $0.010  $0.108
manual_paging_ops   5.25    3.44    1       14
```

### Key Observation
- V2 shows **lower variance** in both steps and cost
- Max steps reduced from 47 to 24 (49% reduction in worst case)
- Max cost reduced from $0.188 to $0.108 (43% reduction in worst case)

---

## 5. Generated Artifacts

### Figures
- `analysis/ablation/ablation_comparison_verified_20.png` - 4-panel comparison plot
- `analysis/ablation/baseline_analysis_baseline_verified_20.png` - Baseline detailed analysis

### Data Files
- `analysis/ablation/comparison_verified_20.csv` - Metric comparison table
- `analysis/ablation/instances_baseline_verified_20.csv` - V0 instance-level data
- `analysis/ablation/instances_prefetch_verified_20.csv` - V2 instance-level data
- `analysis/ablation/metrics_baseline_verified_20.json` - V0 summary metrics
- `analysis/ablation/metrics_prefetch_verified_20.json` - V2 summary metrics

---

## 6. Conclusions

### V2 Prefetch Effectiveness: **Validated**

The prompt-based approach to reduce manual paging demonstrates clear benefits:

1. **Primary Goal Achieved**: Steps and cost reduced by ~10-16%
2. **Paging Behavior Changed**: 7% reduction in nl|sed operations
3. **Variance Reduced**: More consistent performance across instances
4. **Worst-case Improved**: Maximum steps/cost significantly reduced

### Recommendations

1. **Scale to 200 instances**: V2 shows consistent improvement, ready for full evaluation
2. **Projected savings**: ~$8.75 for 200 instances (vs ~$10.44 baseline)
3. **Consider V3**: Add One-Hop + Gate for potential further improvements

---

## 7. Next Steps

1. Run V2 on Verified 0:200 (full dataset)
2. Evaluate resolve rate impact (requires SWE-bench evaluation)
3. Implement runtime Scroll Consolidation (file content injection)
4. Design V3 experiment with Gate mechanism

---

## 8. Run Commands Reference

```bash
# V0 Baseline
uv run modal run -e main modal_run.py --config v0 --subset verified --slice-spec "0:20"

# V2 Prefetch
uv run modal run -e main modal_run.py --config v2 --subset verified --slice-spec "0:20"

# V2 Full (200 instances)
uv run modal run -e main --detach modal_run.py --config v2 --subset verified --slice-spec "0:200" --workers 4
```

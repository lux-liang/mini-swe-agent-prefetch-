# Hybrid Ablation Experiment Report

**Date**: 2026-01-30 00:13
**Groups**: prefetch_v2_all_0_20, prefetch_v2_all_20_40, prefetch_v2_all_280_300, v0
**Total Trajectories**: 87

---

## 1. Group Summaries

### prefetch_v2_all_0_20

| Metric | Value |
|--------|-------|
| Instances | 20 |
| Success Rate | 100.0% |
| Avg Steps | 11.0 |
| Median Steps | 10.0 |
| Avg Cost | $0.0367 |
| Avg Paging Ops | 0.0 |
| Avg Unique Lines | 0 |
| Read Dup Rate | 0.0% |
| Front 25% Explore | 84.8% |

### prefetch_v2_all_20_40

| Metric | Value |
|--------|-------|
| Instances | 20 |
| Success Rate | 100.0% |
| Avg Steps | 10.9 |
| Median Steps | 8.5 |
| Avg Cost | $0.0291 |
| Avg Paging Ops | 0.1 |
| Avg Unique Lines | 24 |
| Read Dup Rate | 0.0% |
| Front 25% Explore | 89.5% |

### prefetch_v2_all_280_300

| Metric | Value |
|--------|-------|
| Instances | 3 |
| Success Rate | 66.7% |
| Avg Steps | 15.0 |
| Median Steps | 14.0 |
| Avg Cost | $0.0426 |
| Avg Paging Ops | 0.0 |
| Avg Unique Lines | 0 |
| Read Dup Rate | 0.0% |
| Front 25% Explore | 88.9% |

### v0

| Metric | Value |
|--------|-------|
| Instances | 44 |
| Success Rate | 90.9% |
| Avg Steps | 11.5 |
| Median Steps | 10.5 |
| Avg Cost | $0.0324 |
| Avg Paging Ops | 3.9 |
| Avg Unique Lines | 552 |
| Read Dup Rate | 10.8% |
| Front 25% Explore | 80.6% |

---

## 2. Paired Comparisons (vs v0)

### prefetch_v2_all_0_20 vs v0

**Paired instances**: 20

| Metric | Mean Δ | Median Δ | Direction |
|--------|--------|----------|-----------|
| Steps | -1.55 | -1.50 | ↓ better |
| Cost ($) | -0.00 | -0.00 | ↓ better |
| Paging Ops | -4.45 | -4.00 | ↓ better |
| Unique Lines | -626.35 | -490.00 | ↓ better |
| Read Dup Rate | -0.15 | -0.09 | ↓ better |

### prefetch_v2_all_20_40 vs v0

**Paired instances**: 17

| Metric | Mean Δ | Median Δ | Direction |
|--------|--------|----------|-----------|
| Steps | -0.65 | +0.00 | ↓ better |
| Cost ($) | -0.00 | +0.00 | ↓ better |
| Paging Ops | -3.71 | -2.00 | ↓ better |
| Unique Lines | -494.94 | -400.00 | ↓ better |
| Read Dup Rate | -0.09 | -0.04 | ↓ better |

### prefetch_v2_all_280_300 vs v0

**Paired instances**: 2

| Metric | Mean Δ | Median Δ | Direction |
|--------|--------|----------|-----------|
| Steps | -8.50 | -8.50 | ↓ better |
| Cost ($) | -0.02 | -0.02 | ↓ better |
| Paging Ops | -4.00 | -4.00 | ↓ better |
| Unique Lines | -670.00 | -670.00 | ↓ better |
| Read Dup Rate | -0.07 | -0.07 | ↓ better |

- **Regressions**: 0
- **Improvements**: 1

---

## 3. Summary Table

| Group | N | Success% | Avg Steps | Median | Avg Cost | Paging Ops | Unique Lines | Read Dup% |
|-------|---|----------|-----------|--------|----------|------------|--------------|-----------|
| prefetch_v2_all_0_20 | 20 | 100.0% | 11.0 | 10 | $0.0367 | 0.0 | 0 | 0.0% |
| prefetch_v2_all_20_40 | 20 | 100.0% | 10.9 | 8 | $0.0291 | 0.1 | 24 | 0.0% |
| prefetch_v2_all_280_300 | 3 | 66.7% | 15.0 | 14 | $0.0426 | 0.0 | 0 | 0.0% |
| v0 | 44 | 90.9% | 11.5 | 10 | $0.0324 | 3.9 | 552 | 10.8% |

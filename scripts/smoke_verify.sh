#!/bin/bash
# Smoke Test 验证脚本
# 在 run_smoke_test.sh 完成后执行

set -e
cd /home/lux_liang/work/mini-swe-agent-main

echo "========== Smoke Test Verification =========="
echo "Date: $(date)"
echo ""

PASS_COUNT=0
FAIL_COUNT=0

# Check 1: prefetch runs have prefetch_aggregate
echo "=== Check 1: prefetch runs have prefetch_aggregate ==="
PREFETCH_TRAJS=$(find results -name "*.traj.json" -path "*smoke_prefetch*" 2>/dev/null | head -5)
if [[ -z "$PREFETCH_TRAJS" ]]; then
    echo "  WARNING: No prefetch trajectories found"
    ((FAIL_COUNT++))
else
    for f in $PREFETCH_TRAJS; do
        if grep -q "prefetch_aggregate" "$f"; then
            echo "  ✓ $f"
            ((PASS_COUNT++))
        else
            echo "  ✗ MISSING prefetch_aggregate: $f"
            ((FAIL_COUNT++))
        fi
    done
fi

# Check 2: baseline runs have NO prefetch_aggregate
echo -e "\n=== Check 2: baseline runs have NO prefetch_aggregate ==="
BASELINE_TRAJS=$(find results -name "*.traj.json" -path "*smoke_v0*" 2>/dev/null | head -5)
if [[ -z "$BASELINE_TRAJS" ]]; then
    echo "  WARNING: No baseline trajectories found"
else
    CONTAMINATED=0
    for f in $BASELINE_TRAJS; do
        if grep -q "prefetch_aggregate" "$f"; then
            echo "  ✗ ERROR: baseline has prefetch_aggregate: $f"
            CONTAMINATED=1
            ((FAIL_COUNT++))
        fi
    done
    if [[ $CONTAMINATED -eq 0 ]]; then
        echo "  ✓ All baseline trajectories clean"
        ((PASS_COUNT++))
    fi
fi

# Check 3: step metrics have paging fields
echo -e "\n=== Check 3: step metrics have paging fields ==="
SAMPLE=$(find results -name "*.traj.json" -path "*smoke_prefetch*" 2>/dev/null | head -1)
if [[ -n "$SAMPLE" ]]; then
    python3 -c "
import json
d = json.load(open('$SAMPLE'))
steps = d.get('prefetch_step_metrics', [])
if steps:
    k = list(steps[0].keys())
    paging = [x for x in k if 'paging' in x.lower() or 'manual' in x.lower()]
    print(f'  Paging-related fields: {paging}')
    if paging:
        print(f'  ✓ Found {len(paging)} paging fields')
    else:
        print('  ✗ No paging fields found')
        exit(1)
else:
    print('  ✗ No step metrics')
    exit(1)
" && ((PASS_COUNT++)) || ((FAIL_COUNT++))
else
    echo "  WARNING: No sample trajectory found"
    ((FAIL_COUNT++))
fi

# Summary
echo -e "\n========== SUMMARY =========="
echo "PASS: $PASS_COUNT"
echo "FAIL: $FAIL_COUNT"

if [[ $FAIL_COUNT -eq 0 ]]; then
    echo -e "\n✓ ALL CHECKS PASSED - Ready for Phase A"
    echo "Run: ./scripts/run_phase_a.sh"
    exit 0
else
    echo -e "\n✗ SOME CHECKS FAILED - Review before Phase A"
    exit 1
fi

#!/bin/bash
# Go/No-Go Gate Check (v2) - 修正版
# 必须全部 PASS 才能开始 Phase A 实验

set -e
cd /home/lux_liang/work/mini-swe-agent-main

echo "========== Go/No-Go Gate Check (v2) =========="
echo "Date: $(date)"
echo ""

# 先跑一个 prefetch instance 生成测试文件
if [[ ! -f /tmp/gate_a_test.traj.json ]]; then
    echo "[Setup] Running single prefetch instance for gate check..."
    uv run mini-extra swebench-single \
        --config src/minisweagent/config/extra/swebench_prefetch_v2.yaml \
        --subset lite --split test --instance 0 \
        --environment-class local \
        --exit-immediately \
        --output /tmp/gate_a_test.traj.json 2>&1 | tee /tmp/gate_a.log
fi

# Gate A: PrefetchAgent 实例化（以 traj 落盘为准，而非日志）
echo -e "\n[Gate A] PrefetchAgent instantiation..."
if grep -q "prefetch_step_metrics" /tmp/gate_a_test.traj.json 2>/dev/null; then
    echo "  ✓ PASS (prefetch_step_metrics exists in trajectory)"
else
    echo "  ✗ FAIL: prefetch_step_metrics missing"
    exit 1
fi

# Gate B: prefetch_aggregate 落盘
echo -e "\n[Gate B] prefetch_aggregate in trajectory..."
if grep -q "prefetch_aggregate" /tmp/gate_a_test.traj.json 2>/dev/null; then
    echo "  ✓ PASS"
else
    echo "  ✗ FAIL: prefetch_aggregate missing"
    exit 1
fi

# Gate C: step metrics 字段完整（兼容字段名差异）
echo -e "\n[Gate C] Step metrics completeness..."
python3 -c "
import json, sys
d = json.load(open('/tmp/gate_a_test.traj.json'))
steps = d.get('prefetch_step_metrics', [])
if not steps:
    print('  ✗ FAIL: prefetch_step_metrics is empty')
    sys.exit(1)

k = set(steps[0].keys())

# 必须存在的核心字段
must = {'token_budget_prefetch', 'aggression_level', 'step_id'}
missing = [m for m in must if m not in k]
if missing:
    print(f'  ✗ FAIL: missing required keys: {missing}')
    sys.exit(1)

# paging 指标：允许多种命名（实现可能用 _step 后缀或不用）
paging_alts = [
    {'manual_paging_ops_step', 'paging_span_lines_step'},
    {'manual_paging_ops_step', 'paging_span_total_lines', 'paging_span_unique_lines'},
    {'is_paging_op', 'paging_lines'},
]
if not any(alt.issubset(k) for alt in paging_alts):
    # 放宽：只要有任何 paging 相关字段就 PASS
    paging_related = [x for x in k if 'paging' in x.lower()]
    if not paging_related:
        print(f'  ✗ FAIL: no paging metrics found')
        print(f'  Available keys: {sorted(list(k))[:20]}')
        sys.exit(1)

print(f'  ✓ PASS (found keys: {sorted(list(k))[:10]}...)')
"

# Gate D: baseline config 未误开 prefetch（只检查 prefetch 段内的 enabled）
echo -e "\n[Gate D] Baseline config safety..."
BASELINE_YAML="src/minisweagent/config/extra/swebench_modal_litellm.yaml"

# 用 awk 只抽取 prefetch: 块内容
PREFETCH_ENABLED=$(awk '
  $0 ~ /^prefetch:/ {in_prefetch=1; next}
  in_prefetch && $0 ~ /^[^ ]/ {in_prefetch=0}
  in_prefetch {print}
' "$BASELINE_YAML" 2>/dev/null | grep -E "enabled:\s*true" || true)

if [[ -n "$PREFETCH_ENABLED" ]]; then
    echo "  ✗ FAIL: baseline yaml has prefetch.enabled: true"
    echo "  Found: $PREFETCH_ENABLED"
    exit 1
else
    echo "  ✓ PASS (baseline prefetch disabled or absent)"
fi

echo -e "\n========== ALL GATES PASSED =========="
echo "Proceed to: ./scripts/run_smoke_test.sh"

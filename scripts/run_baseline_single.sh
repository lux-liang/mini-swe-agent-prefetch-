#!/bin/bash
# 运行 SWE-bench 测试的脚本（修复代理问题）
# 用法:
#   bash scripts/run_baseline_single.sh [instance_id]
#   默认: django__django-11039

set -euo pipefail

# 修正 NO_PROXY 环境变量（移除 IPv6 [::1] 格式，httpx 无法解析）
export NO_PROXY="localhost,127.0.0.1"
export no_proxy="localhost,127.0.0.1"

# 确保 API 配置已加载
CONFIG_FILE="$HOME/.config/mini-swe-agent/.env"
if [ -f "$CONFIG_FILE" ]; then
    while IFS='=' read -r key value; do
        [[ "$key" =~ ^#.*$ || -z "$key" ]] && continue
        value=$(echo "$value" | sed "s/^['\"]//;s/['\"]$//")
        case "$key" in
            api_key)  export OPENAI_API_KEY="$value" ;;
            base_url) export OPENAI_API_BASE="$value"; export LITELLM_BASE_URL="$value" ;;
        esac
    done < "$CONFIG_FILE"
fi

INSTANCE="${1:-django__django-11039}"
OUTPUT="results/test_${INSTANCE//\//_}.json"

cd ~/work/mini-swe-agent-main

echo "=== 运行单个 instance 测试 ==="
echo "Instance: $INSTANCE"
echo "Output:   $OUTPUT"
echo ""

env NO_PROXY="localhost,127.0.0.1" no_proxy="localhost,127.0.0.1" \
  uv run mini-extra swebench-single \
  --config src/minisweagent/config/extra/swebench_baseline.yaml \
  --instance "$INSTANCE" --split test \
  --output "$OUTPUT" \
  --exit-immediately

echo ""
echo "=== 测试完成 ==="
echo "结果文件: $OUTPUT"
ls -lh "$OUTPUT"

# 显示关键指标
python3 -c "
import json, sys
with open('$OUTPUT') as f:
    d = json.load(f)
info = d['info']
print(f\"exit_status: {info.get('exit_status')}\")
print(f\"api_calls:   {info['model_stats']['api_calls']}\")
print(f\"cost:        \${info['model_stats'].get('instance_cost', 0):.4f}\")
has_sub = bool(info.get('submission'))
print(f\"has_patch:   {has_sub}\")
"

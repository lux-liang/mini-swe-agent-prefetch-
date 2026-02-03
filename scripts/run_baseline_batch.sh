#!/bin/bash
# 运行 SWE-bench Lite 批量测试 (20 instances)
# 用法:
#   bash scripts/run_baseline_batch.sh [start:end] [workers]
#   默认: 0:20 (前20个instance), 4个并行worker

set -euo pipefail

SLICE="${1:-0:20}"
WORKERS="${2:-4}"
OUTPUT_DIR="results/baseline_$(date +%Y%m%d_%H%M%S)"

# 修正 NO_PROXY 环境变量（移除 IPv6 [::1] 格式，httpx 无法解析）
export NO_PROXY="localhost,127.0.0.1"
export no_proxy="localhost,127.0.0.1"

# 加载 API 配置
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

cd ~/work/mini-swe-agent-main

echo "=========================================="
echo "SWE-bench Lite Baseline Batch Run"
echo "=========================================="
echo "Slice:      $SLICE"
echo "Workers:    $WORKERS"
echo "Output:     $OUTPUT_DIR"
echo "Config:     src/minisweagent/config/extra/swebench_baseline.yaml"
echo ""
echo "Starting at $(date)"
echo "=========================================="

mkdir -p "$OUTPUT_DIR"

# 运行批量测试
env NO_PROXY="localhost,127.0.0.1" no_proxy="localhost,127.0.0.1" \
    OPENAI_API_KEY="$OPENAI_API_KEY" \
    OPENAI_API_BASE="$OPENAI_API_BASE" \
    uv run mini-extra swebench \
    --config src/minisweagent/config/extra/swebench_baseline.yaml \
    --subset lite --split test --slice "$SLICE" \
    -w "$WORKERS" -o "$OUTPUT_DIR" 2>&1 | tee "$OUTPUT_DIR/run.log"

echo ""
echo "=========================================="
echo "Completed at $(date)"
echo "=========================================="

# 生成摘要报告
echo ""
echo "Generating summary report..."
python3 scripts/summarize_baseline.py "$OUTPUT_DIR" -o "$OUTPUT_DIR/summary.md"

echo ""
echo "Results saved to: $OUTPUT_DIR"
echo "Summary: $OUTPUT_DIR/summary.md"
ls -la "$OUTPUT_DIR"

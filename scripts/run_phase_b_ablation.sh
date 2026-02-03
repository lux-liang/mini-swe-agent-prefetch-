#!/bin/bash
# Phase B: Ablation Study on Slice 20:40
# Tests all 4 prefetch configurations on single slice for detailed comparison
#
# Configs:
#   - prefetch_v2 (base, anchor)
#   - prefetch_v2_cache (+ read cache)
#   - prefetch_v2_pager (+ pager policy)
#   - prefetch_v2_all (all features)
#
# Usage: ./scripts/run_phase_b_ablation.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_blue() { echo -e "${BLUE}[RUN]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }

cd "$PROJECT_ROOT"

echo "============================================================"
echo "  Phase B: Ablation Study"
echo "  Slice: 20:40 (django + astropy boundary)"
echo "  Configs: prefetch_v2, prefetch_v2_cache, prefetch_v2_pager, prefetch_v2_all"
echo "============================================================"
echo ""

# Fixed slice for ablation
SLICE="20:40"
START="${SLICE%%:*}"
END="${SLICE##*:}"
SLICE_TAG="${START}_${END}"

# All ablation configs
CONFIGS=(
    "prefetch_v2"
    "prefetch_v2_cache"
    "prefetch_v2_pager"
    "prefetch_v2_all"
)

WORKERS=2
STAGGER_SECONDS=45

LAUNCHED=()

for CONFIG in "${CONFIGS[@]}"; do
    RUN_TAG="${CONFIG}_verified_${SLICE_TAG}"

    # Check if already exists
    if uv run modal volume ls research-data 2>/dev/null | grep -q "$RUN_TAG"; then
        log_warn "Skipping (exists): $RUN_TAG"
        continue
    fi

    log_blue "Launching: $RUN_TAG"

    uv run modal run --detach -e main modal_run.py \
        --config "$CONFIG" \
        --subset verified \
        --slice-spec "$SLICE" \
        --workers $WORKERS \
        --run-tag "$RUN_TAG" &

    LAUNCHED+=("$RUN_TAG")

    # Stagger launches
    log_info "Waiting ${STAGGER_SECONDS}s before next launch..."
    sleep $STAGGER_SECONDS
done

echo ""
echo "============================================================"
echo "  Phase B Launch Complete"
echo "============================================================"
echo "  Launched: ${#LAUNCHED[@]} ablation runs"
echo ""
echo "  Monitor:"
echo "    uv run modal app list | grep swebench"
echo ""
echo "  Check results:"
echo "    uv run modal volume ls research-data | grep prefetch_v2"
echo ""

#!/bin/bash
# Phase A: Full Baseline vs Prefetch Comparison
# 2 configs × 3 slices × 20 instances = 120 total evaluations
#
# Prerequisites: Run smoke test first (./scripts/run_smoke_test.sh)
#
# Usage: ./scripts/run_phase_a.sh

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
echo "  Phase A: Baseline vs Prefetch Full Comparison"
echo "  Matrix: 2 configs × 3 slices = 6 runs (120 instances)"
echo "============================================================"
echo ""

# Full slices
SLICES=("0:20" "20:40" "280:300")
CONFIGS=("v0" "prefetch_v2_all")
WORKERS=2
STAGGER_SECONDS=60

# Track launched runs
LAUNCHED=()

for SLICE in "${SLICES[@]}"; do
    START="${SLICE%%:*}"
    END="${SLICE##*:}"
    SLICE_TAG="${START}_${END}"

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

        # Stagger launches to avoid cold-start storm
        log_info "Waiting ${STAGGER_SECONDS}s before next launch..."
        sleep $STAGGER_SECONDS
    done
done

echo ""
echo "============================================================"
echo "  Phase A Launch Complete"
echo "============================================================"
echo "  Launched: ${#LAUNCHED[@]} runs"
echo ""
echo "  Monitor progress:"
echo "    uv run modal app list"
echo ""
echo "  Check results:"
echo "    uv run modal volume ls research-data | grep -E '(v0|prefetch)'"
echo ""

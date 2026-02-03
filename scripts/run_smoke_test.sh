#!/bin/bash
# Phase A: Smoke Test for LSR-Engine v2 Ablation
# Tests 2 instances per slice with single worker to verify Modal stability
#
# Usage: ./scripts/run_smoke_test.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

cd "$PROJECT_ROOT"

echo "============================================================"
echo "  LSR-Engine v2 Smoke Test"
echo "  Testing: 2 instances × 3 slices × 2 configs = 12 instances"
echo "============================================================"

# Smoke slices: first 2 instances of each main slice
SMOKE_SLICES=("0:2" "20:22" "280:282")
CONFIGS=("v0" "prefetch_v2_all")

SUCCESS_COUNT=0
FAIL_COUNT=0

for SLICE in "${SMOKE_SLICES[@]}"; do
    START="${SLICE%%:*}"
    END="${SLICE##*:}"
    SLICE_TAG="${START}_${END}"

    for CONFIG in "${CONFIGS[@]}"; do
        RUN_TAG="smoke_${CONFIG}_${SLICE_TAG}"

        log_info "Running: $RUN_TAG"

        # Run with single worker, no detach (wait for completion)
        if uv run modal run -e main modal_run.py \
            --config "$CONFIG" \
            --subset verified \
            --slice-spec "$SLICE" \
            --workers 1 \
            --run-tag "$RUN_TAG" 2>&1 | tee -a "results/${RUN_TAG}.log"; then

            log_info "✓ Completed: $RUN_TAG"
            ((SUCCESS_COUNT++))
        else
            log_error "✗ Failed: $RUN_TAG"
            ((FAIL_COUNT++))
        fi

        # Brief pause between runs
        sleep 10
    done
done

echo ""
echo "============================================================"
echo "  Smoke Test Complete"
echo "============================================================"
echo "  Success: $SUCCESS_COUNT"
echo "  Failed:  $FAIL_COUNT"
echo ""

if [ $FAIL_COUNT -gt 2 ]; then
    log_error "Too many failures. Check Modal status before proceeding."
    exit 1
else
    log_info "Smoke test passed. Safe to proceed with full run."
fi

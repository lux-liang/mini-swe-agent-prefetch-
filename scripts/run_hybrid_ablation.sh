#!/bin/bash
# Hybrid Ablation Experiment Runner
# Runs Prefetch V2 ablation across 5 stratified sampling windows
#
# Protocol: Repo-Aware Stratified Sampling
# - Sequential: 0:20, 20:40
# - Boundary: crosses repo boundary (auto-selected)
# - Anchors: from index >= 260 (long-tail repos)
#
# Groups:
# - prefetch_v2 (anchor baseline)
# - prefetch_v2_cache
# - prefetch_v2_pager
# - prefetch_v2_all

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SLICES_JSON="$PROJECT_ROOT/analysis/ablation/slices_selected.json"
RESULTS_DIR="$PROJECT_ROOT/results"
WORKERS=2

# Config files
declare -A CONFIGS=(
    ["prefetch_v2"]="src/minisweagent/config/extra/swebench_prefetch_v2.yaml"
    ["prefetch_v2_cache"]="src/minisweagent/config/extra/swebench_prefetch_v2_cache.yaml"
    ["prefetch_v2_pager"]="src/minisweagent/config/extra/swebench_prefetch_v2_pager.yaml"
    ["prefetch_v2_all"]="src/minisweagent/config/extra/swebench_prefetch_v2_all.yaml"
)

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_banner() {
    echo ""
    echo "============================================================"
    echo "  Prefetch V2 Hybrid Ablation Experiment"
    echo "  Protocol: Repo-Aware Stratified Sampling"
    echo "============================================================"
    echo ""
}

# Step 1: Generate slices if not exist
generate_slices() {
    log_info "Checking slices configuration..."

    if [[ ! -f "$SLICES_JSON" ]]; then
        log_info "Generating hybrid sampling windows..."
        cd "$PROJECT_ROOT"
        uv run python analysis/ablation/select_hybrid_slices.py \
            --output "$SLICES_JSON" \
            --seed 42
        log_success "Slices generated: $SLICES_JSON"
    else
        log_info "Using existing slices: $SLICES_JSON"
    fi

    # Read slices from JSON
    SLICES=$(uv run python -c "
import json
with open('$SLICES_JSON') as f:
    data = json.load(f)
print(' '.join(data['slices']))
")
    log_info "Slices to run: $SLICES"
}

# Step 2: Run experiment for one group × one slice
run_single() {
    local group="$1"
    local slice="$2"
    local config="${CONFIGS[$group]}"

    # Parse slice
    local start="${slice%%:*}"
    local end="${slice##*:}"
    local slice_tag="${start}_${end}"

    local output_dir="$RESULTS_DIR/${group}_verified_${slice_tag}"

    echo ""
    log_info "Running: group=$group slice=$slice"
    log_info "  Config: $config"
    log_info "  Output: $output_dir"

    # Check if already completed
    if [[ -d "$output_dir" ]] && [[ -f "$output_dir/summary.json" ]]; then
        log_warn "Output exists, skipping: $output_dir"
        return 0
    fi

    # Create output directory
    mkdir -p "$output_dir"

    # Run via Modal
    cd "$PROJECT_ROOT"
    uv run modal run -e main modal_run.py \
        --config "$group" \
        --subset verified \
        --slice-spec "$slice" \
        --workers "$WORKERS" \
        --run-tag "${group}_verified_${slice_tag}" \
        2>&1 | tee "$output_dir/run.log"

    local exit_code=${PIPESTATUS[0]}

    if [[ $exit_code -eq 0 ]]; then
        log_success "Completed: $group $slice"
    else
        log_error "Failed: $group $slice (exit=$exit_code)"
    fi

    return $exit_code
}

# Step 3: Run all experiments
run_all() {
    local groups=("prefetch_v2" "prefetch_v2_cache" "prefetch_v2_pager" "prefetch_v2_all")

    log_info "Starting hybrid ablation experiment..."
    log_info "Groups: ${groups[*]}"
    log_info "Workers: $WORKERS"

    local total_runs=$((${#groups[@]} * 5))
    local completed=0
    local failed=0

    for group in "${groups[@]}"; do
        echo ""
        echo "============================================================"
        echo "  Group: $group"
        echo "============================================================"

        for slice in $SLICES; do
            if run_single "$group" "$slice"; then
                ((completed++))
            else
                ((failed++))
            fi
            echo "Progress: $completed/$total_runs completed, $failed failed"
        done
    done

    echo ""
    echo "============================================================"
    echo "  EXPERIMENT COMPLETE"
    echo "============================================================"
    echo "  Total runs: $total_runs"
    echo "  Completed: $completed"
    echo "  Failed: $failed"
    echo "============================================================"
}

# Step 4: Run summarization
run_summary() {
    log_info "Running summarization..."
    cd "$PROJECT_ROOT"
    uv run python analysis/ablation/summarize_hybrid_ablation.py \
        --results-dir "$RESULTS_DIR" \
        --slices-json "$SLICES_JSON" \
        --output-dir "$PROJECT_ROOT/analysis/ablation"
}

# Main
main() {
    print_banner

    cd "$PROJECT_ROOT"

    # Parse arguments
    local cmd="${1:-all}"

    case "$cmd" in
        slices)
            generate_slices
            ;;
        run)
            generate_slices
            run_all
            ;;
        summary)
            run_summary
            ;;
        all)
            generate_slices
            run_all
            run_summary
            ;;
        *)
            echo "Usage: $0 [slices|run|summary|all]"
            echo ""
            echo "Commands:"
            echo "  slices  - Generate hybrid sampling windows only"
            echo "  run     - Run all experiments"
            echo "  summary - Generate summary report"
            echo "  all     - Run everything (default)"
            exit 1
            ;;
    esac
}

main "$@"

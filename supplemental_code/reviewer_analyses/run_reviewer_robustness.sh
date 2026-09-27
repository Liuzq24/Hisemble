#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN=${PYTHON_BIN:-python}
: "${DATA_ROOT:?Set DATA_ROOT to the directory containing ADATA/ and lsi_results/}"
OUTPUT_ROOT=${OUTPUT_ROOT:-./reviewer_robustness_results}
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

mkdir -p "$OUTPUT_ROOT/seed_stability" "$OUTPUT_ROOT/degraded_view"

export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export NUMEXPR_NUM_THREADS=${NUMEXPR_NUM_THREADS:-1}

"$PYTHON_BIN" "$SCRIPT_DIR/reviewer_seed_stability.py" \
  --data-root "$DATA_ROOT" \
  --output-dir "$OUTPUT_ROOT/seed_stability" \
  >"$OUTPUT_ROOT/seed_stability.log" 2>&1

"$PYTHON_BIN" "$SCRIPT_DIR/reviewer_degraded_view.py" \
  --data-root "$DATA_ROOT" \
  --output-dir "$OUTPUT_ROOT/degraded_view" \
  >"$OUTPUT_ROOT/degraded_view.log" 2>&1

"$PYTHON_BIN" "$SCRIPT_DIR/plot_reviewer_robustness.py" \
  --input-root "$OUTPUT_ROOT" \
  --output-dir "$OUTPUT_ROOT/figures" \
  >"$OUTPUT_ROOT/plotting.log" 2>&1

echo "Reviewer robustness analyses completed in $OUTPUT_ROOT"

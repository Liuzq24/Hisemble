#!/usr/bin/env bash
set -u

PYTHON_BIN=${PYTHON_BIN:-python}
: "${DATA_ROOT:?Set DATA_ROOT to the directory containing ADATA/ and lsi_results/}"
OUTPUT_DIR=${OUTPUT_DIR:-./component_scaling_results}
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
DATASET=${DATASET:-mca_Cerebellum_62216}
TIMEOUT_SECONDS=${TIMEOUT_SECONDS:-1800}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}

sample_sizes=(250 500 1000 1500 2000)
mkdir -p "$OUTPUT_DIR"
status_file="$OUTPUT_DIR/run_status.tsv"
printf "dataset\tvariant\tn_cells\tstatus\n" > "$status_file"

for n_cells in "${sample_sizes[@]}"; do
  variants=(full no_diffusion no_dynamic_sparsification)
  if (( n_cells <= 1000 )); then
    variants+=(dense_initial)
  fi
  for variant in "${variants[@]}"; do
    log="$OUTPUT_DIR/${DATASET}__${variant}__n${n_cells}.log"
    timeout "$TIMEOUT_SECONDS" "$PYTHON_BIN" "$SCRIPT_DIR/component_ablation.py" \
      --data-root "$DATA_ROOT" \
      --dataset "$DATASET" \
      --variant "$variant" \
      --sample-size "$n_cells" \
      --skip-clustering \
      --output-dir "$OUTPUT_DIR" >"$log" 2>&1
    status=$?
    printf "%s\t%s\t%s\t%s\n" "$DATASET" "$variant" "$n_cells" "$status" >> "$status_file"
  done
done

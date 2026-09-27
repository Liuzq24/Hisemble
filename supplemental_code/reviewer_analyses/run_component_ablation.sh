#!/usr/bin/env bash
set -u

PYTHON_BIN=${PYTHON_BIN:-python}
: "${DATA_ROOT:?Set DATA_ROOT to the directory containing ADATA/ and lsi_results/}"
OUTPUT_DIR=${OUTPUT_DIR:-./component_ablation_results}
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
TIMEOUT_SECONDS=${TIMEOUT_SECONDS:-2400}

datasets=(
  FCA_fetal_33184180_spleen
  FCA_fetal_33184180_cerebellum
  mca_Cerebellum_62216
  mca_LargeIntestineA_62816
)
variants=(full no_diffusion no_dynamic_sparsification)

mkdir -p "$OUTPUT_DIR"
status_file="$OUTPUT_DIR/run_status.tsv"
printf "dataset\tvariant\tstatus\n" > "$status_file"

for dataset in "${datasets[@]}"; do
  for variant in "${variants[@]}"; do
    log="$OUTPUT_DIR/${dataset}__${variant}.log"
    timeout "$TIMEOUT_SECONDS" "$PYTHON_BIN" "$SCRIPT_DIR/component_ablation.py" \
      --data-root "$DATA_ROOT" \
      --dataset "$dataset" \
      --variant "$variant" \
      --output-dir "$OUTPUT_DIR" >"$log" 2>&1
    status=$?
    printf "%s\t%s\t%s\n" "$dataset" "$variant" "$status" >> "$status_file"
  done
done

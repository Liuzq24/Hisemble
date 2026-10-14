# Hisemble reproduction tutorial

This tutorial runs on a MacBook through Docker Desktop, OrbStack, or another Docker-compatible runtime. It also works on Linux; Windows users may use WSL2. The validated MacBook environment uses Linux/arm64 containers. No GPU is needed for the Hisemble core.

The commands below run from the root of this GitHub checkout.

## 1. Obtain the code

Clone the repository and enter its root directory:

```bash
git clone https://github.com/Liuzq24/Hisemble.git
cd Hisemble
```

Use this checkout for the commands below. The corresponding journal archive is supplied as Supplemental Code.

## 2. Start Docker and build

Open Docker Desktop or OrbStack and wait for its engine to start. Then check:

```bash
docker version
docker build -f reproducibility/Dockerfile -t hisemble:reproduce .
```

The build uses the archived source and pinned core packages, plus the separate Scanpy/DPT dependency list. It accepts precomputed embeddings and does not train all upstream encoders. The first build downloads packages; subsequent builds can reuse the cache.

## 3. Run the checks

```bash
docker run --rm hisemble:reproduce \
  python -m unittest discover -s /opt/hisemble/tests -p 'test_*.py'
```

Expected outcome: `OK` (11 tests). These cover aligned input validation, NaN/Inf rejection, mismatched cells, degenerate embeddings, view-order invariance, the AnnData interface, and input-view order invariance in the production core.

A broader executable check additionally runs the four convergence-equivalence checks and both examples, compares serial and parallel seed results, verifies finite graph-based DPT, and checks numerical CLI entry points:

```bash
docker run --rm hisemble:reproduce \
  python /opt/hisemble/reproducibility/verify_installation.py
```

Expected final line begins `PASS: all packaged tests`. These quick checks use synthetic inputs; they do not require downloaded biological data.

## 4. Run Hisemble and save an AnnData result

```bash
mkdir -p example_output
docker run --rm \
  -v "$PWD/example_output:/output" \
  hisemble:reproduce \
  python /opt/hisemble/examples/quickstart_synthetic.py --output-dir /output
```

Expected output:

```text
cells=180
clusters=4
graph_shape=(180, 180)
graph_nnz=7372
```

The file `example_output/synthetic_hisemble.h5ad` stores the input embeddings in `.obsm`, the consensus graph in `.obsp['connectivities']`, distances in `.obsp['distances']`, Leiden labels in `.obs['hisemble_clusters']`, and neighbor metadata in `.uns['neighbors']`.

## 5. Re-run from a saved input file

This exercises the command used for your own aligned embeddings:

```bash
docker run --rm \
  -v "$PWD/example_output:/data" \
  hisemble:reproduce \
  python /opt/hisemble/examples/run_hisemble_from_h5ad.py \
    --input /data/synthetic_hisemble.h5ad \
    --output /data/synthetic_hisemble_rerun.h5ad \
    --keys view_a,view_b --metrics gaussian,gaussian \
    --k 15 --iterations 20 --resolution 0.5 --seed 42
```

For biological data, substitute your `.h5ad` filename and actual `.obsm` keys. All embeddings must describe the same cells in the same row order. For scATAC inputs, typical keys are `scBasset,SnapATAC2,LSI`, with `gaussian,gaussian,cosine` metrics. Keep each view paired with its intended metric.

NaN/Inf or invalid dimensions stop validation before graph construction. Finite extreme row norms produce a diagnostic warning by default. No cells are automatically deleted or imputed.

## 6. Reproduce supplementary analyses

These commands require external biological inputs. The ZIP excludes raw data and large precomputed embeddings; it includes compact numerical references. The synthetic example above establishes that the deployment works, but is not a reproduction of the biological benchmarks.

Use this aligned AnnData-pickle layout:

```text
ADATA/<dataset>_scBasset.pkl        # obsm['projection']
ADATA/<dataset>_snap.pkl           # obsm['X_spectral']; obs['cell_type']
lsi_results/ADATA/<dataset>_lsi.pkl # obsm['X_lsi']
```

All three files for a dataset must have identical `obs_names` order. Set your local path once:

```bash
HISEMBLE_INPUT_DIR=/absolute/path/to/hisemble_inputs
mkdir -p example_output/analyses
```

### Ten-seed clustering stability

```bash
docker run --rm \
  -v "$HISEMBLE_INPUT_DIR:/data:ro" \
  -v "$PWD/example_output/analyses:/results" \
  hisemble:reproduce \
  python /opt/hisemble/analyses/seed_stability.py \
    --data-root /data --output-dir /results/seed_stability
```

The default is eight full datasets: `Chen-2019-ATAC`, `FCA_fetal_33184180_cerebellum`, `mca_Cerebellum_62216`, `FCA_fetal_33184180_pancreas`, `mca_HeartA_62816`, `mca_Lung1_62216`, `FCA_fetal_33184180_intestine`, and `TME-All`. For a first biological run, append `--datasets mca_Cerebellum_62216`. Every seed repeats the resolution search with fixed embeddings and graph. The full reference includes 80 per-seed records.

For parallel seeds inside Docker, substitute `seed_stability_parallel.py` and append `--seed-workers 2` (or more workers if RAM allows). Each dataset builds its consensus graph once.

### Degraded-LSI analysis

```bash
docker run --rm \
  -v "$HISEMBLE_INPUT_DIR:/data:ro" \
  -v "$PWD/example_output/analyses:/results" \
  hisemble:reproduce \
  python /opt/hisemble/analyses/degraded_view_analysis.py \
    --data-root /data --output-dir /results/degraded_view
```

Defaults reproduce the 66-run design on Mouse Cortex, FCA Cerebellum, and MCA Cerebellum. Use `--datasets mca_Cerebellum_62216` for a smaller first run.

### Components and controlled scaling

The component entry point is `component_ablation.py`, with `--data-root`, `--dataset`, `--variant`, and `--output-dir`. Variants are `full`, `no_diffusion`, and `no_dynamic_sparsification`; default k=15, T=20, seed=42. The full component runner is `run_component_ablation.sh`, configured for complete Mouse Cortex, FCA Intestine, and MCA Cerebellum with a four-hour per-run cap.

No completed reference metrics are supplied for FCA Intestine without dynamic sparsification. The runner enforces a four-hour limit per variant; this condition may require substantially more RAM than a MacBook. Preserve the logs from each new run to distinguish timeout, failure and successful completion.

`run_component_scaling.sh` runs the MCA Cerebellum downsampling analysis. `lsi_neighbor_analysis.py --data-root /data --dataset mca_Cerebellum_62216 --output-dir /results/lsi` evaluates LSI neighborhood complementarity. Run these through the same input/output mounts used above.

### Numerical lineage analysis

Set a directory containing `mca_BoneMarrow_62216_Hisemble_result.h5ad` with `.obsp['connectivities']` and `.obs['cell_type']`. The reported B-cell lineage comprises hematopoietic progenitors, immature B cells, B cells, and activated B cells.
```bash
HISEMBLE_GRAPH_DIR=/absolute/path/to/saved_hisemble_graphs
docker run --rm \
  -v "$HISEMBLE_GRAPH_DIR:/graphs:ro" \
  -v "$PWD/example_output/analyses:/results" \
  hisemble:reproduce \
  python /opt/hisemble/analyses/lineage_trajectory_analysis.py \
    --data-root /graphs --output-dir /results/lineage_trajectory
```

The script computes DPT for the 1,943-cell MCA Bone Marrow B-cell lineage shown in Supplemental Fig. S10, then repeats the analysis using the ten highest-weighted-degree progenitors as roots. Outputs include cell-level `pseudotime.csv`, `metrics.json`, `root_sensitivity.csv`, and `lineage_summary.csv`. Compare the summary and alternative-root results with Supplemental Table S8.

## 7. Locate reference results and analysis workflows

The full-precision reference tables `Supplemental_Table_S1.csv`--`Supplemental_Table_S10.csv` are under `results/` and match the separately submitted CSV files. Cell-level pseudotime for Supplemental Fig. S10 is in `lineage_trajectory/mca_BoneMarrow_62216/pseudotime.csv`. MCA Lung A inputs contain 5,122 aligned cells. Generated resource fields named `peak_memory_mb` use MiB; divide by 1,024 to compare with the GiB values in the supplementary tables.

The archive contains `ANALYSIS_MAP.md`, `INPUTS_AND_ENVIRONMENTS.md`, and configurable preprocessing, encoder and benchmark workflows under `workflows/`. These require separate upstream environments and external inputs. Their complete execution is not covered by the core Docker checks. Pure figure-layout and export programs are excluded.

### Prepare modality-specific workflows

Use `workflows/scatac/` for chromatin-accessibility analyses, `workflows/scrna/`
for the Liver and Pancreas benchmarks, `workflows/scrna/pancreas/` for the
Pancreas comparison, `workflows/spatial/` for spatial RNA, and
`workflows/downstream/` for peak-to-gene analysis.

Set paths in `config.example.json`, save a configured copy, and prepare a new
working directory:

```bash
python prepare_workflows.py --config /absolute/path/config.json \
  --output /absolute/path/configured_workflows
export PYTHONPATH="/absolute/path/configured_workflows${PYTHONPATH:+:$PYTHONPATH}"
```

`WORKFLOW_ROOT` in the configuration should point to this same output directory.
Launch each configured script in the input directory described in
`INPUTS_AND_ENVIRONMENTS.md`. The shared helper is `configured_workflows/utils.py`;
the Pancreas comparison keeps its own helper alongside its two scripts.

## Troubleshooting

- Docker engine unavailable: start Docker Desktop or OrbStack and retry `docker version`.
- Missing embedding key: inspect your `.obsm` keys and supply the exact names.
- NaN/Inf or incompatible cells: correct or explicitly align the upstream inputs before fusion.
- Memory pressure: begin with the synthetic example and one small biological dataset; reduce parallel workers. Large atlas and dense-ablation runs may exceed a MacBook's RAM.
- Run all commands from one checkout; do not mix files from different versions.

## Current convergence scripts (2026-10-11)

The convergence workflow now uses fixed initial W, exactly as the primary analysis. Its original `Relative Change` and pre-update `Delta_Q` columns are retained. The audited full-dataset runner, 24-dataset launcher and original-metric exporter are in [analyses/convergence/](analyses/convergence/README.md). Use `--iterations 50` for the figure range.

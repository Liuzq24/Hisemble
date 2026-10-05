# Hisemble reproduction tutorial

This tutorial runs on a MacBook through Docker Desktop, OrbStack, or another Docker-compatible runtime. It also works on Linux; Windows users may use WSL2. The validated MacBook environment uses Linux/arm64 containers. No GPU is needed for the Hisemble core.

The journal archive is named **Supplemental_Code.zip**. Start with either route below, then follow the same commands. The archive is an independent snapshot; the public GitHub package retains its existing APIs. No figure-drawing scripts are required.

## 1. Obtain the code

### Route A: journal Supplemental Code (recommended for review)

Download the journal's `Supplemental_Code.zip`, or obtain the same file from the revision repository:

```bash
curl -fL -o Supplemental_Code.zip \
  https://raw.githubusercontent.com/Liuzq24/Hisemble/main/supplemental_code/Supplemental_Code.zip
unzip Supplemental_Code.zip
cd Supplemental_Code/revision_core
```

If a directory named `Supplemental_Code` already exists, unpack in a new working directory to avoid mixing versions.

### Route B: GitHub

```bash
git clone https://github.com/Liuzq24/Hisemble.git
cd Hisemble
```

The revision is maintained in this repository. The original project repository is https://github.com/BioX-NKU/Hisemble.

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

Expected outcome: `OK` (nine tests in the journal snapshot; ten in the existing public package, which has one additional API test). These cover aligned input validation, NaN/Inf rejection, mismatched cells, degenerate embeddings, view-order invariance, and the AnnData interface.

A broader executable check additionally runs both examples, compares serial and parallel seed results, verifies finite graph-based DPT, and checks numerical CLI entry points:

```bash
docker run --rm hisemble:reproduce \
  python /opt/hisemble/reproducibility/validate_snapshot.py
```

Expected final line begins `PASS: all packaged tests`. These quick checks use synthetic inputs; they do not require downloaded biological data.

## 4. Run Hisemble and save an AnnData result

```bash
mkdir -p reviewer_output
docker run --rm \
  -v "$PWD/reviewer_output:/output" \
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

The file `reviewer_output/synthetic_hisemble.h5ad` stores the input embeddings in `.obsm`, the consensus graph in `.obsp['connectivities']`, distances in `.obsp['distances']`, Leiden labels in `.obs['hisemble_clusters']`, and neighbor metadata in `.uns['neighbors']`.

## 5. Re-run from a saved input file

This exercises the command used for your own aligned embeddings:

```bash
docker run --rm \
  -v "$PWD/reviewer_output:/data" \
  hisemble:reproduce \
  python /opt/hisemble/examples/run_hisemble_from_h5ad.py \
    --input /data/synthetic_hisemble.h5ad \
    --output /data/synthetic_hisemble_rerun.h5ad \
    --keys view_a,view_b --metrics gaussian,gaussian \
    --k 15 --iterations 20 --resolution 0.5 --seed 42
```

For biological data, substitute your `.h5ad` filename and actual `.obsm` keys. All embeddings must describe the same cells in the same row order. For scATAC inputs, typical keys are `scBasset,SnapATAC2,LSI`, with `gaussian,gaussian,cosine` metrics. Keep each view paired with its intended metric.

NaN/Inf or invalid dimensions stop validation before graph construction. Finite extreme row norms produce a diagnostic warning by default. No cells are automatically deleted or imputed.

## 6. Reproduce reviewer analyses

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
mkdir -p reviewer_output/analyses
```

### Ten-seed clustering stability

```bash
docker run --rm \
  -v "$HISEMBLE_INPUT_DIR:/data:ro" \
  -v "$PWD/reviewer_output/analyses:/results" \
  hisemble:reproduce \
  python /opt/hisemble/supplemental_code/reviewer_analyses/reviewer_seed_stability.py \
    --data-root /data --output-dir /results/seed_stability
```

The default is eight full datasets: `Chen-2019-ATAC`, `FCA_fetal_33184180_cerebellum`, `mca_Cerebellum_62216`, `FCA_fetal_33184180_pancreas`, `mca_HeartA_62816`, `mca_Lung1_62216`, `FCA_fetal_33184180_intestine`, and `TME-All`. For a first biological run, append `--datasets mca_Cerebellum_62216`. Every seed repeats the resolution search with fixed embeddings and graph. The full reference includes 80 per-seed records.

For parallel seeds inside Docker, substitute `reviewer_seed_stability_parallel.py` and append `--seed-workers 2` (or more workers if RAM allows). Each dataset builds its consensus graph once.

### Degraded-LSI analysis

```bash
docker run --rm \
  -v "$HISEMBLE_INPUT_DIR:/data:ro" \
  -v "$PWD/reviewer_output/analyses:/results" \
  hisemble:reproduce \
  python /opt/hisemble/supplemental_code/reviewer_analyses/reviewer_degraded_view.py \
    --data-root /data --output-dir /results/degraded_view
```

Defaults reproduce the 66-run design on Mouse Cortex, FCA Cerebellum, and MCA Cerebellum. Use `--datasets mca_Cerebellum_62216` for a smaller first run.

### Components, scaling and FCA screen

The component entry point is `component_ablation.py`, with `--data-root`, `--dataset`, `--variant`, and `--output-dir`. Variants are `full`, `no_diffusion`, and `no_dynamic_sparsification`; default k=15, T=20, seed=42. The full component runner is `run_component_ablation.sh`, configured for complete Mouse Cortex, FCA Intestine, and MCA Cerebellum with a four-hour per-run cap.

The reported FCA Intestine dense variant was manually stopped after 1,224.44 seconds for computational budget; it did not reach the four-hour cap. Its unfinished accuracy and completed-run memory metrics are unavailable. Re-running that variant may require substantially more RAM than a MacBook.

`run_component_scaling.sh` runs the MCA Cerebellum downsampling analysis. `run_fca_screen.py --data-root /data --output-dir /results/fca_screen` runs all five exploratory FCA comparisons, preserving every outcome. `reviewer_lsi_neighbor_analysis.py --data-root /data --dataset mca_Cerebellum_62216 --output-dir /results/lsi` evaluates LSI neighborhood complementarity. Run these through the same input/output mounts used above.

### Numerical lineage analysis

Set a directory containing six saved `<dataset>_Hisemble_result.h5ad` files with `.obsp['connectivities']` and `.obs['cell_type']`: `Chen-2019-ATAC`, `mca_BoneMarrow_62216`, `TME-All`, `FCA_fetal_33184180_muscle`, `mca_Lung1_62216`, and `mca_Lung2_62216`.

```bash
HISEMBLE_GRAPH_DIR=/absolute/path/to/saved_hisemble_graphs
docker run --rm \
  -v "$HISEMBLE_GRAPH_DIR:/graphs:ro" \
  -v "$PWD/reviewer_output/analyses:/results" \
  hisemble:reproduce \
  python /opt/hisemble/supplemental_code/reviewer_analyses/lineage_trajectory_analysis.py \
    --data-root /graphs --output-dir /results/lineage_trajectory
```

This numerical-only script runs DPT for all six predefined lineages, preserves every candidate result and computes ten-root sensitivity for the representative lineage. It performs no UMAP calculation or figure drawing. Annotated stages assess pseudotime ordering; they do not demonstrate experimentally tracked ancestry.

## 7. Locate reference results and historical workflows

Reference tables are under `supplemental_code/reviewer_results/` in both routes. Compare numerical CSV/JSON records, including failed or stopped conditions, with the supplied references. MCA Lung A inputs contain 5,122 aligned cells; the manuscript's older table lists 5,112, a discrepancy recorded in the reference notes.

The journal ZIP additionally contains `ANALYSIS_MAP.md`, `INPUTS_AND_ENVIRONMENTS.md`, and configurable historical preprocessing/encoder/benchmark workflows under `manuscript_templates/`. These require separate upstream environments and external inputs. Their complete historical execution is not covered by the core Docker checks. Pure figure-layout and export programs are excluded.

## Troubleshooting

- Docker engine unavailable: start Docker Desktop or OrbStack and retry `docker version`.
- Missing embedding key: inspect your `.obsm` keys and supply the exact names.
- NaN/Inf or incompatible cells: correct or explicitly align the upstream inputs before fusion.
- Memory pressure: begin with the synthetic example and one small biological dataset; reduce parallel workers. Large atlas and dense-ablation runs may exceed a MacBook's RAM.
- Do not mix old extracted folders with the current `Supplemental_Code.zip`.

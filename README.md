# Hisemble

A representation-fusion framework for ensemble clustering of single-cell data. Hisemble combines aligned, precomputed embeddings and returns a weighted cell–cell consensus graph with Leiden labels directly in AnnData.

## Getting started with the original tutorials

Hisemble operates on precomputed embeddings stored in an AnnData object's `.obsm`. Obtain these representations with upstream tools such as scBasset, SnapATAC2, LSI or scVI before applying Hisemble to your own data. Input views must contain the same cells in the same row order; their feature dimensions may differ.

The original notebooks and demonstration data remain available:

| Tutorial | Demonstration input | What it shows |
| --- | --- | --- |
| [Single-cell quick start](tutorials/single_cell_quick_start.ipynb) | [mouse_brain.h5ad](demo_data/mouse_brain.h5ad) | Fuse `X_lsi`, `X_snap` and `X_scBasset`, save labels as `Hisemble_clusters`, and visualize the consensus graph with Scanpy UMAP. |
| [Spatial quick start](tutorials/spatial_quick_start.ipynb) | [E11_0.h5ad](demo_data/E11_0.h5ad) | Fuse `spatial` coordinates and `X_lsi`, save labels as `Hisemble_domains`, and display the spatial domains. |

To run the notebooks locally, clone the repository and install the package and notebook dependencies:

```bash
git clone https://github.com/Liuzq24/Hisemble.git
cd Hisemble
conda env create -f reproducibility/environment-core.yml
conda activate hisemble-core
python -m pip install -r reproducibility/requirements-trajectory.txt
python -m pip install -e .
python -m pip install notebook ipykernel
cd tutorials
python -m notebook
```

Open either notebook and run its cells in order. Run from the `tutorials/` directory so that the existing `../demo_data/` paths resolve correctly. For your own AnnData input, replace the data path and `input_keys` with the keys in your `.obsm`, and choose the corresponding metrics. The examples below describe the same public API.

## Repository guide

- [tutorials/](tutorials/): the original single-cell and spatial notebooks.
- [demo_data/](demo_data/): the AnnData inputs used by those notebooks.
- [hisemble/](hisemble/): the installable package, including graph construction, fusion, clustering and input validation.
- [requirements.txt](requirements.txt) and [setup.py](setup.py): the original dependency list and package installation configuration. Use the pinned environment above for reproducible dependency versions.
- [Reproduction_Tutorial.md](Reproduction_Tutorial.md): the additional Docker and numerical reproduction walkthrough.
- [examples/](examples/), [reproducibility/](reproducibility/) and [tests/](tests/): executable examples, environment specifications and verification checks.

## Reproduce on macOS or Linux

Start with [Reproduction_Tutorial.md](Reproduction_Tutorial.md). The tutorial covers Docker setup, input checks, a complete synthetic example, saved AnnData inputs, and the reviewer analyses.

```bash
git clone https://github.com/Liuzq24/Hisemble.git
cd Hisemble
docker build -f reproducibility/Dockerfile -t hisemble:reproduce .
mkdir -p reviewer_output
docker run --rm -v "$PWD/reviewer_output:/output" hisemble:reproduce \
  python /opt/hisemble/examples/quickstart_synthetic.py --output-dir /output
```

The example needs no biological download or GPU. It writes a 180-cell AnnData result with four clusters. Docker Desktop or OrbStack must be running on a MacBook.

## Direct AnnData use

Store aligned cell embeddings under user-chosen `.obsm` keys, then run:

```python
from hisemble import run_hisemble

run_hisemble(
    adata,
    input_keys=["scBasset", "SnapATAC2", "LSI"],
    metrics=["gaussian", "gaussian", "cosine"],
    k_neighbors=15,
    n_iterations=20,
    resolution=1.0,
    random_seed=42,
)
```

The consensus graph is stored in `.obsp['connectivities']`, distances in `.obsp['distances']`, labels in `.obs['hisemble_clusters']`, and graph metadata in `.uns['neighbors']`. These can be used by Scanpy graph-based workflows. Hisemble produces a consensus graph; upstream embeddings remain in `.obsm`.

## Input validation

All views must contain the same cells in the same row order, finite numeric values and non-degenerate cell-to-cell variation. Views may have different feature dimensions. `run_hisemble` invokes `validate_embeddings` before affinity construction. NaN/Inf triggers an informative error. Finite extreme row norms produce a diagnostic warning by default; explicit per-view identifiers and stricter outlier handling are available through `validate_embeddings`. Cells are not automatically imputed or deleted.

## Numerical revision analyses

[analyses/](analyses/) contains component ablation/scaling, LSI neighborhood complementarity, eight-dataset ten-seed stability, three-dataset degraded-view robustness and numerical DPT lineage analysis. [results/](results/) contains the current Supplemental Tables S4–S9 and cell-level pseudotime. [workflows/](workflows/), [ANALYSIS_MAP.md](ANALYSIS_MAP.md) and [INPUTS_AND_ENVIRONMENTS.md](INPUTS_AND_ENVIRONMENTS.md) document preprocessing, representations, benchmarks and their separate input/environment requirements. Configure external paths with `prepare_workflows.py` and `config.example.json`.

The journal archive is [supplemental_code/Supplemental_Code.zip](supplemental_code/Supplemental_Code.zip). The older `supplemental_code/reviewer_analyses` and `reviewer_results` directories are historical snapshots; use the root-level analyses and results for the current revision.

## Installation without Docker

A pinned Conda specification is provided in [reproducibility/environment-core.yml](reproducibility/environment-core.yml). Then install this package with `pip install -e .`. Numerical lineage analysis additionally needs [requirements-trajectory.txt](reproducibility/requirements-trajectory.txt). Upstream encoders have separate dependencies; they are not all trained by the core environment.

## Verification

```bash
docker run --rm hisemble:reproduce \
  python /opt/hisemble/reproducibility/verify_installation.py
```

This runs the packaged tests, both examples, serial/parallel seed comparison, a graph-based DPT check and CLI checks. Details are recorded in [VALIDATION.md](reproducibility/VALIDATION.md).

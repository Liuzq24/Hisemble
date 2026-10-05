# Hisemble

A representation-fusion framework for ensemble clustering of single-cell data. Hisemble combines aligned, precomputed embeddings and returns a weighted cell–cell consensus graph with Leiden labels directly in AnnData.

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

## Journal Supplemental Code

The current journal archive is [Supplemental_Code.zip](supplemental_code/Supplemental_Code.zip), with [SHA-256 checksum](supplemental_code/Supplemental_Code.zip.sha256). It includes the numerical analysis code, compact reference results, environment files, tutorial and configurable historical workflows. Figure-drawing scripts, raw sequencing data and large input embeddings are excluded.

The archive is independently versioned; its core computational functions match this repository. The existing public API is retained. The original project repository is [BioX-NKU/Hisemble](https://github.com/BioX-NKU/Hisemble).

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

[supplemental_code/reviewer_analyses](supplemental_code/reviewer_analyses) contains full-data component ablation/scaling, LSI neighborhood complementarity, eight-dataset ten-seed stability, three-dataset degraded-view robustness, the complete FCA screen, and numerical-only DPT lineage analysis. [Reference tables](supplemental_code/reviewer_results) retain the reported outcomes and explain stopped conditions. No figure-generation entry points are included.

## Installation without Docker

A pinned Conda specification is provided in [reproducibility/environment-core.yml](reproducibility/environment-core.yml). Then install this package with `pip install -e .`. Numerical lineage analysis additionally needs [requirements-trajectory.txt](reproducibility/requirements-trajectory.txt). Upstream encoders have separate dependencies; they are not all trained by the core environment.

## Verification

```bash
docker run --rm hisemble:reproduce \
  python /opt/hisemble/reproducibility/validate_snapshot.py
```

This runs the packaged tests, both examples, serial/parallel seed comparison, a graph-based DPT check and CLI checks. Details are recorded in [VALIDATION.md](reproducibility/VALIDATION.md).

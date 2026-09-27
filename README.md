# Hisemble 🧬

> **A scalable representation fusion framework for robust ensemble clustering of single-cell data.**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## Prerequisites

Hisemble is designed as a lightweight, downstream representation fusion aggregator. **It operates on pre-computed embeddings.**

Before running Hisemble, you should have already extracted low-dimensional embeddings using your preferred upstream tools (e.g., scBasset, SnapATAC2, LSI, scVI) and stored them in the `.obsm` attribute of your Scanpy `AnnData` object. We do not restrict or include these heavy upstream models in our dependencies to maintain maximum flexibility and speed.

---

## Installation

**Step 1: Clone the repository**
```bash
git clone https://github.com/BioX-NKU/Hisemble.git
cd Hisemble
```

**Step 2: Create and activate a Conda environment**
```bash
conda create -n Hisemble python=3.8.18 -y
conda activate Hisemble
```

**Step 3: Install dependencies**
```bash
pip install -r requirements.txt
```

**Step 4: Install Hisemble (Editable mode)**
```bash
pip install -e .
```

## Input requirements and validation

Hisemble constructs an affinity graph independently for every view, so input
embeddings may have different dimensions, coordinate systems, numerical scales,
and marginal distributions. They must nevertheless describe the same cells in
the same row order, contain finite numeric values, retain non-degenerate
cell-to-cell variation, and use a similarity metric that is meaningful for the
embedding geometry.

`run_hisemble` performs these checks before graph construction. The public
`validate_embeddings` utility additionally supports explicit cell identifiers
and configurable robust row-norm outlier warnings. It never silently imputes
values or removes cells, because independent row removal would invalidate
cross-view alignment.

## Reproducible core environment

The quick-start container covers Hisemble affinity construction, fusion,
clustering, validation, and export from precomputed embeddings. It intentionally
does not bundle the separate training environments for upstream encoders.

```bash
docker build -f reproducibility/Dockerfile -t hisemble-core:revision .
docker run --rm hisemble-core:revision \
  python /opt/hisemble/tests/test_hisemble_io.py
```

A pinned Conda alternative is provided in
`reproducibility/environment-core.yml`. See
`reproducibility/README_container.md` for the complete commands and validation
record.

## AnnData and Seurat interoperability

Hisemble returns a weighted cell-cell consensus graph rather than a new
Euclidean joint embedding. `store_consensus_in_anndata` stores this sparse graph
in `AnnData.obsp` with optional clusters in `AnnData.obs`.
`export_consensus_for_seurat` writes a sparse Matrix Market graph, ordered cell
identifiers, cluster metadata, and an R import example for Seurat workflows.

## Supplemental analyses

Custom scripts added for the revision, including component ablations,
downsampling, LSI complementarity, random-seed stability, degraded-view
robustness, and response-figure generation, are organized under
`supplemental_code/reviewer_analyses/`.

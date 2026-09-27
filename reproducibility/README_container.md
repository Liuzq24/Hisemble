# Hisemble core reproducibility environment

This environment reproduces the **Hisemble fusion and clustering core** from
precomputed cell embeddings. It intentionally does not bundle training
environments for scBasset, SnapATAC2, scVI, or other upstream encoders. Those
tools have separate GPU and platform requirements, whereas Hisemble only needs
their cell-by-feature embeddings.

## Docker

Build from the root of the repository:

```bash
docker build -f reproducibility/Dockerfile -t hisemble-core:revision .
```

Open a shell with the current directory mounted at `/work`:

```bash
docker run --rm -it -v "$PWD:/work" hisemble-core:revision
```

The numerical-library thread count defaults to one for reproducible resource
benchmarks. Users may override `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`,
`MKL_NUM_THREADS`, and `NUMEXPR_NUM_THREADS` when throughput is more important
than strict benchmark comparability.

Run the input-validation unit tests inside the image:

```bash
docker run --rm hisemble-core:revision \
  python /opt/hisemble/tests/test_hisemble_io.py
```

Run the core view-order invariance tests:

```bash
docker run --rm hisemble-core:revision \
  python /opt/hisemble/tests/test_hisemble_core.py
```

Run the package API and AnnData/Seurat interoperability tests:

```bash
docker run --rm hisemble-core:revision \
  python /opt/hisemble/tests/test_hisemble_api.py
```

The revision-stage build and test record is documented in `VALIDATION.md`.

## Conda

```bash
conda env create -f reproducibility/environment-core.yml
conda activate hisemble-core
```

## Input and output safeguards

`hisemble/io.py` provides:

- `validate_embeddings` for shape, finite-value, cell-order, degeneracy, and
  robust row-norm outlier checks;
- `store_consensus_in_anndata` for Scanpy/AnnData workflows; and
- `export_consensus_for_seurat` for sparse Matrix Market export with ordered
  cell identifiers and cluster metadata.

Hisemble produces a weighted cell-cell consensus graph. It does not silently
remove cells, impute upstream embeddings, or claim to produce a new Euclidean
joint embedding.

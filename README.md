# Hisemble

Hisemble fuses complementary, precomputed cell embeddings through training-free sparse cross-diffusion for scalable ensemble clustering of single-cell chromatin accessibility data.

Provide two or more embeddings for the same cells in identical order. Their feature dimensions may differ. Hisemble returns a consensus graph and cluster assignments in an AnnData object; no GPU is required.

## Installation

Use Python 3.10. Clone the repository first:

```bash
git clone https://github.com/Liuzq24/Hisemble.git
cd Hisemble
```

### pip

Create a virtual environment and install Hisemble with the pinned dependencies:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install -r reproducibility/requirements-core.txt -r reproducibility/requirements-trajectory.txt
python -m pip install --no-deps -e .
```

### Conda

Alternatively, create the supplied Conda environment:

```bash
conda env create -f reproducibility/environment-core.yml
conda activate hisemble-core
python -m pip install -r reproducibility/requirements-trajectory.txt
python -m pip install --no-deps -e .
```

## Quick start

Run the synthetic example from the repository root:

```bash
python examples/quickstart_synthetic.py --output-dir example_output
```

This example uses 180 cells, requires no biological data download, and writes `example_output/synthetic_hisemble.h5ad` with the consensus graph and cluster labels.

## Use your own AnnData object

Store the embeddings in `adata.obsm` and replace the example keys below with their actual names:

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

Use Gaussian affinity for continuous latent embeddings and cosine similarity for LSI profiles. Inputs must contain finite values and non-degenerate variation. The consensus graph is stored in `adata.obsp["connectivities"]`, distances in `adata.obsp["distances"]`, cluster labels in `adata.obs["hisemble_clusters"]`, and neighbor metadata in `adata.uns["neighbors"]`.

## Reproduce the paper analyses

[Reproduction_Tutorial.md](Reproduction_Tutorial.md) describes Docker setup, tests, analysis commands, and configuration of the benchmark workflows. External data and upstream encoder requirements are detailed in [INPUTS_AND_ENVIRONMENTS.md](INPUTS_AND_ENVIRONMENTS.md). See [ANALYSIS_MAP.md](ANALYSIS_MAP.md) for the correspondence between analyses, scripts, and reference results.

## License

Hisemble is distributed under the [MIT license](LICENSE).

# Hisemble

Training-free fusion of precomputed cell embeddings for scalable ensemble clustering of single-cell chromatin accessibility data.

Start with [Reproduction_Tutorial.md](Reproduction_Tutorial.md) to build the pinned environment, run the tests and synthetic example, and reproduce the analyses from external inputs.

| Location | Contents |
|---|---|
| `hisemble/` | Fusion and AnnData API |
| `tests/` and `examples/` | Tests and runnable examples |
| `reproducibility/` | Docker, pip and Conda specifications; executable verification |
| `analyses/` | Supplementary analysis scripts |
| `results/` | Supplemental Tables S1--S10 and Fig. S10 cell-level pseudotime |
| `workflows/` | Preprocessing, encoder, benchmark and downstream workflows grouped by data modality |
| `ANALYSIS_MAP.md` | Analysis-to-script mapping |
| `INPUTS_AND_ENVIRONMENTS.md` | Required data and environment boundaries |
| `WORKFLOW_MANIFEST.csv` and `SHA256SUMS` | Script inventory and file checksums |

The core accepts precomputed embeddings. Raw data and large embeddings are external inputs; upstream encoders require separate environments. Figures are supplied with the manuscript.

To prepare workflows with your input paths, replace the paths in `config.example.json`, then run:

```bash
python prepare_workflows.py --config /absolute/path/config.json --output /absolute/path/new-workdir
```

Project repository: https://github.com/Liuzq24/Hisemble

## Run a first example

From this `Supplemental_Code` directory:

```bash
docker build -f reproducibility/Dockerfile -t hisemble:reproduce .
mkdir -p example_output
docker run --rm -v "$PWD/example_output:/output" hisemble:reproduce \
  python /opt/hisemble/examples/quickstart_synthetic.py --output-dir /output
```

The synthetic example uses 180 cells and needs no biological download or GPU.

## Use the AnnData interface

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

The consensus graph is stored in `.obsp['connectivities']`, distances in
`.obsp['distances']`, labels in `.obs['hisemble_clusters']`, and graph metadata in
`.uns['neighbors']`. Input views must describe the same cells in the same row
order, contain finite values and exhibit non-degenerate variation.

For installation without Docker, use `reproducibility/environment-core.yml` and
then `pip install -e .`. DPT analysis also uses
`reproducibility/requirements-trajectory.txt`. See `INPUTS_AND_ENVIRONMENTS.md`
for upstream encoder and data requirements.

## Current convergence scripts (2026-10-11)

The convergence workflow now uses fixed initial W, exactly as the primary analysis. Its original `Relative Change` and pre-update `Delta_Q` columns are retained. The audited full-dataset runner, 24-dataset launcher and original-metric exporter are in [analyses/convergence/](analyses/convergence/README.md). Use `--iterations 50` for the figure range.

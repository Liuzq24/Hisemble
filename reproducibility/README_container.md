# Hisemble reproducibility environment

Start with the root-level `Reproduction_Tutorial.md`. Build from the repository root:

```bash
docker build -f reproducibility/Dockerfile -t hisemble:reproduce .
docker run --rm hisemble:reproduce python /opt/hisemble/reproducibility/verify_installation.py
```

The image runs fusion, Leiden clustering and graph-based DPT from precomputed embeddings. Upstream encoders have separate input and environment requirements described in `INPUTS_AND_ENVIRONMENTS.md`. Numerical library thread counts default to one.

The current public interface exposes `run_hisemble`, `hisemble_fusion` and `validate_embeddings`. `run_hisemble` writes connectivity and distance graphs, categorical clustering labels and neighbors metadata into AnnData. The older optional AnnData storage and Seurat export helper functions are not included in the current journal archive or package.

The archive has nine tests; the GitHub checkout also tests all six view permutations directly in the production fusion core. See `VALIDATION.md` for actual local checks and `requirements-resolved-linux-arm64.txt` for the full dependency versions resolved during that run. This resolved list records the tested Linux/arm64 environment; it is not a claim that upstream workflows or other platforms were tested.

For Conda installation, use `environment-core.yml`, install `requirements-trajectory.txt` for DPT, then run `pip install -e .` from the repository root.

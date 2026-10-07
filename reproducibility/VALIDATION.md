# Local validation of the current revision

Validation: 2026-10-06, America/Toronto (2026-10-07 UTC), on an Apple Silicon Mac using OrbStack Docker, Linux/arm64 and Python 3.10.13. Both the journal code and the updated GitHub checkout were built from their supplied Dockerfiles. The full resolved Python environment is recorded in `requirements-resolved-linux-arm64.txt`.

The submitted archive SHA-256 is `a486c2af63620685a6579ff5bae6e5eb59c3e49887b1edb0e0c604a674d348db`. Its extracted source files match the ZIP; all its recorded file checksums passed. The current four computational modules (`core.py`, `api.py`, `graph.py`, `cluster.py`) are unchanged from the prior repository commit `5df28c7`.

Passed checks:

1. All nine journal-archive tests and all eleven GitHub tests. The two additional GitHub tests exercise all six permutations directly in the production fusion core, both with diffusion and with zero iterations. The separate production check had maximum error 1.1102230246251565e-16 against tolerance 1e-12.
2. Synthetic example: 180 cells, four clusters, 180 × 180 connectivity graph, 7,372 nonzeros. Reloading the same embeddings and parameters produced exactly equal graphs and clustering labels, with unchanged cell identifiers. These numerical assertions are now part of the GitHub verification script.
3. Exact equality of serial and two-worker parallel seed records on a 90-cell synthetic fixture, seeds 0, 42 and 101.
4. Finite Scanpy diffusion pseudotime on a connected sparse graph; six current numerical-analysis CLI entry points accepted `--help`.
5. Normal package installation succeeded and `hisemble` imported from `site-packages` with the source `PYTHONPATH` removed and Python isolated mode enabled.
6. Every code cell in both original notebooks executed, including plotting (headless Matplotlib), using the downloaded GitHub demonstration inputs: spatial data, 1,258 cells, seven clusters, 98,328 graph nonzeros; single-cell data, 13,671 cells, 35 clusters, 1,387,489 graph nonzeros. These are execution results, not a claim of agreement with independently supplied biological ground truth.
7. Workflow preparation generated 26 configured files; current Python sources and notebook JSON parsed successfully. Preparation does not run upstream analyses.

Repeat the package checks with:

```bash
docker build -f reproducibility/Dockerfile -t hisemble:reproduce .
docker run --rm hisemble:reproduce python /opt/hisemble/reproducibility/verify_installation.py
```

The current root-level analyses/results and bundled journal ZIP supersede the older historical `supplemental_code/reviewer_analyses` and `reviewer_results` snapshots. The original notebooks and demonstration data remain available. Current package exports follow the journal archive; the older optional storage/Seurat helper exports were removed in that archive.

## Scope

Full manuscript benchmarks, all Supplemental Tables S4–S9 and the MCA Bone Marrow lineage result were not regenerated in this local run: their external precomputed representations and saved consensus graph were not available in the located revision inputs. Upstream encoder training, GPU workflows, native macOS dependency installation and other architectures were not tested. Archived reference tables are not substitutes for newly executed experiments. Historical validation records are not evidence for this run.

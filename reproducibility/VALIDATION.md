# Validation of Supplemental Code 20261004

Validation date: 2026-10-04 (America/Toronto). The submitted Dockerfile was built locally on Linux/arm64 with tag `hisemble:reproduce`. It includes the unchanged core dependency pins plus the separate trajectory dependency list. No image was uploaded to a registry.

Passed checks:

1. All nine original submission tests (6 input validation, 2 view-order invariance, 1 AnnData API).
2. Synthetic example: 180 cells, four clusters, graph shape 180×180, 7,372 nonzeros.
3. Reloading its embeddings with `run_hisemble_from_h5ad.py`.
4. Exact equality of serial and two-worker parallel seed metrics on a complete 90-cell synthetic fixture (seeds 0, 42, 101).
5. Finite Scanpy diffusion pseudotime on a connected sparse graph.
6. Seven numerical analysis CLI entry points accept `--help`.
7. All ten tests in the existing GitHub checkout passed when read-only mounted into the new dependency environment. The public package itself was imported for this check; no repository files were modified.
8. The FCA screening summary regenerated exactly from the archived numerical records.
9. All packaged Python sources parsed successfully; the computational core matches the prior submission snapshot, and the main four computational modules match the existing GitHub checkout.

The numerical-only DPT script was additionally run against the existing server MCA Bone Marrow graph: 1,943 lineage cells, Spearman rho 0.7644229782983922, ordinal concordance 0.9823018832373037; ten-root median rho 0.7628491749195323 and concordance 0.9814750537996966. These match the reported results after removing all UMAP and figure generation.

To repeat the image checks:

```bash
docker run --rm hisemble:reproduce python /opt/hisemble/reproducibility/validate_snapshot.py
```

Full manuscript-data analyses are not rerun by this quick check; they require the external precomputed embeddings or saved consensus graphs listed in the reviewer tutorial. Historical upstream encoder environments remain separate.

The current GitHub-route reproduction image was rebuilt on the MacBook from the updated repository, and the full validate_snapshot.py check passed with all ten existing public tests. The renamed tutorial is Reproduction_Tutorial.md.

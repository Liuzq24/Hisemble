# Reviewer-analysis results

This directory contains the compact source tables, machine-readable summaries,
and response figures produced by the revision analyses.

- `lsi_complementarity/`: MCA cerebellum full-versus-no-LSI metrics,
  LSI-specific neighborhood purity summaries, cell-type summaries, and UMAP.
- `component_ablation/`: four-dataset component ablation and controlled
  downsampling summaries with the corresponding response figures.
- `robustness/`: ten-seed clustering stability, pairwise partition agreement,
  and the 88-run degraded-LSI analysis with both response figures.

Large input embeddings and private server paths are not included. The scripts
under `../reviewer_analyses/` accept a portable data root and recreate these
outputs from appropriately named AnnData pickle files.

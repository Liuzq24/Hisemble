# Supplemental revision analyses

This independently versioned journal snapshot includes numerical reviewer analyses and compact results. It does not require a matching public GitHub release. The existing Hisemble package interfaces are unchanged.

`reviewer_analyses/` contains component ablation/downsampling, LSI complementarity, eight-dataset Leiden seed stability, three-dataset degraded-view robustness, and numerical-only DPT lineage analysis. No standalone plotting or panel-export scripts are included.

`reviewer_results/` contains reference tables. Raw inputs and precomputed embeddings are external; see the root reviewer tutorial for exact names and commands. The original seed analysis still calculates ancillary pairwise agreement, but the reported seed figure and table use only per-seed ARI/NMI.

# Computational analysis entry points

| Reported analysis | Entry point(s) | Principal output |
|---|---|---|
| scATAC Hisemble benchmark | `workflows/scatac/hisemble/run_Hisemble.py` | clustered AnnData and benchmark metrics |
| LSI ablation | `workflows/scatac/hisemble/Hisemble_ablation_lsi.py` | full/no-LSI metrics |
| Neighborhood sensitivity | `workflows/scatac/hisemble/Hisemble_K_Sensitive.py` | metrics across k |
| Convergence | `workflows/scatac/hisemble/Hisemble_convergence.py` | iteration-wise convergence records |
| SC3 and CHAI baselines | `workflows/scatac/baselines/` | baseline metrics |
| Scaling/downsampling | `workflows/scatac/downsampling/` | runtime, memory, and clustering metrics |
| scBasset, SnapATAC2, and LSI representations | `workflows/scatac/representations/` | precomputed cell embeddings |
| scATAC preprocessing | `workflows/scatac/preprocess/` | processed AnnData inputs |
| scRNA benchmark (Liver and Pancreas; Fig. 5A--B) | `workflows/scrna/` (Liver and Pancreas) and `workflows/scrna/pancreas/` (Pancreas comparison) | clustering metrics and labels |
| Spatial RNA benchmark (Fig. 5C) | `workflows/spatial/run_GraphST.py` | clustering metrics and labels |
| Peak-to-gene downstream analysis | `workflows/downstream/peakbed2gene/peakbed2gene.py` | peak-to-gene result table |
| LSI complementarity | `workflows/scatac/hisemble/Hisemble_ablation_lsi.py` and `analyses/lsi_neighbor_analysis.py` | full/no-LSI metrics and neighborhood summaries |
| Component ablation/scaling | `analyses/component_ablation.py`, `analyses/run_component_ablation.sh`, `analyses/run_component_scaling.sh` | CSV/JSON measurements |
| Seed stability | `analyses/seed_stability.py` | 80 per-seed ARI/NMI records on eight datasets |
| Degraded-view robustness | `analyses/degraded_view_analysis.py` | 66-run result table |
| Graph-based lineage analysis | `analyses/lineage_trajectory_analysis.py` | MCA Bone Marrow B-cell DPT, per-cell pseudotime, ten-root sensitivity |

Pure plotting and panel-layout scripts are not included. Figures and legends are
provided separately in the manuscript and Supplemental Material.


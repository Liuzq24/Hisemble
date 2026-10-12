# Numerical reference results

The CSV files match the full-precision tables submitted with the paper.

| File | Reported analysis |
|---|---|
| `Supplemental_Table_S1.csv` | Original Supplemental Table S1 |
| `Supplemental_Table_S2.csv` | Original Supplemental Table S2 |
| `Supplemental_Table_S3.csv` | Original Supplemental Table S3 |
| `Supplemental_Table_S4.csv` | LSI complementarity and clustering ablation (Fig. S5) |
| `Supplemental_Table_S5.csv` | Component ablation and all 18 scaling runs (Figs. S6--S7) |
| `Supplemental_Table_S6.csv` | 80 seed-wise ARI/NMI records, including resolution and cluster count (Fig. S8) |
| `Supplemental_Table_S7.csv` | 66 degraded-view runs (Fig. S9) |
| `Supplemental_Table_S8.csv` | MCA Bone Marrow B-cell DPT, stage summaries, and ten-root sensitivity (Fig. S10) |
| `Supplemental_Table_S9.csv` | Atlas-scale runtime and peak resident memory (Fig. 3B) |
| `Supplemental_Table_S10.csv` | Versioned TF-motif identifiers and source species |
| `lineage_trajectory/mca_BoneMarrow_62216/pseudotime.csv` | Cell-level DPT for the 1,943 lineage cells (Fig. S10) |

The FCA Intestine condition without dynamic sparsification timed out after exceeding four hours; its unavailable measurements remain blank in Table S5.

Table memory values are in GiB. Newly generated fields named `peak_memory_mb` use MiB and are divided by 1,024 for comparison.

See the root `Reproduction_Tutorial.md` for inputs and execution commands.

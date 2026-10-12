# Inputs and environments

## Tested Hisemble core

Use `Reproduction_Tutorial.md` and the pinned Docker/Conda specifications in
`reproducibility/`. The core accepts precomputed embeddings. Its
recorded tests do not validate all analysis scripts or upstream encoders.

## scATAC preprocessing and benchmarks

`DATA_ROOT` follows the original layout:

```
ADATA/<dataset>_scBasset.pkl
ADATA/<dataset>_snap.pkl
lsi_results/ADATA/<dataset>_lsi.pkl
0.01_h5ad/<dataset>_processed.h5ad
```

The pickles hold AnnData objects with `obsm['projection']`, `obsm['X_spectral']`,
and `obsm['X_lsi']`, respectively. See each script for the expected annotation field
and dataset list. Some benchmark scripts instead expect LSI under
`ADATA/`; provide the appropriate input location for the selected variant. Do not
assume that all variants use interchangeable filenames. `RESULTS_ROOT` holds
`Hisemble_final/`, baseline-specific folders, and `downsampling/` outputs, which downstream
scripts also read. Preprocessing scripts take h5ad/FASTA manifests; genome assemblies
must match each dataset. Reference FASTA files are supplied under `GENOME_ROOT`.

These scripts import additional packages such as Scanpy, seaborn, TensorFlow,
SnapATAC2 and scBasset. Baseline dependencies are recorded in `DEPENDENCY_IMPORTS.csv`.
Upstream encoder versions and hardware-specific environments are not pinned in this archive.
Do not install every imported package into the tested core environment as one stack.

## scRNA

The scripts in `workflows/scrna/` read `Tabula_Sapiens/TS_<tissue>_30k_processed.h5ad` relative
to its launch directory and writes `results/Hisemble_RNA/`. It accepts `--tissues`;
its defaults are Liver and Pancreas. For the separate Pancreas comparison, run the scripts from
`workflows/scrna/pancreas/` in the configured working tree.
`run_Hisemble_pancreas.py` reads `../data/scRNA_pancreas.h5ad` and writes
`../data/bestpancreas_hisemble_results.h5ad`; `run_SC3.py` reads that saved file.
Keep `utils.py` alongside these scripts. The upstream preprocessing/scVI
environment remains to be confirmed.

## Spatial omics

`SPATIAL_ROOT` contains the processed MISAR-seq sections, including
`E11_0-S1_expr_preprocessed.h5ad`, `E13_5-S1_expr_preprocessed.h5ad`,
`E15_5-S1_expr_preprocessed.h5ad`, and `E18_5-S1_expr_preprocessed.h5ad` for the
GraphST RNA workflow. Inputs need the spatial coordinates and annotation fields
accessed in that script. GraphST/PyTorch needs a separate upstream environment. Its accelerator use differs from the
CPU-only Hisemble core benchmark. Exact versions for the original upstream runs are not specified in this archive.

## Downstream analyses

`workflows/downstream/peakbed2gene/peakbed2gene.py` maps BED intervals to genes
using `ANNOTATION_ROOT/gencode.vM25.annotation.gtf`. Run it from the directory
containing the BED input specified in the script, such as `cluster_0_GREAT.bed`.
The required GTF assembly and BED coordinates must agree.

## Presentation files

Pure plotting, panel-layout, typography, and figure-export code is not part of the
Supplemental Code archive. Numerical result tables produced by the analyses are
reported in the manuscript and Supplemental Tables; full-precision reference CSVs are in `results/`.

## Numerical lineage analysis

The Docker image installs `requirements-trajectory.txt`, including Scanpy 1.9.8.
Supply `mca_BoneMarrow_62216_Hisemble_result.h5ad` containing the saved Hisemble
consensus graph in `.obsp['connectivities']` and annotations in `.obs['cell_type']`.
The script reproduces the B-cell lineage DPT and ten-root sensitivity analysis
reported in Supplemental Fig. S10 and Supplemental Table S8.

## Current convergence scripts (2026-10-11)

The convergence workflow now uses fixed initial W, exactly as the primary analysis. Its original `Relative Change` and pre-update `Delta_Q` columns are retained. The audited full-dataset runner, 24-dataset launcher and original-metric exporter are in [analyses/convergence/](analyses/convergence/README.md). Use `--iterations 50` for the figure range.

# Supplemental Code

This directory contains custom analysis code added for the Hisemble revision.
The journal-hosted Supplemental Code archive should be generated from the same
tag or commit as the public `BioX-NKU/Hisemble` release.

`reviewer_analyses/` contains the component ablation, controlled downsampling,
LSI complementarity, random-seed stability, degraded-view robustness, and
response-figure scripts. Data files are not bundled. Each executable accepts a
data root and output directory through command-line arguments or the documented
runner environment variables.

Before submission, add the remaining manuscript figure-generation and
benchmarking scripts required to reproduce every reported panel. Do not include
private data, server credentials, absolute user paths, caches, or intermediate
binary files.

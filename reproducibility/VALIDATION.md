# Local verification: 2026-10-11

## Algorithm and convergence workflows

The live convergence workflows use fixed initial affinity matrices W, followed by the primary analysis's sparsification and normalization. The original P metric and pre-update Q timing are preserved. The primary numerical core is unchanged.

Eighteen comparisons of the two live workflows against the primary analysis passed, covering mixed floating-point dtypes, tied rows and sparse zero rows at 20, 21 and 50 iterations. Final and T20 checkpoint matrices have zero differing stored entries, and the original P/Q metrics agree with the audited diagnostic runner. The original-metric exporter was also checked against existing logs from 24 datasets: 3,600 T1-T50 records matched the approved figure data. Convergence numerical results and server logs are not included in this code archive.

## English text and method names

All Chinese comments, docstrings, run messages and command-line help were translated into English (750 occurrences across 22 Python files). Historical SNF names referring to Hisemble were updated in function calls, variables, output fields, filenames and result directories. The CHAI-SNF baseline retains its actual method name. Computational expressions and numerical constants were structurally checked before and after editing. Nine direct before/after fusion-helper comparisons produced identical matrices and convergence history.

New Hisemble output names include `Hisemble_final/`, `hisemble_fixed_params_results.csv`, `hisemble_sensitivity_k_neighbors.csv`, `Hisemble_Time_s` and `Hisemble_Peak_Mem_MB`.

## Package checks

The current sources passed 9 core/API tests, 4 convergence-equivalence tests, synthetic graph and cluster assertions, an exact H5AD reload comparison, serial/parallel seed equality, finite graph-based DPT, and six analysis CLI checks in the reproduction container. Supplemental Tables S1-S10 match the submitted CSV files byte for byte.

These checks use synthetic fixtures and previously generated results; they do not rerun the full biological benchmarks.

Run `python analyses/convergence/test_runner_equivalence.py` for the convergence checks. Run `python reproducibility/verify_installation.py` in the documented container environment for the complete package checks.

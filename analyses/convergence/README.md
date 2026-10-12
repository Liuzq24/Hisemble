# Current fixed-W convergence analysis

The numerical runner uses the fixed initial graph in every propagation:

```python
W_sparse = W_list[i]
P_updated = W_sparse @ avg_P_others @ W_sparse.T
```

The product is followed by the main analysis's top-q sparsification and lazy row normalization. All views update synchronously. `vendor_core/` is the exact verified core snapshot; `source_manifest.json` checks its hashes. The compatibility workflow `workflows/scatac/hisemble/Hisemble_convergence.py` in both live code distributions now uses the same fixed-W propagation, keeping its original CLI and metric columns.

Run one full dataset in a fresh output directory:

```sh
python run_core_convergence.py --data-root /path/to/ensemble --output-dir /path/to/run/mca_Cerebellum_62216 --dataset mca_Cerebellum_62216 --iterations 50 --chunk-size 2048 --seed 42
python export_original_metrics.py --input-dir /path/to/run --output-dir /path/to/run/original_metrics --iterations 50
```

Run all 24 datasets:

```sh
python launch_all.py --data-root /path/to/ensemble --run-root /path/to/new_run --iterations 50 --workers 4 --threads 8
```

The batch launcher exports original metrics after successful numerical runs. Figure rendering is separate from this numerical pipeline.

Paper metric definitions:

- `Relative Change = sum(abs(P_next-P_old))/(N*q)`, where k=15 and q=2k=30.
- `Delta_Q` is recorded before each P update, as in the original monitoring code. Record 1 is 0; record t>=2 uses diagnostic `Q_delta_fro` at iteration t-1.

The full numerical runner retains additional diagnostics in its raw CSV. `export_original_metrics.py` exports only the paper's original P/Q metrics, checks trajectory completeness and converts the Q timing. Resource measurements include diagnostics and do not replace benchmark measurements.

All views use identical cells and row order; scBasset and SnapATAC2 use Gaussian affinity and LSI uses nonnegative cosine affinity. The runner checks embeddings, saves consensus_T20.npz, and records partial/complete status without silently resuming incomplete runs. The checked-in dataset list preserves all 24 benchmark IDs. Default diagnostic length is 100; specify `--iterations 50` to reproduce the displayed range. Historical biological input pickles require their compatible AnnData/pandas environment.

Run small-fixture checks with `python test_runner_equivalence.py`. These checks do not rerun the full biological benchmarks.

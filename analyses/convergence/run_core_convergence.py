#!/usr/bin/env python3
"""Observe the submitted fixed-W Hisemble core without changing its calculations.

The vendored graph.py/core.py/io.py are verbatim source snapshots. Iteration
updates below repeat core.py's statements, including dtype, sparse operation
order, tie handling, normalization, q=2k, and final symmetrization. Diagnostics
observe those arrays; they never threshold, reorder, or mutate a core state.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import pickle
import platform
import resource
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import scipy
from scipy.sparse import csr_matrix, save_npz
import sklearn

from vendor_core.graph import build_knn_graph_chunked_cpu, normalize_sparse, sparsify_matrix
from vendor_core.io import validate_embeddings, validation_report_as_dict

HERE = Path(__file__).resolve().parent
VIEWS = ("scBasset", "SnapATAC2", "LSI")
METRICS = ("gaussian", "gaussian", "cosine")


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, value):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def atomic_csv(path: Path, rows):
    if not rows:
        return
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def array_hash(array: np.ndarray):
    digest = hashlib.sha256()
    digest.update(str(array.shape).encode())
    digest.update(str(array.dtype).encode())
    for start in range(0, len(array), 2048):
        digest.update(np.ascontiguousarray(array[start:start + 2048]).tobytes())
    return digest.hexdigest()


def source_manifest():
    manifest = json.loads((HERE / "source_manifest.json").read_text())
    for name, expected in manifest["sha256"].items():
        actual = hashlib.sha256((HERE / "vendor_core" / name).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"Vendored source changed: {name}")
    manifest["runner_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    return manifest


def peak_rss_bytes():
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(raw if sys.platform == "darwin" else 1024 * raw)


def load_embeddings(data_root: Path, dataset: str):
    paths = (
        data_root / "ADATA" / f"{dataset}_scBasset.pkl",
        data_root / "ADATA" / f"{dataset}_snap.pkl",
        data_root / "lsi_results" / "ADATA" / f"{dataset}_lsi.pkl",
    )
    keys = ("projection", "X_spectral", "X_lsi")
    arrays, identifiers, input_records = {}, {}, {}
    for name, path, key in zip(VIEWS, paths, keys):
        with path.open("rb") as handle:
            value = pickle.load(handle)
        arrays[name] = np.asarray(value.obsm[key])
        identifiers[name] = np.asarray(value.obs_names, dtype=str)
        stat = path.stat()
        input_records[name] = {
            "path": str(path.resolve()), "file_size": stat.st_size,
            "file_mtime_ns": stat.st_mtime_ns, "embedding_key": key,
            "shape": list(arrays[name].shape), "dtype": str(arrays[name].dtype),
            "embedding_sha256": array_hash(arrays[name]),
            "ordered_cell_ids_sha256": hashlib.sha256(
                "\0".join(identifiers[name]).encode()
            ).hexdigest(),
        }
        del value
    reports = validate_embeddings(arrays, cell_ids_by_view=identifiers)
    n_cells = len(identifiers[VIEWS[0]])
    del identifiers
    return arrays, n_cells, input_records, validation_report_as_dict(reports)


def sparse_norm(matrix):
    data = matrix.data.astype(np.float64)
    return float(np.sqrt(np.dot(data, data)))


def sparse_delta(new, old):
    delta = (new - old).tocsr()
    delta.eliminate_zeros()
    absolute_fro = sparse_norm(delta)
    baseline = sparse_norm(old)
    # A zero denominator is explicitly undefined unless both matrices are zero.
    relative = absolute_fro / baseline if baseline else (0.0 if not absolute_fro else None)
    return {
        "delta_fro": absolute_fro,
        "relative_fro": relative,
        "delta_max_abs": float(np.max(np.abs(delta.data), initial=0.0)),
        "delta_l1": float(np.abs(delta.data).sum(dtype=np.float64)),
        "previous_fro": baseline,
    }


def actual_support_change(new, old):
    # Copies are used: stored zeros are discarded only in the diagnostic copies.
    left, right = new.copy(), old.copy()
    left.eliminate_zeros()
    right.eliminate_zeros()
    left.data = np.ones(left.nnz, dtype=np.int8)
    right.data = np.ones(right.nnz, dtype=np.int8)
    difference = (left - right).tocsr()
    difference.eliminate_zeros()
    union_size = (left.nnz + right.nnz + difference.nnz) // 2
    return {
        "support_symmetric_difference_nnz": int(difference.nnz),
        "support_union_nnz": int(union_size),
        "support_change_fraction": float(difference.nnz / union_size) if union_size else 0.0,
        "support_changed_rows": int(np.count_nonzero(np.diff(difference.indptr))),
        "support_changed_row_fraction": float(np.count_nonzero(np.diff(difference.indptr)) / new.shape[0]),
        "previous_support_nnz": int(right.nnz),
        "current_support_nnz": int(left.nnz),
    }


def boundary_gap(matrix, q):
    """Exact q-th minus (q+1)-th value, including implicit zeros; no dense N×N."""
    matrix = matrix.tocsr(copy=False)
    gaps = np.empty(matrix.shape[0], dtype=np.float64)
    positive_counts = np.empty(matrix.shape[0], dtype=np.int64)
    for row in range(matrix.shape[0]):
        values = matrix.data[matrix.indptr[row]:matrix.indptr[row + 1]]
        if np.any(values < 0):
            raise ValueError("Negative raw affinity: nonnegative top-q diagnostics would be invalid")
        positive = values[values > 0]
        size = len(positive)
        positive_counts[row] = size
        if size > q:
            partition = np.partition(positive, (size - q - 1, size - q))
            gaps[row] = float(partition[size - q] - partition[size - q - 1])
        elif size == q:
            gaps[row] = float(positive.min())
        else:
            gaps[row] = 0.0
    enough = positive_counts >= q
    # The strict gap premise is a minimum across ALL rows, not an average gap.
    return {
        "raw_A_nnz": int(matrix.nnz),
        "boundary_gap_min": float(gaps.min()),
        "boundary_gap_median": float(np.median(gaps)),
        "boundary_gap_q05": float(np.quantile(gaps, 0.05)),
        "boundary_gap_zero_rows": int(np.count_nonzero(gaps == 0)),
        "boundary_gap_strict_fraction": float(np.mean(gaps > 0)),
        "rows_with_fewer_than_q_positive_entries": int(np.count_nonzero(~enough)),
        "boundary_gap_min_rows_with_at_least_q_positive": float(gaps[enough].min()) if enough.any() else None,
    }


def view_average(P_list, excluded):
    # Statement order and float32 initial accumulator match core.py exactly.
    n_samples = P_list[0].shape[0]
    sum_P_others = csr_matrix((n_samples, n_samples), dtype=np.float32)
    for j in range(len(P_list)):
        if excluded != j:
            sum_P_others += P_list[j]
    return sum_P_others / (len(P_list) - 1)


def final_consensus(P_list):
    num_views = len(P_list)
    n_samples = P_list[0].shape[0]
    final_P = csr_matrix((n_samples, n_samples), dtype=np.float32)
    for P in P_list:
        final_P += P
    final_P /= num_views
    return (final_P + final_P.T) / 2


def instrumented_fusion(
    W_list, *, dataset="synthetic", k_neighbors=15, n_iterations=100,
    random_seed=42, output_dir=None, track_a=False, progress=None,
):
    num_views = len(W_list)
    n_samples = W_list[0].shape[0]
    P_list = [normalize_sparse(W) for W in W_list]
    k_retained = k_neighbors * 2
    rows = []
    previous_A = [None] * num_views
    graph_T20 = None
    for iter_num in range(n_iterations):
        iteration_started = time.perf_counter()
        P_updated_list, Q_previous_list, iteration_diagnostics = [], [], []
        for i in range(num_views):
            # These numerical statements repeat the submitted core exactly.
            sum_P_others = csr_matrix((n_samples, n_samples), dtype=np.float32)
            for j in range(num_views):
                if i != j:
                    sum_P_others += P_list[j]
            avg_P_others = sum_P_others / (num_views - 1)
            W_sparse = W_list[i]
            P_updated = W_sparse @ avg_P_others @ W_sparse.T
            P_updated_sparsified = sparsify_matrix(P_updated, k_retained, random_seed=random_seed)
            P_next = normalize_sparse(P_updated_sparsified)
            P_updated_list.append(P_next)
            Q_previous_list.append(avg_P_others)

            # Read-only observations AFTER the exact update has been calculated.
            item = boundary_gap(P_updated, k_retained)
            if track_a and previous_A[i] is not None:
                item.update({"A_" + key: value for key, value in sparse_delta(P_updated, previous_A[i]).items()})
            else:
                item.update({"A_" + key: None for key in ("delta_fro", "relative_fro", "delta_max_abs", "delta_l1", "previous_fro")})
            if track_a:
                previous_A[i] = P_updated
            iteration_diagnostics.append(item)
            del P_updated, P_updated_sparsified

        for i in range(num_views):
            pd = sparse_delta(P_updated_list[i], P_list[i])
            qd = sparse_delta(view_average(P_updated_list, i), Q_previous_list[i])
            row_sums = np.asarray(P_updated_list[i].sum(axis=1)).ravel()
            row = {
                "Dataset": dataset, "Iteration": iter_num + 1, "View": VIEWS[i],
                "n_cells": n_samples, "k_initial": k_neighbors, "q_retained": k_retained,
                **{"P_" + key: value for key, value in pd.items()},
                **{"Q_" + key: value for key, value in qd.items()},
                **actual_support_change(P_updated_list[i], P_list[i]),
                **iteration_diagnostics[i],
                "legacy_scaled_absolute_L1": pd["delta_l1"] / (n_samples * k_retained),
                "P_row_sum_min": float(row_sums.min()), "P_row_sum_max": float(row_sums.max()),
                "P_min_stored_value": float(P_updated_list[i].data.min()) if P_updated_list[i].nnz else 0.0,
                "P_max_stored_value": float(P_updated_list[i].data.max()) if P_updated_list[i].nnz else 0.0,
                "iteration_diagnostic_wall_s": None,
                "process_peak_rss_bytes_including_diagnostics": peak_rss_bytes(),
            }
            rows.append(row)
        P_list = P_updated_list
        del Q_previous_list
        elapsed = time.perf_counter() - iteration_started
        for row in rows[-num_views:]:
            row["iteration_diagnostic_wall_s"] = elapsed
        if iter_num + 1 == 20:
            graph_T20 = final_consensus(P_list)
            if output_dir is not None:
                temporary = output_dir / "consensus_T20.tmp.npz"
                save_npz(temporary, graph_T20)
                os.replace(temporary, output_dir / "consensus_T20.npz")
                graph_T20 = None  # The saved T20 graph need not remain resident through T100.
        if output_dir is not None:
            atomic_csv(output_dir / "convergence_history.csv", rows)
        if progress is not None:
            progress(iter_num + 1, rows[-num_views:])
        print(json.dumps({"dataset": dataset, "iteration": iter_num + 1, "P_relative_fro": [r["P_relative_fro"] for r in rows[-num_views:]], "support_change_fraction": [r["support_change_fraction"] for r in rows[-num_views:]], "seconds_including_diagnostics": elapsed}), flush=True)
    return final_consensus(P_list), graph_T20, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--dataset")
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--chunk-size", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--k", type=int, default=15)
    parser.add_argument("--track-a", action="store_true", help="Retain previous raw A products for max/Fro change; substantial extra memory")
    parser.add_argument("--list-datasets", action="store_true")
    args = parser.parse_args()
    datasets = json.loads((HERE / "datasets.json").read_text())
    if args.list_datasets:
        print("\n".join(datasets))
        return
    for name in ("data_root", "output_dir", "dataset"):
        if getattr(args, name) is None:
            parser.error("--" + name.replace("_", "-") + " is required")
    if args.dataset not in datasets:
        parser.error("Dataset is not one of the original 24 convergence datasets")
    if args.iterations < 20 or args.k < 1 or args.chunk_size < 1:
        parser.error("iterations must be >=20, k and chunk-size must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if (args.output_dir / "metadata.json").exists():
        parser.error("Output already has metadata.json; choose a fresh directory to avoid overwriting a run")
    started = time.perf_counter()
    metadata = {
        "dataset": args.dataset, "status": "running", "stage": "loading_inputs",
        "started_utc": utc_now(), "completed_iterations": 0,
        "target_iterations": args.iterations, "views": list(VIEWS), "metrics": list(METRICS),
        "k_initial": args.k, "q_retained": 2 * args.k, "random_seed": args.seed,
        "chunk_size": args.chunk_size, "track_A_change": args.track_a,
        "source": source_manifest(),
        "environment": {"python": sys.version, "numpy": np.__version__, "scipy": scipy.__version__, "scikit_learn": sklearn.__version__, "platform": platform.platform(), "hostname": platform.node(), "logical_cpu_count": os.cpu_count(), "thread_environment": {key: os.environ.get(key) for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")}},
        "timing_scope": "Includes input validation, hashing, graph construction, fusion, diagnostics, and checkpoint I/O; not comparable to main-paper runtime benchmark",
        "metric_definitions": {
            "P_relative_fro": "||P_t-P_(t-1)||_F / ||P_(t-1)||_F",
            "Q_relative_fro": "||Q_t-Q_(t-1)||_F / ||Q_(t-1)||_F; Q_t mean of other P_t views",
            "A_delta_max_abs": "Optional: max|A_(t-1)-A_(t-2)|; raw product A_s=W_i Q_s W_i^T before top-q; first iteration undefined",
            "support_change_fraction": "Symmetric difference of nonzero supports of P_t and P_(t-1), divided by support union; diagonal included",
            "boundary_gap": "Raw A_(t-1) row q-th minus (q+1)-th largest value, including implicit zeros; q=2k; nonnegative entries",
            "legacy_scaled_absolute_L1": "sum|P_t-P_(t-1)|/(N*q), not a relative matrix norm",
        },
        "consensus_T20_saved": False,
    }
    def write_status():
        metadata["updated_utc"] = utc_now()
        metadata["elapsed_s_including_diagnostics"] = time.perf_counter() - started
        metadata["process_peak_rss_bytes_including_diagnostics"] = peak_rss_bytes()
        atomic_json(args.output_dir / "metadata.json", metadata)
    write_status()
    try:
        embeddings, n_cells, input_records, validation = load_embeddings(args.data_root, args.dataset)
        if 2 * args.k >= n_cells:
            raise ValueError("q=2k must be smaller than N for a q/(q+1) boundary")
        metadata.update({"n_cells": n_cells, "inputs": input_records, "input_validation": validation, "stage": "building_graphs"})
        write_status()
        W_list = []
        build_times = {}
        for name, metric in zip(VIEWS, METRICS):
            graph_started = time.perf_counter()
            W_list.append(build_knn_graph_chunked_cpu(embeddings[name], args.k, metric, chunk_size=args.chunk_size, random_seed=args.seed))
            build_times[name] = time.perf_counter() - graph_started
            metadata["graphs_built"] = len(W_list)
            metadata["graph_build_wall_s"] = build_times
            write_status()
        del embeddings
        metadata["stage"] = "fusion_and_diagnostics"
        metadata["initial_graph_nnz"] = [int(w.nnz) for w in W_list]
        write_status()
        def progress(iteration, latest):
            metadata["completed_iterations"] = iteration
            metadata["consensus_T20_saved"] = iteration >= 20
            metadata["latest_P_relative_fro"] = [row["P_relative_fro"] for row in latest]
            metadata["latest_support_change_fraction"] = [row["support_change_fraction"] for row in latest]
            write_status()
        instrumented_fusion(W_list, dataset=args.dataset, k_neighbors=args.k, n_iterations=args.iterations, random_seed=args.seed, output_dir=args.output_dir, track_a=args.track_a, progress=progress)
        metadata.update({"status": "complete", "stage": "complete", "finished_utc": utc_now(), "trajectory_count": len(VIEWS), "history_row_count": args.iterations * len(VIEWS)})
        write_status()
        atomic_json(args.output_dir / "COMPLETE.json", {"dataset": args.dataset, "finished_utc": metadata["finished_utc"], "iterations": args.iterations, "views": list(VIEWS), "runner_sha256": metadata["source"]["runner_sha256"]})
    except BaseException as error:
        metadata.update({"status": "failed", "stage": "failed", "error_type": type(error).__name__, "error": str(error), "traceback": traceback.format_exc()})
        write_status()
        raise


if __name__ == "__main__":
    main()

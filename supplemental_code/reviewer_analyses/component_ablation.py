#!/usr/bin/env python3
"""Single-component ablation study for Hisemble.

Variants
--------
full
    Sparse kNN affinity + iterative cross-diffusion + dynamic top-k sparsification.
dense_initial
    Dense initial affinity + cross-diffusion + dynamic top-k sparsification.
no_diffusion
    Sparse kNN affinity matrices are normalized and averaged without diffusion.
no_dynamic_sparsification
    Sparse initial affinity + cross-diffusion, but no per-iteration top-k operator.

Run each variant in a separate process when comparing peak memory.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import resource
import time
from pathlib import Path

import igraph as ig
import leidenalg
import numpy as np
import scipy.sparse as sp
from scipy.sparse import csr_matrix
from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    homogeneity_score,
    normalized_mutual_info_score,
)
from sklearn.metrics.pairwise import cosine_similarity, rbf_kernel


VARIANTS = (
    "full",
    "dense_initial",
    "no_diffusion",
    "no_dynamic_sparsification",
)


def build_affinity(
    embedding: np.ndarray,
    k_neighbors: int,
    metric: str,
    dense: bool,
    chunk_size: int = 2048,
) -> csr_matrix:
    n_cells = embedding.shape[0]
    gamma = None
    if metric == "gaussian":
        variance = float(embedding.var())
        gamma = 1.0 / (embedding.shape[1] * max(variance, 1e-12))

    if dense:
        matrix = np.empty((n_cells, n_cells), dtype=np.float32)
        for start in range(0, n_cells, chunk_size):
            stop = min(start + chunk_size, n_cells)
            chunk = embedding[start:stop]
            if metric == "gaussian":
                similarity = rbf_kernel(chunk, embedding, gamma=gamma)
            elif metric == "cosine":
                similarity = np.maximum(cosine_similarity(chunk, embedding), 0)
            else:
                raise ValueError(metric)
            matrix[start:stop] = similarity.astype(np.float32, copy=False)
        np.fill_diagonal(matrix, 0.0)
        matrix = (matrix + matrix.T) / 2.0
        return csr_matrix(matrix)

    rows_all, cols_all, data_all = [], [], []
    for start in range(0, n_cells, chunk_size):
        stop = min(start + chunk_size, n_cells)
        chunk = embedding[start:stop]
        if metric == "gaussian":
            similarity = rbf_kernel(chunk, embedding, gamma=gamma)
        elif metric == "cosine":
            similarity = np.maximum(cosine_similarity(chunk, embedding), 0)
        else:
            raise ValueError(metric)
        indices = np.argsort(-similarity, axis=1)[:, 1 : k_neighbors + 1]
        values = np.take_along_axis(similarity, indices, axis=1)
        rows_all.append(np.repeat(np.arange(start, stop), k_neighbors))
        cols_all.append(indices.ravel())
        data_all.append(values.ravel())
    graph = csr_matrix(
        (np.concatenate(data_all), (np.concatenate(rows_all), np.concatenate(cols_all))),
        shape=(n_cells, n_cells),
    )
    return ((graph + graph.T) / 2).tocsr()


def normalize(graph: csr_matrix) -> csr_matrix:
    row_sums = graph.sum(axis=1).A1
    denominators = 2.0 * row_sums
    denominators[denominators == 0] = 1e-9
    return (sp.diags(1.0 / denominators) @ graph + sp.eye(graph.shape[0]) * 0.5).tocsr()


def top_k(matrix: csr_matrix, k_neighbors: int) -> csr_matrix:
    rows_all, cols_all, data_all = [], [], []
    for row_index in range(matrix.shape[0]):
        row = matrix.getrow(row_index)
        if row.nnz > k_neighbors:
            keep = np.argpartition(-row.data, k_neighbors)[:k_neighbors]
            cols = row.indices[keep]
            data = row.data[keep]
        else:
            cols = row.indices
            data = row.data
        rows_all.append(np.full(len(cols), row_index, dtype=np.int32))
        cols_all.append(cols)
        data_all.append(data)
    return csr_matrix(
        (np.concatenate(data_all), (np.concatenate(rows_all), np.concatenate(cols_all))),
        shape=matrix.shape,
    )


def fuse(
    graphs: list[csr_matrix],
    k_neighbors: int,
    n_iterations: int,
    use_diffusion: bool,
    use_dynamic_sparsification: bool,
) -> tuple[csr_matrix, list[int]]:
    states = [normalize(graph) for graph in graphs]
    nnz_history = [sum(state.nnz for state in states)]
    if not use_diffusion:
        fused = sum(states, start=csr_matrix(graphs[0].shape)) / len(states)
        return ((fused + fused.T) / 2).tocsr(), nnz_history

    for _ in range(n_iterations):
        updated = []
        for view_index, graph in enumerate(graphs):
            consensus = sum(
                (state for index, state in enumerate(states) if index != view_index),
                start=csr_matrix(graph.shape),
            ) / (len(states) - 1)
            propagated = graph @ consensus @ graph.T
            if use_dynamic_sparsification:
                propagated = top_k(propagated, 2 * k_neighbors)
            updated.append(normalize(propagated))
        states = updated
        nnz_history.append(sum(state.nnz for state in states))
    fused = sum(states, start=csr_matrix(graphs[0].shape)) / len(states)
    return ((fused + fused.T) / 2).tocsr(), nnz_history


def leiden_membership(network: ig.Graph, resolution: float, seed: int) -> np.ndarray:
    partition = leidenalg.find_partition(
        network,
        leidenalg.RBConfigurationVertexPartition,
        weights=network.es["weight"],
        resolution_parameter=resolution,
        seed=seed,
    )
    return np.asarray(partition.membership)


def cluster_to_count(
    fused: csr_matrix,
    target_count: int,
    seed: int,
    low: float = 0.1,
    high: float = 5.0,
    tolerance: float = 0.01,
) -> tuple[np.ndarray, float]:
    sources, targets = fused.nonzero()
    network = ig.Graph(n=fused.shape[0], edges=list(zip(sources, targets)), directed=False)
    network.es["weight"] = fused[sources, targets].A1
    best_membership = leiden_membership(network, high, seed)
    best_resolution = high
    best_delta = abs(len(np.unique(best_membership)) - target_count)
    while high - low > tolerance:
        middle = (low + high) / 2
        membership = leiden_membership(network, middle, seed)
        count = len(np.unique(membership))
        delta = abs(count - target_count)
        if delta < best_delta:
            best_membership, best_resolution, best_delta = membership, middle, delta
        if count < target_count:
            low = middle
        elif count > target_count:
            high = middle
        else:
            return membership, middle
    return best_membership, best_resolution


def peak_memory_mb() -> float:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Linux reports KiB; macOS reports bytes.
    return value / (1024.0 if os.uname().sysname == "Linux" else 1024.0**2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--k", type=int, default=15)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample-size", type=int)
    parser.add_argument("--skip-clustering", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    paths = {
        "scBasset": args.data_root / "ADATA" / f"{args.dataset}_scBasset.pkl",
        "SnapATAC2": args.data_root / "ADATA" / f"{args.dataset}_snap.pkl",
        "LSI": args.data_root / "lsi_results" / "ADATA" / f"{args.dataset}_lsi.pkl",
    }
    adata = {}
    for name, path in paths.items():
        with path.open("rb") as handle:
            adata[name] = pickle.load(handle)
    names = np.asarray(adata["SnapATAC2"].obs_names)
    for name, value in adata.items():
        if not np.array_equal(np.asarray(value.obs_names), names):
            raise ValueError(f"Cell order differs for {name}")
    labels = adata["SnapATAC2"].obs["cell_type"].astype(str).to_numpy()

    sample_suffix = ""
    if args.sample_size is not None:
        if args.sample_size > len(labels):
            raise ValueError("sample-size exceeds the number of cells")
        rng = np.random.default_rng(args.seed)
        selected = np.sort(rng.choice(len(labels), args.sample_size, replace=False))
        labels = labels[selected]
        embeddings = {
            "scBasset": adata["scBasset"].obsm["projection"][selected],
            "SnapATAC2": adata["SnapATAC2"].obsm["X_spectral"][selected],
            "LSI": adata["LSI"].obsm["X_lsi"][selected],
        }
        sample_suffix = f"__n{args.sample_size}"
    else:
        embeddings = {
            "scBasset": adata["scBasset"].obsm["projection"],
            "SnapATAC2": adata["SnapATAC2"].obsm["X_spectral"],
            "LSI": adata["LSI"].obsm["X_lsi"],
        }

    dense_initial = args.variant == "dense_initial"
    graph_started = time.perf_counter()
    graph_cpu_started = time.process_time()
    graphs = [
        build_affinity(embeddings["scBasset"], args.k, "gaussian", dense_initial),
        build_affinity(embeddings["SnapATAC2"], args.k, "gaussian", dense_initial),
        build_affinity(embeddings["LSI"], args.k, "cosine", dense_initial),
    ]
    graph_seconds = time.perf_counter() - graph_started
    graph_cpu_seconds = time.process_time() - graph_cpu_started

    fusion_started = time.perf_counter()
    fusion_cpu_started = time.process_time()
    fused, nnz_history = fuse(
        graphs,
        args.k,
        args.iterations,
        use_diffusion=args.variant != "no_diffusion",
        use_dynamic_sparsification=args.variant != "no_dynamic_sparsification",
    )
    fusion_seconds = time.perf_counter() - fusion_started
    fusion_cpu_seconds = time.process_time() - fusion_cpu_started
    if args.skip_clustering:
        clusters = None
        resolution = None
    else:
        clusters, resolution = cluster_to_count(fused, len(np.unique(labels)), args.seed)
    result = {
        "dataset": args.dataset,
        "variant": args.variant,
        "n_cells": len(labels),
        "source_n_cells": len(names),
        "subsampled": args.sample_size is not None,
        "k": args.k,
        "iterations": args.iterations,
        "seed": args.seed,
        "resolution": resolution,
        "n_clusters": int(len(np.unique(clusters))) if clusters is not None else None,
        "ARI": adjusted_rand_score(labels, clusters) if clusters is not None else None,
        "AMI": adjusted_mutual_info_score(labels, clusters) if clusters is not None else None,
        "NMI": normalized_mutual_info_score(labels, clusters) if clusters is not None else None,
        "HOM": homogeneity_score(labels, clusters) if clusters is not None else None,
        "graph_construction_seconds": graph_seconds,
        "graph_construction_cpu_seconds": graph_cpu_seconds,
        "fusion_seconds": fusion_seconds,
        "fusion_cpu_seconds": fusion_cpu_seconds,
        "total_seconds": time.perf_counter() - started,
        "peak_memory_mb": peak_memory_mb(),
        "initial_total_nnz": nnz_history[0],
        "maximum_total_nnz": max(nnz_history),
        "final_total_nnz": nnz_history[-1],
        "fused_nnz": fused.nnz,
        "nnz_history": nnz_history,
    }
    output = args.output_dir / f"{args.dataset}__{args.variant}{sample_suffix}.json"
    with output.open("w") as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

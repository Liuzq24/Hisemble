#!/usr/bin/env python3
"""Visualize the contribution of LSI to Hisemble on one dataset.

The script runs the same sparse cross-diffusion workflow with all three
representations and with LSI omitted.  All panels use a single UMAP layout
derived from the full fused graph so that apparent differences are not caused
by independent UMAP rotations or stochastic layouts.
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import igraph as ig
import leidenalg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse as sp
import seaborn as sns
import umap
from scipy.optimize import linear_sum_assignment
from scipy.sparse import csr_matrix
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.metrics.pairwise import cosine_similarity, rbf_kernel


def build_knn_graph(
    embedding: np.ndarray,
    k_neighbors: int,
    metric: str,
    chunk_size: int = 2048,
) -> csr_matrix:
    rows_all, cols_all, data_all = [], [], []
    gamma = None
    if metric == "gaussian":
        variance = float(embedding.var())
        gamma = 1.0 / (embedding.shape[1] * max(variance, 1e-12))

    for start in range(0, embedding.shape[0], chunk_size):
        stop = min(start + chunk_size, embedding.shape[0])
        chunk = embedding[start:stop]
        if metric == "gaussian":
            similarity = rbf_kernel(chunk, embedding, gamma=gamma)
        elif metric == "cosine":
            similarity = np.maximum(cosine_similarity(chunk, embedding), 0)
        else:
            raise ValueError(f"Unsupported metric: {metric}")

        indices = np.argsort(-similarity, axis=1)[:, 1 : k_neighbors + 1]
        values = np.take_along_axis(similarity, indices, axis=1)
        rows_all.append(np.repeat(np.arange(start, stop), k_neighbors))
        cols_all.append(indices.ravel())
        data_all.append(values.ravel())

    graph = csr_matrix(
        (np.concatenate(data_all), (np.concatenate(rows_all), np.concatenate(cols_all))),
        shape=(embedding.shape[0], embedding.shape[0]),
    )
    return ((graph + graph.T) / 2).tocsr()


def normalize_sparse(graph: csr_matrix) -> csr_matrix:
    row_sums = graph.sum(axis=1).A1
    denominators = 2.0 * row_sums
    denominators[denominators == 0] = 1e-9
    normalized = sp.diags(1.0 / denominators) @ graph
    return (normalized + sp.diags(np.full(graph.shape[0], 0.5))).tocsr()


def sparsify(matrix: csr_matrix, k_neighbors: int) -> csr_matrix:
    rows_all, cols_all, data_all = [], [], []
    for row_index in range(matrix.shape[0]):
        row = matrix.getrow(row_index)
        if row.nnz > k_neighbors:
            keep = np.argpartition(-row.data, k_neighbors)[:k_neighbors]
            data = row.data[keep]
            cols = row.indices[keep]
        else:
            data = row.data
            cols = row.indices
        rows_all.append(np.full(len(cols), row_index, dtype=np.int32))
        cols_all.append(cols)
        data_all.append(data)
    return csr_matrix(
        (np.concatenate(data_all), (np.concatenate(rows_all), np.concatenate(cols_all))),
        shape=matrix.shape,
    )


def hisemble(
    graphs: list[csr_matrix], k_neighbors: int, n_iterations: int
) -> csr_matrix:
    states = [normalize_sparse(graph) for graph in graphs]
    retained = 2 * k_neighbors
    for _ in range(n_iterations):
        updated = []
        for view_index, graph in enumerate(graphs):
            consensus = sum(
                (state for index, state in enumerate(states) if index != view_index),
                start=csr_matrix(graph.shape, dtype=np.float64),
            ) / (len(states) - 1)
            propagated = graph @ consensus @ graph.T
            updated.append(normalize_sparse(sparsify(propagated, retained)))
        states = updated
    fused = sum(states, start=csr_matrix(graphs[0].shape, dtype=np.float64)) / len(states)
    return ((fused + fused.T) / 2).tocsr()


def leiden_membership(network: ig.Graph, resolution: float, seed: int) -> np.ndarray:
    partition = leidenalg.find_partition(
        network,
        leidenalg.RBConfigurationVertexPartition,
        weights=network.es["weight"],
        resolution_parameter=resolution,
        seed=seed,
    )
    return np.asarray(partition.membership)


def cluster_to_known_count(
    fused: csr_matrix,
    target_count: int,
    seed: int,
    low: float = 0.1,
    high: float = 5.0,
    tolerance: float = 0.01,
) -> tuple[np.ndarray, float]:
    sources, targets = fused.nonzero()
    weights = fused[sources, targets].A1
    network = ig.Graph(n=fused.shape[0], edges=list(zip(sources, targets)), directed=False)
    network.es["weight"] = weights

    best_resolution = high
    best_membership = leiden_membership(network, high, seed)
    best_difference = abs(len(np.unique(best_membership)) - target_count)
    while high - low > tolerance:
        middle = (low + high) / 2
        membership = leiden_membership(network, middle, seed)
        count = len(np.unique(membership))
        difference = abs(count - target_count)
        if difference < best_difference:
            best_resolution = middle
            best_membership = membership
            best_difference = difference
        if count < target_count:
            low = middle
        elif count > target_count:
            high = middle
        else:
            best_resolution = middle
            best_membership = membership
            break
    return best_membership, best_resolution


def match_clusters_to_labels(labels: np.ndarray, clusters: np.ndarray) -> np.ndarray:
    label_levels = np.unique(labels)
    cluster_levels = np.unique(clusters)
    counts = np.zeros((len(cluster_levels), len(label_levels)), dtype=int)
    for row, cluster in enumerate(cluster_levels):
        for column, label in enumerate(label_levels):
            counts[row, column] = np.sum((clusters == cluster) & (labels == label))
    row_ind, col_ind = linear_sum_assignment(-counts)
    mapping = {cluster_levels[row]: label_levels[column] for row, column in zip(row_ind, col_ind)}
    for cluster in cluster_levels:
        if cluster not in mapping:
            subset = labels[clusters == cluster]
            mapping[cluster] = pd.Series(subset).value_counts().index[0]
    return np.asarray([mapping[cluster] for cluster in clusters], dtype=object)


def shared_umap(fused: csr_matrix, seed: int) -> np.ndarray:
    affinity = fused.toarray()
    affinity = np.maximum(affinity, affinity.T)
    scale = float(affinity.max())
    if scale > 0:
        affinity /= scale
    distance = 1.0 - affinity
    np.fill_diagonal(distance, 0.0)
    return umap.UMAP(
        n_neighbors=15,
        min_dist=0.25,
        metric="precomputed",
        random_state=seed,
    ).fit_transform(distance)


def plot_panels(
    coordinates: np.ndarray,
    labels: np.ndarray,
    full_labels: np.ndarray,
    no_lsi_labels: np.ndarray,
    metrics: dict,
    output_path: Path,
) -> None:
    levels = list(pd.Series(labels).value_counts().index)
    colors = sns.color_palette("tab20", n_colors=len(levels))
    palette = dict(zip(levels, colors))
    figure, axes = plt.subplots(1, 3, figsize=(15.5, 5.2), sharex=True, sharey=True)
    panels = [
        (labels, "Ground-truth cell types"),
        (
            full_labels,
            f"Hisemble (three views)\nARI={metrics['full']['ARI']:.3f}, NMI={metrics['full']['NMI']:.3f}",
        ),
        (
            no_lsi_labels,
            f"Hisemble without LSI\nARI={metrics['no_lsi']['ARI']:.3f}, NMI={metrics['no_lsi']['NMI']:.3f}",
        ),
    ]
    for axis, (values, title) in zip(axes, panels):
        for level in levels:
            mask = values == level
            axis.scatter(
                coordinates[mask, 0],
                coordinates[mask, 1],
                s=7,
                color=palette[level],
                linewidths=0,
                alpha=0.85,
                label=level,
            )
        axis.set_title(title, fontsize=11, fontweight="bold")
        axis.set_xlabel("UMAP 1")
        axis.set_ylabel("UMAP 2")
        axis.spines[["top", "right"]].set_visible(False)
    handles, legend_labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        legend_labels,
        loc="center left",
        bbox_to_anchor=(0.995, 0.5),
        frameon=False,
        fontsize=7,
        markerscale=1.8,
    )
    figure.tight_layout(rect=(0, 0, 0.82, 1))
    figure.savefig(output_path, dpi=600, bbox_inches="tight")
    figure.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--dataset", default="mca_Cerebellum_62216")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--k", type=int, default=15)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "scbasset": args.data_root / "ADATA" / f"{args.dataset}_scBasset.pkl",
        "snap": args.data_root / "ADATA" / f"{args.dataset}_snap.pkl",
        "lsi": args.data_root / "lsi_results" / "ADATA" / f"{args.dataset}_lsi.pkl",
    }
    objects = {}
    for name, path in paths.items():
        with path.open("rb") as handle:
            objects[name] = pickle.load(handle)

    reference_names = np.asarray(objects["snap"].obs_names)
    for name, adata in objects.items():
        if not np.array_equal(np.asarray(adata.obs_names), reference_names):
            raise ValueError(f"Cell order differs for {name}; explicit alignment is required")

    true_labels = objects["snap"].obs["cell_type"].astype(str).to_numpy()
    graphs = {
        "scbasset": build_knn_graph(objects["scbasset"].obsm["projection"], args.k, "gaussian"),
        "snap": build_knn_graph(objects["snap"].obsm["X_spectral"], args.k, "gaussian"),
        "lsi": build_knn_graph(objects["lsi"].obsm["X_lsi"], args.k, "cosine"),
    }

    full_fused = hisemble(list(graphs.values()), args.k, args.iterations)
    no_lsi_fused = hisemble([graphs["scbasset"], graphs["snap"]], args.k, args.iterations)
    target_count = len(np.unique(true_labels))
    full_clusters, full_resolution = cluster_to_known_count(full_fused, target_count, args.seed)
    no_lsi_clusters, no_lsi_resolution = cluster_to_known_count(no_lsi_fused, target_count, args.seed)

    full_mapped = match_clusters_to_labels(true_labels, full_clusters)
    no_lsi_mapped = match_clusters_to_labels(true_labels, no_lsi_clusters)
    metrics = {
        "full": {
            "ARI": adjusted_rand_score(true_labels, full_clusters),
            "NMI": normalized_mutual_info_score(true_labels, full_clusters),
            "matched_accuracy": float(np.mean(full_mapped == true_labels)),
            "resolution": full_resolution,
        },
        "no_lsi": {
            "ARI": adjusted_rand_score(true_labels, no_lsi_clusters),
            "NMI": normalized_mutual_info_score(true_labels, no_lsi_clusters),
            "matched_accuracy": float(np.mean(no_lsi_mapped == true_labels)),
            "resolution": no_lsi_resolution,
        },
    }
    metrics["delta"] = {
        key: metrics["full"][key] - metrics["no_lsi"][key]
        for key in ("ARI", "NMI", "matched_accuracy")
    }

    coordinates = shared_umap(full_fused, args.seed)
    per_cell = pd.DataFrame(
        {
            "cell_id": reference_names,
            "cell_type": true_labels,
            "umap_1": coordinates[:, 0],
            "umap_2": coordinates[:, 1],
            "full_cluster": full_clusters,
            "no_lsi_cluster": no_lsi_clusters,
            "full_matched_label": full_mapped,
            "no_lsi_matched_label": no_lsi_mapped,
            "full_correct": full_mapped == true_labels,
            "no_lsi_correct": no_lsi_mapped == true_labels,
        }
    )
    per_cell.to_csv(args.output_dir / "per_cell_assignments.csv", index=False)

    summary = []
    for label, frame in per_cell.groupby("cell_type", sort=False):
        summary.append(
            {
                "cell_type": label,
                "n_cells": len(frame),
                "full_matched_accuracy": frame["full_correct"].mean(),
                "no_lsi_matched_accuracy": frame["no_lsi_correct"].mean(),
                "accuracy_delta": frame["full_correct"].mean() - frame["no_lsi_correct"].mean(),
                "rescued_by_lsi": int((frame["full_correct"] & ~frame["no_lsi_correct"]).sum()),
                "lost_with_lsi": int((~frame["full_correct"] & frame["no_lsi_correct"]).sum()),
            }
        )
    pd.DataFrame(summary).sort_values("accuracy_delta", ascending=False).to_csv(
        args.output_dir / "per_cell_type_summary.csv", index=False
    )
    with (args.output_dir / "metrics.json").open("w") as handle:
        json.dump(metrics, handle, indent=2)
    plot_panels(
        coordinates,
        true_labels,
        full_mapped,
        no_lsi_mapped,
        metrics,
        args.output_dir / "mca_Cerebellum_LSI_ablation_UMAP.png",
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

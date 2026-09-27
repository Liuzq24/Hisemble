#!/usr/bin/env python3
"""Reviewer analysis: stability of Hisemble across Leiden random seeds.

The input embeddings and fused graph are held fixed. Only the seed used by the
final Leiden resolution search and partition is varied. This isolates the
stochasticity of the downstream community-detection step from upstream encoder
training and from the deterministic Hisemble fusion calculation.
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from component_ablation import build_affinity, cluster_to_count, fuse


DEFAULT_DATASETS = (
    "FCA_fetal_33184180_spleen",
    "FCA_fetal_33184180_cerebellum",
    "mca_Cerebellum_62216",
    "mca_LargeIntestineA_62816",
)
DEFAULT_SEEDS = (0, 1, 2, 3, 4, 42, 101, 202, 314, 777)


def load_dataset(data_root: Path, dataset: str) -> tuple[dict[str, np.ndarray], np.ndarray]:
    paths = {
        "scBasset": data_root / "ADATA" / f"{dataset}_scBasset.pkl",
        "SnapATAC2": data_root / "ADATA" / f"{dataset}_snap.pkl",
        "LSI": data_root / "lsi_results" / "ADATA" / f"{dataset}_lsi.pkl",
    }
    objects = {}
    for name, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(path)
        with path.open("rb") as handle:
            objects[name] = pickle.load(handle)

    reference_names = np.asarray(objects["SnapATAC2"].obs_names)
    for name, value in objects.items():
        if not np.array_equal(np.asarray(value.obs_names), reference_names):
            raise ValueError(f"Cell order differs for {name} in {dataset}")

    embeddings = {
        "scBasset": np.asarray(objects["scBasset"].obsm["projection"]),
        "SnapATAC2": np.asarray(objects["SnapATAC2"].obsm["X_spectral"]),
        "LSI": np.asarray(objects["LSI"].obsm["X_lsi"]),
    }
    for name, value in embeddings.items():
        if value.ndim != 2 or value.shape[0] != len(reference_names):
            raise ValueError(f"Invalid embedding shape for {name}: {value.shape}")
        if not np.isfinite(value).all():
            raise ValueError(f"Non-finite values in {name} for {dataset}")

    labels = objects["SnapATAC2"].obs["cell_type"].astype(str).to_numpy()
    return embeddings, labels


def summarize(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    return {
        "mean": float(array.mean()),
        "sd": float(array.std(ddof=1)) if len(array) > 1 else 0.0,
        "minimum": float(array.min()),
        "maximum": float(array.max()),
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--datasets", nargs="+", default=list(DEFAULT_DATASETS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--k", type=int, default=15)
    parser.add_argument("--iterations", type=int, default=20)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    per_seed_rows: list[dict[str, object]] = []
    pairwise_rows: list[dict[str, object]] = []
    summaries: dict[str, object] = {}

    for dataset in args.datasets:
        started = time.perf_counter()
        embeddings, labels = load_dataset(args.data_root, dataset)
        graphs = [
            build_affinity(embeddings["scBasset"], args.k, "gaussian", False),
            build_affinity(embeddings["SnapATAC2"], args.k, "gaussian", False),
            build_affinity(embeddings["LSI"], args.k, "cosine", False),
        ]
        fused, _ = fuse(
            graphs,
            args.k,
            args.iterations,
            use_diffusion=True,
            use_dynamic_sparsification=True,
        )

        memberships: dict[int, np.ndarray] = {}
        dataset_rows: list[dict[str, object]] = []
        for seed in args.seeds:
            membership, resolution = cluster_to_count(
                fused, len(np.unique(labels)), seed
            )
            memberships[seed] = membership
            row = {
                "dataset": dataset,
                "n_cells": len(labels),
                "seed": seed,
                "resolution": resolution,
                "n_clusters": len(np.unique(membership)),
                "ARI": adjusted_rand_score(labels, membership),
                "NMI": normalized_mutual_info_score(labels, membership),
            }
            per_seed_rows.append(row)
            dataset_rows.append(row)

        pairwise_values = []
        for left_index, left_seed in enumerate(args.seeds):
            for right_seed in args.seeds[left_index + 1 :]:
                agreement = adjusted_rand_score(
                    memberships[left_seed], memberships[right_seed]
                )
                pairwise_values.append(agreement)
                pairwise_rows.append(
                    {
                        "dataset": dataset,
                        "seed_1": left_seed,
                        "seed_2": right_seed,
                        "partition_ARI": agreement,
                    }
                )

        summaries[dataset] = {
            "n_cells": len(labels),
            "n_seeds": len(args.seeds),
            "seeds": args.seeds,
            "ARI": summarize([float(row["ARI"]) for row in dataset_rows]),
            "NMI": summarize([float(row["NMI"]) for row in dataset_rows]),
            "pairwise_partition_ARI": summarize(pairwise_values),
            "elapsed_seconds": time.perf_counter() - started,
        }

    write_csv(args.output_dir / "seed_stability_per_seed.csv", per_seed_rows)
    write_csv(args.output_dir / "seed_stability_pairwise_partitions.csv", pairwise_rows)
    with (args.output_dir / "seed_stability_summary.json").open("w") as handle:
        json.dump(
            {
                "analysis": "fixed embeddings and fused graph; Leiden seed varied",
                "k": args.k,
                "iterations": args.iterations,
                "datasets": summaries,
            },
            handle,
            indent=2,
        )


if __name__ == "__main__":
    main()

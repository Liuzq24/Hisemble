#!/usr/bin/env python3
"""Reviewer analysis: effect of a degraded upstream view on Hisemble.

The scBasset and SnapATAC2 graphs are held fixed. The LSI representation is
either clean, omitted, corrupted with feature-scale-matched Gaussian noise, or
row-permuted. Gaussian noise is defined as

    E_corrupt = E + sigma * G * feature_sd(E),

where G contains independent standard-normal values. Multiple noise seeds are
used. Leiden is held at seed 42 so that the analysis isolates input-view
degradation rather than downstream partition stochasticity.
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
DEFAULT_NOISE_LEVELS = (0.5, 1.0, 2.0)
DEFAULT_NOISE_SEEDS = (0, 1, 2, 3, 4)


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


def evaluate(
    graphs,
    labels: np.ndarray,
    k: int,
    iterations: int,
    leiden_seed: int,
) -> tuple[float, float, float, int]:
    fused, _ = fuse(
        graphs,
        k,
        iterations,
        use_diffusion=True,
        use_dynamic_sparsification=True,
    )
    membership, resolution = cluster_to_count(
        fused, len(np.unique(labels)), leiden_seed
    )
    return (
        adjusted_rand_score(labels, membership),
        normalized_mutual_info_score(labels, membership),
        resolution,
        len(np.unique(membership)),
    )


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--datasets", nargs="+", default=list(DEFAULT_DATASETS))
    parser.add_argument(
        "--noise-levels", nargs="+", type=float, default=list(DEFAULT_NOISE_LEVELS)
    )
    parser.add_argument(
        "--noise-seeds", nargs="+", type=int, default=list(DEFAULT_NOISE_SEEDS)
    )
    parser.add_argument("--k", type=int, default=15)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--leiden-seed", type=int, default=42)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    for dataset in args.datasets:
        embeddings, labels = load_dataset(args.data_root, dataset)
        fixed_graphs = [
            build_affinity(embeddings["scBasset"], args.k, "gaussian", False),
            build_affinity(embeddings["SnapATAC2"], args.k, "gaussian", False),
        ]
        lsi = embeddings["LSI"].astype(np.float64, copy=False)
        lsi_feature_sd = lsi.std(axis=0, keepdims=True)
        lsi_feature_sd[lsi_feature_sd == 0] = 1.0

        conditions: list[tuple[str, float | None, int | None, np.ndarray | None]] = [
            ("clean", 0.0, None, lsi),
            ("omit_lsi", None, None, None),
        ]
        for noise_level in args.noise_levels:
            for noise_seed in args.noise_seeds:
                rng = np.random.default_rng(noise_seed)
                corrupt = lsi + noise_level * rng.normal(size=lsi.shape) * lsi_feature_sd
                conditions.append(("gaussian_noise", noise_level, noise_seed, corrupt))
        for noise_seed in args.noise_seeds:
            rng = np.random.default_rng(noise_seed)
            conditions.append(("row_permuted", None, noise_seed, lsi[rng.permutation(len(lsi))]))

        for condition, noise_level, noise_seed, lsi_input in conditions:
            started = time.perf_counter()
            graphs = list(fixed_graphs)
            if lsi_input is not None:
                graphs.append(build_affinity(lsi_input, args.k, "cosine", False))
            ari, nmi, resolution, n_clusters = evaluate(
                graphs, labels, args.k, args.iterations, args.leiden_seed
            )
            rows.append(
                {
                    "dataset": dataset,
                    "n_cells": len(labels),
                    "condition": condition,
                    "noise_level": noise_level,
                    "noise_seed": noise_seed,
                    "leiden_seed": args.leiden_seed,
                    "n_views": len(graphs),
                    "resolution": resolution,
                    "n_clusters": n_clusters,
                    "ARI": ari,
                    "NMI": nmi,
                    "elapsed_seconds": time.perf_counter() - started,
                }
            )
        # Keep a valid checkpoint after every dataset so that completed work is
        # recoverable if a long remote run is interrupted.
        write_csv(args.output_dir / "degraded_view_results.csv", rows)
        print(
            f"completed {dataset}: "
            f"{sum(row['dataset'] == dataset for row in rows)} conditions",
            flush=True,
        )

    with (args.output_dir / "degraded_view_design.json").open("w") as handle:
        json.dump(
            {
                "datasets": args.datasets,
                "noise_levels": args.noise_levels,
                "noise_seeds": args.noise_seeds,
                "leiden_seed": args.leiden_seed,
                "k": args.k,
                "iterations": args.iterations,
                "gaussian_noise_definition": (
                    "E_corrupt = E + noise_level * N(0,1) * feature_sd(E)"
                ),
                "row_permutation_definition": (
                    "LSI rows randomly permuted while scBasset, SnapATAC2, and labels remain fixed"
                ),
            },
            handle,
            indent=2,
        )


if __name__ == "__main__":
    main()

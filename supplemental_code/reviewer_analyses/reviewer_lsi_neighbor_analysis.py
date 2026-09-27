#!/usr/bin/env python3
"""Quantify label-consistent local neighborhoods unique to the LSI view."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors


def neighbors(embedding: np.ndarray, k: int, metric: str) -> np.ndarray:
    model = NearestNeighbors(n_neighbors=k + 1, metric=metric, algorithm="brute")
    model.fit(embedding)
    indices = model.kneighbors(return_distance=False)
    # Remove self robustly instead of assuming it is always in column zero.
    result = []
    for cell, row in enumerate(indices):
        result.append(row[row != cell][:k])
    return np.asarray(result)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--dataset", default="mca_Cerebellum_62216")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--k", type=int, default=15)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "scBasset": args.data_root / "ADATA" / f"{args.dataset}_scBasset.pkl",
        "SnapATAC2": args.data_root / "ADATA" / f"{args.dataset}_snap.pkl",
        "LSI": args.data_root / "lsi_results" / "ADATA" / f"{args.dataset}_lsi.pkl",
    }
    objects = {}
    for name, path in paths.items():
        with path.open("rb") as handle:
            objects[name] = pickle.load(handle)

    reference = objects["SnapATAC2"]
    labels = reference.obs["cell_type"].astype(str).to_numpy()
    cell_ids = np.asarray(reference.obs_names)
    for name, adata in objects.items():
        if not np.array_equal(np.asarray(adata.obs_names), cell_ids):
            raise ValueError(f"Cell order differs for {name}")

    neighbor_sets = {
        "scBasset": neighbors(objects["scBasset"].obsm["projection"], args.k, "euclidean"),
        "SnapATAC2": neighbors(objects["SnapATAC2"].obsm["X_spectral"], args.k, "euclidean"),
        "LSI": neighbors(objects["LSI"].obsm["X_lsi"], args.k, "cosine"),
    }

    rows = []
    for cell in range(len(labels)):
        per_view_purity = {
            name: float(np.mean(labels[index[cell]] == labels[cell]))
            for name, index in neighbor_sets.items()
        }
        other = set(neighbor_sets["scBasset"][cell]) | set(neighbor_sets["SnapATAC2"][cell])
        unique_lsi = [index for index in neighbor_sets["LSI"][cell] if index not in other]
        rows.append(
            {
                "cell_id": cell_ids[cell],
                "cell_type": labels[cell],
                "scBasset_neighbor_purity": per_view_purity["scBasset"],
                "SnapATAC2_neighbor_purity": per_view_purity["SnapATAC2"],
                "LSI_neighbor_purity": per_view_purity["LSI"],
                "n_unique_LSI_neighbors": len(unique_lsi),
                "unique_LSI_same_type_fraction": (
                    float(np.mean(labels[unique_lsi] == labels[cell])) if unique_lsi else np.nan
                ),
                "unique_LSI_same_type_count": int(
                    np.sum(labels[unique_lsi] == labels[cell]) if unique_lsi else 0
                ),
            }
        )
    per_cell = pd.DataFrame(rows)
    per_cell.to_csv(args.output_dir / "neighbor_purity_per_cell.csv", index=False)

    grouped = (
        per_cell.groupby("cell_type", as_index=False)
        .agg(
            n_cells=("cell_id", "size"),
            scBasset_neighbor_purity=("scBasset_neighbor_purity", "mean"),
            SnapATAC2_neighbor_purity=("SnapATAC2_neighbor_purity", "mean"),
            LSI_neighbor_purity=("LSI_neighbor_purity", "mean"),
            mean_unique_LSI_neighbors=("n_unique_LSI_neighbors", "mean"),
            unique_LSI_same_type_fraction=("unique_LSI_same_type_fraction", "mean"),
            unique_LSI_same_type_count=("unique_LSI_same_type_count", "sum"),
        )
        .sort_values(["n_cells", "cell_type"], ascending=[False, True])
    )
    grouped["LSI_minus_best_other_purity"] = grouped["LSI_neighbor_purity"] - grouped[
        ["scBasset_neighbor_purity", "SnapATAC2_neighbor_purity"]
    ].max(axis=1)
    grouped.to_csv(args.output_dir / "neighbor_purity_by_cell_type.csv", index=False)

    global_summary = {
        "dataset": args.dataset,
        "n_cells": len(labels),
        "k": args.k,
        "mean_neighbor_purity": {
            name: float(per_cell[f"{name}_neighbor_purity"].mean())
            for name in ("scBasset", "SnapATAC2", "LSI")
        },
        "mean_unique_LSI_neighbors_per_cell": float(per_cell["n_unique_LSI_neighbors"].mean()),
        "unique_LSI_same_type_fraction": float(
            per_cell["unique_LSI_same_type_count"].sum()
            / per_cell["n_unique_LSI_neighbors"].sum()
        ),
    }
    with (args.output_dir / "neighbor_purity_summary.json").open("w") as handle:
        json.dump(global_summary, handle, indent=2)
    print(json.dumps(global_summary, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Reproduce the MCA Bone Marrow B-cell DPT analysis in Fig. S10 and Table S8.

Use the saved Hisemble consensus graph and the four reported annotation stages.
The root is the highest weighted-degree progenitor; sensitivity uses ten roots.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import scanpy as sc
from scipy.sparse.csgraph import connected_components
from scipy.stats import rankdata, spearmanr


DATA_ROOT = None
OUTPUT_ROOT = None

LINEAGE = {'dataset': 'mca_BoneMarrow_62216',
 'dataset_label': 'MCA Bone Marrow',
 'lineage': 'B-cell lineage',
 'stages': ['Hematopoietic progenitors',
            'Immature B cells',
            'B cells',
            'Activated B cells'],
 'stage_labels': ['Hematopoietic progenitors',
                  'Immature B cells',
                  'B cells',
                  'Activated B cells']}


def central_root(graph, root_mask: np.ndarray) -> int:
    """Select the highest weighted-degree cell within the predefined root stage."""
    degree = np.asarray(graph.sum(axis=1)).ravel()
    root_indices = np.flatnonzero(root_mask)
    return int(root_indices[np.argmax(degree[root_indices])])


def ordinal_concordance(stage_index: np.ndarray, pseudotime: np.ndarray) -> float:
    """Probability that a later-stage cell has greater pseudotime."""
    concordant = 0.0
    total = 0
    for early in np.unique(stage_index):
        for late in np.unique(stage_index):
            if late <= early:
                continue
            left = pseudotime[stage_index == early]
            right = pseudotime[stage_index == late]
            combined = np.concatenate([left, right])
            ranks = rankdata(combined, method="average")
            rank_right = ranks[len(left) :].sum()
            u_right = rank_right - len(right) * (len(right) + 1) / 2
            concordant += float(u_right)
            total += len(left) * len(right)
    return concordant / total if total else float("nan")


def run_lineage(lineage: dict) -> dict:
    path = DATA_ROOT / f"{lineage['dataset']}_Hisemble_result.h5ad"
    adata = sc.read_h5ad(path)
    stage_order = lineage["stages"]
    subset = adata[adata.obs["cell_type"].astype(str).isin(stage_order)].copy()
    graph = subset.obsp["connectivities"].tocsr()

    n_components, labels = connected_components(graph, directed=False)
    root_mask = subset.obs["cell_type"].astype(str).eq(stage_order[0]).to_numpy()
    root_components = np.unique(labels[root_mask])
    chosen_component = int(max(root_components, key=lambda value: np.sum(labels == value)))
    keep = labels == chosen_component
    subset = subset[keep].copy()
    graph = subset.obsp["connectivities"].tocsr()

    root_mask = subset.obs["cell_type"].astype(str).eq(stage_order[0]).to_numpy()
    root_index = central_root(graph, root_mask)
    subset.uns["iroot"] = root_index
    subset.uns["neighbors"] = {
        "connectivities_key": "connectivities",
        "distances_key": "distances",
        "params": {"method": "Hisemble consensus graph", "n_neighbors": 15},
    }
    sc.tl.diffmap(subset, n_comps=15)
    sc.tl.dpt(subset, n_dcs=10, n_branchings=0)

    stages = subset.obs["cell_type"].astype(str).to_numpy()
    pseudotime = subset.obs["dpt_pseudotime"].to_numpy(dtype=float)
    stage_index = np.asarray([stage_order.index(value) for value in stages], dtype=int)
    rho, p_value = spearmanr(stage_index, pseudotime)
    medians = {
        stage: float(np.median(pseudotime[stages == stage])) for stage in stage_order
    }
    pseudotime_percentile = (rankdata(pseudotime, method="average") - 1) / max(
        len(pseudotime) - 1, 1
    )
    percentile_medians = {
        stage: float(np.median(pseudotime_percentile[stages == stage]))
        for stage in stage_order
    }
    counts = {stage: int(np.sum(stages == stage)) for stage in stage_order}
    metrics = {
        "dataset": lineage["dataset"],
        "dataset_label": lineage["dataset_label"],
        "lineage": lineage["lineage"],
        "stages": stage_order,
        "stage_labels": lineage["stage_labels"],
        "n_stages": len(stage_order),
        "n_lineage_cells_total": int(np.sum(adata.obs["cell_type"].astype(str).isin(stage_order))),
        "n_cells_in_root_component": int(subset.n_obs),
        "root_component_coverage": float(subset.n_obs / np.sum(adata.obs["cell_type"].astype(str).isin(stage_order))),
        "n_induced_components": int(n_components),
        "root_stage": stage_order[0],
        "root_cell": str(subset.obs_names[root_index]),
        "root_strategy": "highest weighted degree within the predefined earliest stage",
        "spearman_rho": float(rho),
        "spearman_p_value": float(p_value),
        "ordinal_concordance": float(ordinal_concordance(stage_index, pseudotime)),
        "stage_counts": counts,
        "stage_median_pseudotime": medians,
        "stage_median_dpt_percentile": percentile_medians,
        "strictly_increasing_stage_medians": bool(
            np.all(np.diff([medians[stage] for stage in stage_order]) > 0)
        ),
        "source_h5ad": str(path),
        "graph_key": "obsp['connectivities']",
    }

    lineage_dir = OUTPUT_ROOT / lineage["dataset"]
    lineage_dir.mkdir(parents=True, exist_ok=True)
    with (lineage_dir / "metrics.json").open("w") as handle:
        json.dump(metrics, handle, indent=2)
    with (lineage_dir / "pseudotime.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["cell_id", "cell_type", "stage_index", "dpt_pseudotime"])
        for index in range(subset.n_obs):
            writer.writerow([
                subset.obs_names[index], stages[index], stage_index[index], pseudotime[index],
            ])
    return metrics


def root_sensitivity(lineage: dict, n_roots: int = 10) -> dict:
    """Repeat DPT from the top-degree cells in the predefined earliest stage."""
    path = DATA_ROOT / f"{lineage['dataset']}_Hisemble_result.h5ad"
    adata = sc.read_h5ad(path)
    stage_order = lineage["stages"]
    subset = adata[adata.obs["cell_type"].astype(str).isin(stage_order)].copy()
    graph = subset.obsp["connectivities"].tocsr()
    _, labels = connected_components(graph, directed=False)
    root_mask = subset.obs["cell_type"].astype(str).eq(stage_order[0]).to_numpy()
    root_components = np.unique(labels[root_mask])
    chosen_component = int(max(root_components, key=lambda value: np.sum(labels == value)))
    subset = subset[labels == chosen_component].copy()
    graph = subset.obsp["connectivities"].tocsr()
    root_mask = subset.obs["cell_type"].astype(str).eq(stage_order[0]).to_numpy()
    degree = np.asarray(graph.sum(axis=1)).ravel()
    root_indices = np.flatnonzero(root_mask)
    root_indices = root_indices[np.argsort(-degree[root_indices])[:n_roots]]
    stages = subset.obs["cell_type"].astype(str).to_numpy()
    stage_index = np.asarray([stage_order.index(value) for value in stages], dtype=int)
    subset.uns["neighbors"] = {
        "connectivities_key": "connectivities",
        "distances_key": "distances",
        "params": {"method": "Hisemble consensus graph", "n_neighbors": 15},
    }
    sc.tl.diffmap(subset, n_comps=15)
    rows = []
    for root_index in root_indices:
        subset.uns["iroot"] = int(root_index)
        sc.tl.dpt(subset, n_dcs=10, n_branchings=0)
        pseudotime = subset.obs["dpt_pseudotime"].to_numpy(dtype=float)
        rho, p_value = spearmanr(stage_index, pseudotime)
        rows.append(
            {
                "root_cell": str(subset.obs_names[root_index]),
                "root_weighted_degree": float(degree[root_index]),
                "spearman_rho": float(rho),
                "spearman_p_value": float(p_value),
                "ordinal_concordance": float(ordinal_concordance(stage_index, pseudotime)),
            }
        )
    output = OUTPUT_ROOT / lineage["dataset"] / "root_sensitivity.csv"
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return {
        "n_roots": len(rows),
        "root_definition": "top weighted-degree cells in the prespecified earliest stage",
        "spearman_rho_median": float(np.median([row["spearman_rho"] for row in rows])),
        "spearman_rho_range": [
            float(np.min([row["spearman_rho"] for row in rows])),
            float(np.max([row["spearman_rho"] for row in rows])),
        ],
        "ordinal_concordance_median": float(
            np.median([row["ordinal_concordance"] for row in rows])
        ),
        "ordinal_concordance_range": [
            float(np.min([row["ordinal_concordance"] for row in rows])),
            float(np.max([row["ordinal_concordance"] for row in rows])),
        ],
    }


def main() -> None:
    global DATA_ROOT, OUTPUT_ROOT
    parser = argparse.ArgumentParser(description="Numerical DPT evaluation on saved Hisemble graphs; no plotting")
    parser.add_argument("--data-root", type=Path, required=True, help="Directory of <dataset>_Hisemble_result.h5ad files")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    DATA_ROOT, OUTPUT_ROOT = args.data_root, args.output_dir
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    metrics = run_lineage(LINEAGE)
    sensitivity = root_sensitivity(LINEAGE)
    summary = dict(metrics)
    summary.update({"root_sensitivity_" + key: value for key, value in sensitivity.items()})
    with (OUTPUT_ROOT / "lineage_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary))
        writer.writeheader()
        writer.writerow(summary)
    print(json.dumps(summary, indent=2))



if __name__ == "__main__":
    main()

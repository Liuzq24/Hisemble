#!/usr/bin/env python3
"""Evaluate Scanpy diffusion pseudotime on saved Hisemble consensus graphs.

Candidate lineages and their biological order are fixed below.  Every candidate
is retained in the output.  The representative result is selected among
lineages with at least three annotated stages by the cell-level ordinal
concordance between stage order and DPT pseudotime, with graph coverage used as
a secondary criterion.
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

CANDIDATES = [
    {
        "dataset": "Chen-2019-ATAC",
        "dataset_label": "Mouse Cortex",
        "lineage": "Oligodendrocyte lineage",
        "stages": ["OPC", "OliI", "OliM"],
        "stage_labels": ["OPC", "Immature oligodendrocytes", "Mature oligodendrocytes"],
    },
    {
        "dataset": "mca_BoneMarrow_62216",
        "dataset_label": "MCA Bone Marrow",
        "lineage": "B-cell lineage",
        "stages": [
            "Hematopoietic progenitors",
            "Immature B cells",
            "B cells",
            "Activated B cells",
        ],
        "stage_labels": [
            "Hematopoietic progenitors",
            "Immature B cells",
            "B cells",
            "Activated B cells",
        ],
    },
    {
        "dataset": "TME-All",
        "dataset_label": "Human Tumor",
        "lineage": "CD8 T-cell state progression",
        "stages": ["Naive CD8 T", "Memory CD8 T", "Effector CD8 T", "CD8 TEx"],
        "stage_labels": ["Naive CD8 T", "Memory CD8 T", "Effector CD8 T", "Exhausted CD8 T"],
    },
    {
        "dataset": "FCA_fetal_33184180_muscle",
        "dataset_label": "FCA Muscle",
        "lineage": "Skeletal-muscle lineage",
        "stages": ["Satellite cells", "Skeletal muscle cells?", "Skeletal muscle cells"],
        "stage_labels": ["Satellite cells", "Intermediate skeletal muscle", "Skeletal muscle cells"],
    },
    {
        "dataset": "mca_Lung1_62216",
        "dataset_label": "MCA Lung A",
        "lineage": "Alveolar epithelial lineage",
        "stages": ["Type II pneumocytes", "Type I pneumocytes"],
        "stage_labels": ["Type II pneumocytes", "Type I pneumocytes"],
    },
    {
        "dataset": "mca_Lung2_62216",
        "dataset_label": "MCA Lung B",
        "lineage": "Alveolar epithelial lineage",
        "stages": ["Type II pneumocytes", "Type I pneumocytes"],
        "stage_labels": ["Type II pneumocytes", "Type I pneumocytes"],
    },
]


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


def run_candidate(candidate: dict) -> dict:
    path = DATA_ROOT / f"{candidate['dataset']}_Hisemble_result.h5ad"
    adata = sc.read_h5ad(path)
    stage_order = candidate["stages"]
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
        "dataset": candidate["dataset"],
        "dataset_label": candidate["dataset_label"],
        "lineage": candidate["lineage"],
        "stages": stage_order,
        "stage_labels": candidate["stage_labels"],
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

    candidate_dir = OUTPUT_ROOT / candidate["dataset"]
    candidate_dir.mkdir(parents=True, exist_ok=True)
    with (candidate_dir / "metrics.json").open("w") as handle:
        json.dump(metrics, handle, indent=2)
    with (candidate_dir / "pseudotime.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["cell_id", "cell_type", "stage_index", "dpt_pseudotime"])
        for index in range(subset.n_obs):
            writer.writerow([
                subset.obs_names[index], stages[index], stage_index[index], pseudotime[index],
            ])
    return metrics


def root_sensitivity(candidate: dict, n_roots: int = 10) -> dict:
    """Repeat DPT from the top-degree cells in the predefined earliest stage."""
    path = DATA_ROOT / f"{candidate['dataset']}_Hisemble_result.h5ad"
    adata = sc.read_h5ad(path)
    stage_order = candidate["stages"]
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
    output = OUTPUT_ROOT / candidate["dataset"] / "root_sensitivity.csv"
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
    results = [run_candidate(candidate) for candidate in CANDIDATES]
    eligible = [result for result in results if result["n_stages"] >= 3]
    representative = max(
        eligible,
        key=lambda result: (result["ordinal_concordance"], result["root_component_coverage"]),
    )
    representative_candidate = next(
        candidate for candidate in CANDIDATES
        if candidate["dataset"] == representative["dataset"]
    )
    sensitivity = root_sensitivity(representative_candidate)
    with (OUTPUT_ROOT / "candidate_summary.csv").open("w", newline="") as handle:
        fields = [
            "dataset", "dataset_label", "lineage", "n_stages", "n_lineage_cells_total",
            "n_cells_in_root_component", "root_component_coverage", "n_induced_components",
            "spearman_rho", "spearman_p_value", "ordinal_concordance",
            "strictly_increasing_stage_medians", "root_stage", "root_cell",
            "stages", "stage_counts", "stage_median_pseudotime",
            "stage_median_dpt_percentile",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for result in results:
            writer.writerow({field: result[field] for field in fields})
    with (OUTPUT_ROOT / "analysis_summary.json").open("w") as handle:
        json.dump(
            {
                "selection_rule": (
                    "Among predefined lineages with at least three annotated stages, select the "
                    "highest cell-level ordinal concordance; use graph coverage as a tiebreaker."
                ),
                "representative": representative,
                "representative_root_sensitivity": sensitivity,
                "all_candidates": results,
            },
            handle,
            indent=2,
        )
    print(json.dumps({"representative": representative, "all_candidates": results}, indent=2))


if __name__ == "__main__":
    main()

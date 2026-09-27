#!/usr/bin/env python3
"""Validated input and interoperability utilities for Hisemble.

These utilities deliberately do not impute non-finite values or silently remove
cells. Every view must refer to the same cells in the same order; changing rows
independently would invalidate cross-view fusion.
"""

from __future__ import annotations

import csv
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class ViewValidationReport:
    name: str
    n_cells: int
    n_features: int
    constant_features: int
    flagged_outlier_rows: tuple[int, ...]
    maximum_robust_norm_z: float


def _robust_row_norm_outliers(
    embedding: np.ndarray, threshold: float
) -> tuple[tuple[int, ...], float]:
    norms = np.linalg.norm(embedding, axis=1)
    median = float(np.median(norms))
    mad = float(np.median(np.abs(norms - median)))
    if mad <= np.finfo(float).eps:
        return (), 0.0
    robust_z = 0.6744897501960817 * np.abs(norms - median) / mad
    flagged = tuple(np.flatnonzero(robust_z > threshold).astype(int).tolist())
    return flagged, float(robust_z.max(initial=0.0))


def validate_embeddings(
    embeddings: Mapping[str, np.ndarray],
    *,
    cell_ids_by_view: Mapping[str, Sequence[str]] | None = None,
    minimum_views: int = 2,
    outlier_threshold: float = 12.0,
    outlier_action: str = "warn",
) -> dict[str, ViewValidationReport]:
    """Validate a collection of cell-by-feature embedding matrices.

    Parameters
    ----------
    embeddings
        Mapping from view name to a two-dimensional numeric array.
    cell_ids_by_view
        Optional mapping of view name to cell identifiers. If supplied, every
        view must contain exactly the same identifiers in the same order.
    minimum_views
        Minimum number of views required for cross-view fusion.
    outlier_threshold
        Robust z-score threshold applied to row norms for diagnostics.
    outlier_action
        ``"warn"`` (default), ``"raise"``, or ``"ignore"``. Outliers are never
        automatically clipped, imputed, or removed.
    """

    if len(embeddings) < minimum_views:
        raise ValueError(
            f"Hisemble requires at least {minimum_views} views; received {len(embeddings)}"
        )
    if outlier_action not in {"warn", "raise", "ignore"}:
        raise ValueError("outlier_action must be 'warn', 'raise', or 'ignore'")

    names = list(embeddings)
    expected_cells: int | None = None
    reference_ids: np.ndarray | None = None
    reports: dict[str, ViewValidationReport] = {}

    if cell_ids_by_view is not None:
        missing = set(names) - set(cell_ids_by_view)
        extra = set(cell_ids_by_view) - set(names)
        if missing or extra:
            raise ValueError(
                f"cell_ids_by_view keys do not match embeddings; missing={sorted(missing)}, "
                f"extra={sorted(extra)}"
            )

    for name in names:
        array = np.asarray(embeddings[name])
        if array.ndim != 2:
            raise ValueError(f"View {name!r} must be two-dimensional; shape={array.shape}")
        if array.shape[0] < 2 or array.shape[1] < 1:
            raise ValueError(f"View {name!r} has an invalid shape: {array.shape}")
        if not np.issubdtype(array.dtype, np.number):
            raise TypeError(f"View {name!r} is not numeric; dtype={array.dtype}")

        if expected_cells is None:
            expected_cells = array.shape[0]
        elif array.shape[0] != expected_cells:
            raise ValueError(
                f"Cell-count mismatch: view {name!r} has {array.shape[0]} rows; "
                f"expected {expected_cells}"
            )

        finite_mask = np.isfinite(array)
        if not finite_mask.all():
            bad_rows, bad_columns = np.where(~finite_mask)
            preview = list(zip(bad_rows[:10].tolist(), bad_columns[:10].tolist()))
            raise ValueError(
                f"View {name!r} contains {len(bad_rows)} NaN/Inf values; "
                f"first affected (row, column) positions: {preview}. "
                "Correct or explicitly impute the upstream representation before fusion."
            )

        feature_variances = np.var(array.astype(float, copy=False), axis=0)
        constant_features = int(np.count_nonzero(feature_variances <= np.finfo(float).eps))
        if constant_features == array.shape[1]:
            raise ValueError(f"View {name!r} is degenerate: all features are constant")

        flagged, maximum_z = _robust_row_norm_outliers(
            array.astype(float, copy=False), outlier_threshold
        )
        if flagged:
            message = (
                f"View {name!r} contains {len(flagged)} rows with robust row-norm z-score "
                f"> {outlier_threshold:g}; first rows: {list(flagged[:10])}. "
                "Hisemble did not modify or remove these cells."
            )
            if outlier_action == "raise":
                raise ValueError(message)
            if outlier_action == "warn":
                warnings.warn(message, RuntimeWarning, stacklevel=2)

        if cell_ids_by_view is not None:
            identifiers = np.asarray(cell_ids_by_view[name], dtype=str)
            if len(identifiers) != array.shape[0]:
                raise ValueError(
                    f"Cell-ID count for {name!r} is {len(identifiers)}; "
                    f"embedding has {array.shape[0]} rows"
                )
            if len(np.unique(identifiers)) != len(identifiers):
                raise ValueError(f"Duplicate cell identifiers in view {name!r}")
            if reference_ids is None:
                reference_ids = identifiers
            elif not np.array_equal(identifiers, reference_ids):
                mismatch = int(np.flatnonzero(identifiers != reference_ids)[0])
                raise ValueError(
                    f"Cell order differs in view {name!r} at row {mismatch}: "
                    f"{identifiers[mismatch]!r} != {reference_ids[mismatch]!r}"
                )

        reports[name] = ViewValidationReport(
            name=name,
            n_cells=array.shape[0],
            n_features=array.shape[1],
            constant_features=constant_features,
            flagged_outlier_rows=flagged,
            maximum_robust_norm_z=maximum_z,
        )

    return reports


def validation_report_as_dict(
    reports: Mapping[str, ViewValidationReport],
) -> dict[str, dict[str, object]]:
    return {name: asdict(report) for name, report in reports.items()}


def store_consensus_in_anndata(
    adata,
    consensus,
    *,
    clusters: Sequence[object] | None = None,
    graph_key: str = "hisemble_connectivities",
    cluster_key: str = "hisemble_cluster",
):
    """Store a Hisemble consensus graph and optional clusters in AnnData.

    The input AnnData object is modified in place and also returned.
    """

    from scipy.sparse import csr_matrix

    graph = csr_matrix(consensus)
    if graph.shape != (adata.n_obs, adata.n_obs):
        raise ValueError(
            f"Consensus shape {graph.shape} does not match AnnData observations "
            f"({adata.n_obs}, {adata.n_obs})"
        )
    if not np.isfinite(graph.data).all():
        raise ValueError("Consensus graph contains NaN or infinite edge weights")
    adata.obsp[graph_key] = graph

    if clusters is not None:
        values = np.asarray(clusters)
        if values.ndim != 1 or len(values) != adata.n_obs:
            raise ValueError(
                f"Cluster labels must have length {adata.n_obs}; shape={values.shape}"
            )
        adata.obs[cluster_key] = values.astype(str)

    adata.uns.setdefault("hisemble", {})
    adata.uns["hisemble"].update(
        {
            "graph_key": graph_key,
            "cluster_key": cluster_key if clusters is not None else None,
            "output_type": "weighted cell-cell consensus graph",
        }
    )
    return adata


def export_consensus_for_seurat(
    consensus,
    cell_ids: Sequence[str],
    output_dir: str | Path,
    *,
    clusters: Sequence[object] | None = None,
) -> dict[str, Path]:
    """Export a sparse consensus graph and metadata for R/Seurat workflows."""

    from scipy.io import mmwrite
    from scipy.sparse import csr_matrix

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    graph = csr_matrix(consensus)
    identifiers = np.asarray(cell_ids, dtype=str)
    if graph.shape != (len(identifiers), len(identifiers)):
        raise ValueError(
            f"Consensus shape {graph.shape} does not match {len(identifiers)} cell IDs"
        )
    if len(np.unique(identifiers)) != len(identifiers):
        raise ValueError("Cell identifiers must be unique")
    if not np.isfinite(graph.data).all():
        raise ValueError("Consensus graph contains NaN or infinite edge weights")

    matrix_path = output / "hisemble_consensus.mtx"
    cells_path = output / "hisemble_cells.tsv"
    metadata_path = output / "hisemble_metadata.csv"
    readme_path = output / "README_Seurat_import.txt"
    mmwrite(matrix_path, graph)

    with cells_path.open("w", newline="") as handle:
        for identifier in identifiers:
            handle.write(f"{identifier}\n")

    cluster_values = None if clusters is None else np.asarray(clusters)
    if cluster_values is not None and (
        cluster_values.ndim != 1 or len(cluster_values) != len(identifiers)
    ):
        raise ValueError(
            f"Cluster labels must have length {len(identifiers)}; "
            f"shape={cluster_values.shape}"
        )
    with metadata_path.open("w", newline="") as handle:
        fieldnames = ["cell_id"] + (["hisemble_cluster"] if clusters is not None else [])
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for index, identifier in enumerate(identifiers):
            row = {"cell_id": identifier}
            if cluster_values is not None:
                row["hisemble_cluster"] = str(cluster_values[index])
            writer.writerow(row)

    readme_path.write_text(
        "Hisemble exports a weighted cell-cell graph, not a feature matrix.\n"
        "R example:\n"
        "  library(Matrix)\n"
        "  G <- readMM('hisemble_consensus.mtx')\n"
        "  cells <- read.delim('hisemble_cells.tsv', header=FALSE)$V1\n"
        "  rownames(G) <- cells; colnames(G) <- cells\n"
        "  meta <- read.csv('hisemble_metadata.csv', row.names=1)\n"
        "Verify that cells exactly match colnames(seurat_object) before attaching results.\n",
        encoding="utf-8",
    )
    return {
        "matrix": matrix_path,
        "cells": cells_path,
        "metadata": metadata_path,
        "readme": readme_path,
    }

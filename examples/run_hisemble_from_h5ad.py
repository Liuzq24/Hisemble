#!/usr/bin/env python3
"""Run Hisemble on embeddings already stored in one AnnData .h5ad file."""

from __future__ import annotations

import argparse
from pathlib import Path

import anndata as ad

from hisemble import run_hisemble


def comma_list(value: str) -> list[str]:
    items = [item.strip() for item in value.split(",") if item.strip()]
    if not items:
        raise argparse.ArgumentTypeError("provide at least one comma-separated value")
    return items


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fuse aligned embeddings in AnnData.obsm and write a new h5ad file."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--keys", type=comma_list, required=True)
    parser.add_argument("--metrics", type=comma_list, required=True)
    parser.add_argument("--k", type=int, default=15)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--resolution", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cluster-key", default="hisemble_clusters")
    args = parser.parse_args()
    if len(args.keys) != len(args.metrics):
        parser.error("--keys and --metrics must contain the same number of items")

    adata = ad.read_h5ad(args.input)
    missing = [key for key in args.keys if key not in adata.obsm]
    if missing:
        parser.error(f"embedding keys missing from AnnData.obsm: {missing}")
    run_hisemble(
        adata,
        input_keys=args.keys,
        metrics=args.metrics,
        k_neighbors=args.k,
        n_iterations=args.iterations,
        resolution=args.resolution,
        key_added=args.cluster_key,
        random_seed=args.seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(args.output)
    print(f"wrote={args.output}")
    print(f"cluster_key={args.cluster_key}")
    print("graph_key=connectivities")


if __name__ == "__main__":
    main()

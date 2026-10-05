#!/usr/bin/env python3
"""Run an end-to-end Hisemble smoke test on deterministic synthetic embeddings."""

from __future__ import annotations

import argparse
from pathlib import Path

import anndata as ad
import numpy as np
from sklearn.datasets import make_blobs

from hisemble import run_hisemble


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("reviewer_output"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    latent, labels = make_blobs(
        n_samples=180,
        centers=4,
        n_features=8,
        cluster_std=0.75,
        random_state=42,
    )
    rng = np.random.default_rng(42)
    view_a = latent + rng.normal(0, 0.20, size=latent.shape)
    projection = rng.normal(size=(latent.shape[1], 12))
    view_b = latent @ projection + rng.normal(0, 0.35, size=(len(latent), 12))
    cell_ids = np.asarray([f"cell_{index:03d}" for index in range(len(latent))])

    adata = ad.AnnData(
        X=np.zeros((len(latent), 1), dtype=np.float32),
        obs={"reference_group": labels.astype(str)},
    )
    adata.obs_names = cell_ids
    adata.obsm["view_a"] = view_a.astype(np.float32)
    adata.obsm["view_b"] = view_b.astype(np.float32)

    run_hisemble(
        adata,
        input_keys=["view_a", "view_b"],
        metrics=["gaussian", "gaussian"],
        k_neighbors=15,
        n_iterations=20,
        resolution=0.5,
        random_seed=42,
    )

    h5ad_path = args.output_dir / "synthetic_hisemble.h5ad"
    adata.write_h5ad(h5ad_path)
    print(f"cells={adata.n_obs}")
    print(f"clusters={adata.obs['hisemble_clusters'].nunique()}")
    print(f"graph_shape={adata.obsp['connectivities'].shape}")
    print(f"graph_nnz={adata.obsp['connectivities'].nnz}")
    print(f"wrote={h5ad_path}")


if __name__ == "__main__":
    main()

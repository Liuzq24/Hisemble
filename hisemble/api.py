import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData
from typing import List, Union

from .graph import build_knn_graph_chunked_cpu
from .core import hisemble_fusion
from .cluster import matrix_to_igraph, perform_leiden_clustering

def run_hisemble(
    adata: AnnData, 
    input_keys: List[str],
    metrics: Union[str, List[str]] = 'gaussian',
    k_neighbors: int = 15, 
    n_iterations: int = 20, 
    resolution: float = 1.0, 
    key_added: str = 'hisemble_clusters',
    random_seed: int = 42,
    verbose: bool = True
) -> None:
    """
    Main user interface for Hisemble.
    Executes graph construction, fusion, and Leiden clustering directly on AnnData.
    """
    
    # 1. distance metrics (e.g., 'cosine' is recommended for LSI)
    if isinstance(metrics, str):
        metrics = [metrics] * len(input_keys)
    elif len(metrics) != len(input_keys):
        raise ValueError("metrics list length must match input_keys list length.")

    # 2. Build local KNN affinity graphs for each view
    W_list = []
    for key, metric in zip(input_keys, metrics):
        if key not in adata.obsm.keys():
            raise KeyError(f"Embedding '{key}' not found in adata.obsm.")
        
        if verbose:
            print(f"Building KNN graph for {key} using {metric} metric...")
            
        emb = adata.obsm[key]
        W = build_knn_graph_chunked_cpu(emb, k_neighbors=k_neighbors, metric=metric, random_seed=random_seed)
        W_list.append(W)

    # 3. Run Hisemble fusion
    if verbose:
        print("Running Hisemble sparse cross-diffusion...")
    fused_network = hisemble_fusion(W_list, k_neighbors=k_neighbors, n_iterations=n_iterations, random_seed=random_seed, verbose=verbose)

    # 4. Perform Leiden clustering
    if verbose:
        print(f"Running Leiden clustering with resolution {resolution}...")
    net = matrix_to_igraph(fused_network)
    clusters = perform_leiden_clustering(net, resolution=resolution, seed=random_seed)
    
    # 5. Integrate seamlessly with the Scanpy ecosystem
    adata.obs[key_added] = pd.Categorical(clusters)
    
    fused_network = fused_network.astype(np.float32)
    adata.obsp['connectivities'] = fused_network
    
    distances = fused_network.copy()
    distances.data = 1.0 - distances.data
    distances.eliminate_zeros()
    adata.obsp['distances'] = distances

    adata.uns['neighbors'] = {}
    adata.uns['neighbors']['params'] = {'method': 'custom', 'custom_graph': True}
    adata.uns['neighbors']['connectivities_key'] = 'connectivities' 
    adata.uns['neighbors']['distances_key'] = 'distances'  
    adata.uns['neighbors']['params']['n_neighbors'] = k_neighbors

    if verbose:
        print(f"Finished! Fused graph saved to `.obsp['connectivities']`.")
        print(f"Cluster labels saved to `.obs['{key_added}']`.")
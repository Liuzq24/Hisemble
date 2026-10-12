import pandas as pd
import numpy as np
import sklearn
from sklearn import metrics
import scipy
import scipy.sparse as sp
from scipy.sparse import issparse, csr_matrix
import scanpy as sc
import anndata
import pickle
import matplotlib as mpl
from matplotlib import pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score, normalized_mutual_info_score, homogeneity_score
import igraph as ig
import leidenalg
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
import time
import psutil
import threading
import argparse
import os
from joblib import Parallel, delayed
import fcntl
warnings.filterwarnings("ignore")
# Resource monitor.
class MemoryMonitor: 
    def __init__(self, interval=0.1): 
        self.interval = interval 
        self.peak_memory = 0
        self.peak_mb = 0  
        self.running = False
        self.process = psutil.Process(os.getpid()) 

    def _monitor(self):
        while self.running:
            try:
                mem = self.process.memory_info().rss
                # Include memory used by joblib child processes.
                for child in self.process.children(recursive=True):
                    try:
                        mem += child.memory_info().rss
                    except:
                        pass
                self.peak_memory = max(self.peak_memory, mem)
            except:
                pass
            time.sleep(self.interval)

    def __enter__(self):
        self.running = True
        self.monitor_thread = threading.Thread(target=self._monitor, daemon=True)
        self.monitor_thread.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.running = False
        self.monitor_thread.join()
        self.peak_mb = self.peak_memory / (1024 ** 2)
        print(f"Algorithm peak memory usage: {self.peak_mb:.2f} MB")

# Helper: convert the SC3 consensus matrix to a sparse UMAP graph.
def sparsify_matrix_from_dense(dense_matrix, k_neighbors: int) -> csr_matrix:
    """Convert a dense consensus matrix to a sparse matrix by retaining each row's largest k values."""
    n_samples = dense_matrix.shape[0]
    # Use argsort to select the largest k entries in each row.
    # Larger consensus values indicate greater similarity.
    indices = np.argsort(-dense_matrix, axis=1)[:, :k_neighbors]
    
    row_indices = np.repeat(np.arange(n_samples), k_neighbors)
    col_indices = indices.flatten()
    data = dense_matrix[np.arange(n_samples)[:, None], indices].flatten()
    
    sparse_mat = csr_matrix((data, (row_indices, col_indices)), shape=(n_samples, n_samples))
    return (sparse_mat + sparse_mat.T) / 2

# SC3 consensus baseline.
def run_one_kmeans(data, k, seed):
    # n_init=10 limits the number of k-means initializations.
    # Use the Elkan k-means algorithm.
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=seed, algorithm='elkan')
    return kmeans.fit_predict(data)

def run_sc3_workflow(embeddings, n_clusters, n_runs=20, k_neighbors_for_graph=15):
    N = embeddings[0].shape[0]
    print(f"  [Debug] Cells N={N}; target clusters k={n_clusters}")

    # 1. K-Means
    print(f"  - Running k-means (runs per view={n_runs})...")
    tasks = []
    for i, emb in enumerate(embeddings):
        # Check embedding dimensions.
        if emb.shape[0] != N:
            raise ValueError(f"Embedding {i} has incompatible dimensions: {emb.shape}")
        for r in range(n_runs):
            tasks.append((emb, r + 42))
    
    results = Parallel(n_jobs=-1)(delayed(run_one_kmeans)(emb, n_clusters, seed) for emb, seed in tasks)
    label_matrix = np.array(results).T 
    
    # 2. Consensus Matrix
    print("  - Constructing the consensus matrix...")
    consensus_matrix = np.zeros((N, N), dtype=np.float32)
    n_total_runs = label_matrix.shape[1]
    
    for i in range(n_total_runs):
        lbl = label_matrix[:, i]
        mat = (lbl[:, None] == lbl[None, :])
        consensus_matrix += mat
    
    consensus_matrix /= n_total_runs
    
    # Check the range of the consensus matrix.
    print(f"  [Debug] Consensus range: min={consensus_matrix.min():.4f}, max={consensus_matrix.max():.4f}, mean={consensus_matrix.mean():.4f}")
    if consensus_matrix.mean() < 0.001 or consensus_matrix.mean() > 0.999:
        print("  Warning: the consensus matrix is nearly all zero or all one; clustering may fail.")

    # 3. Hierarchy
    print("  - Running hierarchical clustering...")
    # Keep the distance matrix within [0, 1].
    dist_mat = 1.0 - consensus_matrix
    dist_mat = np.clip(dist_mat, 0, 1) 
    
    # Convert the square distance matrix to the condensed vector required by linkage.
    dist_vec = squareform(dist_mat, checks=False)
    
    Z = linkage(dist_vec, method='average')
    labels = fcluster(Z, t=n_clusters, criterion='maxclust')
    
    print(f"  [Debug] Number of unique cluster labels: {len(np.unique(labels))}")
    print(f"  [Debug] First 20 labels: {labels[:20]}")

    # 4. Graph for UMAP
    print("  - Constructing a sparse graph for visualization...")
    sparse_graph = sparsify_matrix_from_dense(consensus_matrix, k_neighbors=k_neighbors_for_graph)
    
    return labels, sparse_graph


# Main execution block.
def main():
    # 1. Parse arguments.
    parser = argparse.ArgumentParser(description="Run the SC3 consensus clustering baseline.")
    parser.add_argument('--datanames', type=str, nargs='+', required=False, 
                        help="Dataset identifiers to process.")
    args = parser.parse_args()
    
    # Fixed hyperparameters.
    N_RUNS = 20        # Number of k-means runs per view.
    K_NEIGHBORS = 15   # Used only to construct the sparse graph for UMAP.
    RANDOM_SEED = 42
    
    np.random.seed(RANDOM_SEED)
    os.environ['PYTHONHASHSEED'] = str(RANDOM_SEED)

    # Predefined datasets.
    all_datanames_predefined = [
    'Chen-2019-ATAC',
    'FCA_fetal_33184180_cerebellum',
    'FCA_fetal_33184180_intestine',
    'FCA_fetal_33184180_stomach',
    'mca_BoneMarrow_62216',
    'mca_Cerebellum_62216',
    'mca_HeartA_62816',
    'mca_Kidney_62016',
    'mca_LargeIntestineA_62816',
    'mca_LargeIntestineB_62816',
    'mca_Liver_62016',
    'mca_Lung1_62216',
    'mca_SmallIntestine_62816',
    'mca_Testes_62016',
    'mca_WholeBrainA_62216',
    'mca_WholeBrainA_62816'
    ]

    datanames_to_process = args.datanames if args.datanames else all_datanames_predefined

    # Manage paths.
    # Use a baseline-specific output directory to preserve Hisemble results.
    base_output_dir = '__HISEMBLE_RESULTS_ROOT__/SC3_final/'
    os.makedirs(base_output_dir, exist_ok=True)
    save_path = os.path.join(base_output_dir, f'sc3_results.csv')

    save_folder_umap = os.path.join(base_output_dir, 'UMAP_plots')
    os.makedirs(save_folder_umap, exist_ok=True)
    
    # Process the selected datasets.
    for dataname in datanames_to_process:
        print(f"\n{'='*20} Processing: {dataname} {'='*20}")
        try:
            # Load data and verify cell alignment.
            path_basset = f'__HISEMBLE_DATA_ROOT__/ADATA/{dataname}_scBasset.pkl'
            path_snap = f'__HISEMBLE_DATA_ROOT__/ADATA/{dataname}_snap.pkl'
            path_lsi = f'__HISEMBLE_DATA_ROOT__/lsi_results/ADATA/{dataname}_lsi.pkl'

            with open(path_basset, 'rb') as f: ad_scBasset = pickle.load(f)
            with open(path_snap, 'rb') as f: ad_snap = pickle.load(f)
            with open(path_lsi, 'rb') as f: ad_lsi = pickle.load(f)

            print(f"  > [Check] scBasset shape: {ad_scBasset.shape}")
            print(f"  > [Check] snap shape:     {ad_snap.shape}")
            print(f"  > [Check] lsi shape:      {ad_lsi.shape}")

            # Check that cell identifiers and their order match exactly.
            if not (ad_scBasset.obs_names.equals(ad_snap.obs_names) and ad_snap.obs_names.equals(ad_lsi.obs_names)):
                print("  Error: cell identifiers or their order differ among the three inputs.")
                print("  Misaligned cells invalidate the consensus calculation. Attempting automatic alignment...")
                
                # Align the intersection of the cell identifiers.
                common_cells = ad_scBasset.obs_names.intersection(ad_snap.obs_names).intersection(ad_lsi.obs_names)
                print(f"  Cells after alignment: {len(common_cells)}")
                ad_scBasset = ad_scBasset[common_cells].copy()
                ad_snap = ad_snap[common_cells].copy()
                ad_lsi = ad_lsi[common_cells].copy()
            else:
                print("  Cell-alignment check passed.")

            true_labels = ad_snap.obs.cell_type.to_numpy()
            true_cell_types = ad_snap.obs['cell_type'].nunique()
            
            embeddings = [
                ad_scBasset.obsm['projection'],
                ad_snap.obsm['X_spectral'],
                ad_lsi.obsm['X_lsi']
            ]

            # Run SC3.
            with MemoryMonitor() as monitor:
                start_time = time.time()
                clusters, consensus_graph = run_sc3_workflow(
                    embeddings, true_cell_types, N_RUNS, K_NEIGHBORS
                )
                end_time = time.time()

            # Construct an AnnData object for plotting.
            # Reload an AnnData template to retain metadata.
            # Apply the same cell subset used during alignment.
            adata_temp = sc.read_h5ad('__HISEMBLE_DATA_ROOT__/0.01_h5ad/%s_processed.h5ad'%dataname)
            adata = adata_temp[ad_snap.obs_names].copy() # Ensure that the plotting object contains the same cells as the analysis.

            # Store the SC3 graph.
            consensus_graph = consensus_graph.astype(np.float32)
            adata.obsp['connectivities'] = consensus_graph
            
            # Construct graph distances.
            distances = consensus_graph.copy()
            distances.data = 1.0 - distances.data
            distances.data[distances.data < 0] = 0
            adata.obsp['distances'] = distances

            # Provide the neighbor-graph metadata expected by Scanpy.
            adata.uns['neighbors'] = {
                'params': {
                    'n_neighbors': K_NEIGHBORS, 
                    'method': 'umap', # Configure Scanpy neighbor-graph metadata.
                    'metric': 'precomputed'
                },
                'connectivities_key': 'connectivities',
                'distances_key': 'distances'
            }

            # Run UMAP on the configured graph.
            print("  - Computing UMAP...")
            sc.tl.umap(adata, min_dist=0.5, spread=1.0, random_state=RANDOM_SEED)

            adata.obs['SC3'] = clusters.astype(str) # Convert labels to strings for plotting.
            
            # Save the figure.
            sc.pl.umap(adata, color=['cell_type'], show=False, title=f'{dataname} True Labels')
            plt.savefig(os.path.join(save_folder_umap, f'{dataname}_True.pdf'), bbox_inches='tight')
            plt.close()

            sc.pl.umap(adata, color=['SC3'], show=False, title=f'{dataname} SC3 Labels')
            plt.savefig(os.path.join(save_folder_umap, f'{dataname}_SC3.pdf'), bbox_inches='tight')
            plt.close()

            ARI = adjusted_rand_score(true_labels, clusters)
            AMI = adjusted_mutual_info_score(true_labels, clusters)
            NMI = normalized_mutual_info_score(true_labels, clusters)
            HOM = homogeneity_score(true_labels, clusters)
            print(f"  Results: ARI={ARI:.4f}, NMI={NMI:.4f}, time={end_time-start_time:.2f} s")

            output_columns = ['dataname', 'n_runs', 'ARI', 'AMI', 'NMI', 'HOM', 'Time_s', 'Mem_MB']
            
            # 2. Construct the result dictionary.
            res_data = {
                'dataname': dataname, 
                'n_runs': N_RUNS,   # Include the dataset identifier.
                'ARI': ARI, 
                'AMI': AMI,
                'NMI': NMI,
                'HOM': HOM,
                'Time_s': end_time - start_time, 
                'Mem_MB': monitor.peak_mb
            }
            
            res_df = pd.DataFrame([res_data])
            # 3. Enforce a consistent column order.
            res_df = res_df[output_columns]

            # 4. Lock the output file to prevent concurrent writes from interleaving.
            with open(save_path, 'a') as f:
                fcntl.flock(f, fcntl.LOCK_EX) # Acquire the file lock.
                try:
                    # Write the header only if the file is empty.
                    is_empty = os.path.getsize(save_path) == 0
                    res_df.to_csv(f, header=is_empty, index=False)
                finally:
                    fcntl.flock(f, fcntl.LOCK_UN) # Release the file lock.
            
            # Save the h5ad file.
            adata.write(os.path.join(base_output_dir, f'{dataname}_result.h5ad'), compression='gzip')

        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()

if __name__ == '__main__':
    main()
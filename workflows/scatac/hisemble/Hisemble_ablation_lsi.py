import os
import pandas as pd
import numpy as np
import sklearn
from sklearn import metrics
import scipy
import scipy.sparse as sp
from scipy.sparse import issparse
import scanpy as sc
import anndata
import episcanpy.api as epi
import matplotlib as mpl
from matplotlib import pyplot as plt
import seaborn as sns
import pickle
import scib
import anndata as ad
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings("ignore")
import tensorflow as tf
from utils import cluster_evaluation, overcorrection_score # Import shared utilities.
from sklearn.metrics.pairwise import rbf_kernel, euclidean_distances
from sklearn.metrics import adjusted_rand_score,adjusted_mutual_info_score,normalized_mutual_info_score, homogeneity_score
from scipy.spatial.distance import pdist, squareform
import igraph as ig
import leidenalg
from sklearn.metrics import pairwise_distances
from scipy.sparse import csr_matrix, identity as sparse_identity
from sklearn.neighbors import NearestNeighbors
import time
import psutil
import threading
import argparse

class MemoryMonitor: 
    def __init__(self, interval=0.1): 
        self.interval = interval 
        self.peak_memory = 0
        self.peak_mb = 0  
        self.running = False
        self.process = psutil.Process(os.getpid()) 

    def _monitor(self):
        while self.running:
            mem_info = self.process.memory_info()
            self.peak_memory = max(self.peak_memory, mem_info.rss) 
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
        print(f"Hisemble peak memory usage: {self.peak_mb:.2f} MB")

def find_best_matrix(matrix, true_cell_types, low=0.1, high=5.0, tolerance=0.01, seed=42):
    best_resolution = None
    clusters = None
    low_n_clusters = len(remove_duplicates(perform_leiden_clustering(matrix, resolution=low, seed=seed)))
    high_n_clusters = len(remove_duplicates(perform_leiden_clustering(matrix, resolution=high, seed=seed)))

    if high_n_clusters <= true_cell_types:
        best_resolution = high
    elif low_n_clusters >= true_cell_types:
        best_resolution = low
    else:
        while high - low > tolerance:
            mid = (low + high) / 2
            current_clusters = perform_leiden_clustering(matrix, resolution=mid, seed=seed)
            num_clusters = len(remove_duplicates(current_clusters))
            if num_clusters < true_cell_types:
                low = mid
            elif num_clusters > true_cell_types:
                high = mid
            else:
                best_resolution = mid
                break
        if best_resolution is None:
            best_resolution = mid
            
    final_clusters = perform_leiden_clustering(matrix, resolution=best_resolution, seed=seed)
    print(f"Final resolution: {best_resolution:.4f}; clusters: {len(remove_duplicates(final_clusters))}.")
    return final_clusters

def perform_leiden_clustering(net, resolution, seed=42):
    partition = leidenalg.find_partition(net, leidenalg.RBConfigurationVertexPartition, 
                                         weights=net.es['weight'], 
                                         resolution_parameter=resolution,
                                         seed=seed)
    clusters = partition.membership
    return clusters

def remove_duplicates(lst):  
    return list(np.unique(lst))
    
def build_knn_graph_chunked_cpu(embedding: np.ndarray, k_neighbors: int, metric: str, chunk_size: int = 2048, random_seed=42) -> csr_matrix:
    n_samples = embedding.shape[0]
    all_rows, all_cols, all_data = [], [], []
    gamma = None
    if metric == 'gaussian':
        variance = embedding.var()
        if variance == 0:
            variance += 1e-12
        gamma = 1.0 / (embedding.shape[1] * variance)

    print(f"Building the k-NN graph in chunks (chunk_size={chunk_size}, metric={metric})...")
    for i in range(0, n_samples, chunk_size):
        start_idx = i
        end_idx = min(i + chunk_size, n_samples)
        chunk_embedding = embedding[start_idx:end_idx]
        if metric == 'gaussian':
            sim_chunk = rbf_kernel(chunk_embedding, embedding, gamma=gamma)
        elif metric == 'cosine':
            sim_chunk = cosine_similarity(chunk_embedding, embedding)
            sim_chunk = np.maximum(sim_chunk, 0)

        np.random.seed(random_seed) # Use a fixed seed for reproducible tie handling in argsort.
        indices_chunk = np.argsort(-sim_chunk, axis=1)[:, 1:k_neighbors + 1]
        data_chunk = np.take_along_axis(sim_chunk, indices_chunk, axis=1)
        rows_chunk = np.repeat(np.arange(start_idx, end_idx), k_neighbors)
        
        all_rows.append(rows_chunk)
        all_cols.append(indices_chunk.flatten())
        all_data.append(data_chunk.flatten())
        
    final_rows = np.concatenate(all_rows)
    final_cols = np.concatenate(all_cols)
    final_data = np.concatenate(all_data)
    W_sparse = csr_matrix((final_data, (final_rows, final_cols)), shape=(n_samples, n_samples))
    W_symmetric = (W_sparse + W_sparse.T) / 2
    print("Chunked graph construction completed.")
    return W_symmetric

def normalize_sparse(W_sparse: csr_matrix) -> csr_matrix:
    n_samples = W_sparse.shape[0]
    row_sums = W_sparse.sum(axis=1).A.flatten()
    denominators = 2.0 * row_sums
    denominators[denominators == 0] = 1e-9
    inv_denominators = 1.0 / denominators
    normalizer = sp.diags(inv_denominators)
    P_non_diag = normalizer @ W_sparse
    P_sparse = P_non_diag + sp.diags(np.full(n_samples, 0.5))
    return P_sparse.tocsr()

def sparsify_matrix(sparse_matrix: csr_matrix, k_neighbors: int, random_seed=42) -> csr_matrix:
    n_samples = sparse_matrix.shape[0]
    all_rows, all_cols, all_data = [], [], []
    np.random.seed(random_seed) # Use a fixed seed for reproducible argpartition behavior.
    for i in range(n_samples):
        row_slice = sparse_matrix.getrow(i)
        if row_slice.nnz > k_neighbors:
            data = row_slice.data
            indices = row_slice.indices
            top_k_indices_in_slice = np.argpartition(-data, k_neighbors)[:k_neighbors]
            top_data = data[top_k_indices_in_slice]
            top_cols = indices[top_k_indices_in_slice]
            all_rows.append(np.full(k_neighbors, i, dtype=np.int32))
            all_cols.append(top_cols)
            all_data.append(top_data)
        else:
            all_rows.append(np.full(row_slice.nnz, i, dtype=np.int32))
            all_cols.append(row_slice.indices)
            all_data.append(row_slice.data)
    final_rows = np.concatenate(all_rows)
    final_cols = np.concatenate(all_cols)
    final_data = np.concatenate(all_data)
    return csr_matrix((final_data, (final_rows, final_cols)), shape=(n_samples, n_samples))

def Hisemble(W_list: list, k_neighbors: int = 15, n_iterations: int = 20, random_seed=42) -> csr_matrix:
    num_views = len(W_list)
    n_samples = W_list[0].shape[0]
    P_list = [normalize_sparse(W) for W in W_list]
    k_retained = k_neighbors * 2
    for iter_num in range(n_iterations):
        print(f"  - Iteration {iter_num + 1}/{n_iterations}...")
        P_updated_list = []
        for i in range(num_views):
            sum_P_others = csr_matrix((n_samples, n_samples), dtype=np.float32)
            for j in range(num_views):
                if i != j:
                    sum_P_others += P_list[j]
            avg_P_others = sum_P_others / (num_views - 1)
            W_sparse = W_list[i]
            P_updated = W_sparse @ avg_P_others @ W_sparse.T 
            P_updated_sparsified = sparsify_matrix(P_updated, k_retained, random_seed=random_seed)
            P_updated = normalize_sparse(P_updated_sparsified)
            P_updated_list.append(P_updated)
        P_list = P_updated_list
    final_P = csr_matrix((n_samples, n_samples), dtype=np.float32)
    for P in P_list:
        final_P += P
    final_P /= num_views
    final_affinity = (final_P + final_P.T) / 2
    return final_affinity

# Main execution block.
def main():
    # 1. Parse arguments and configure the workflow.
    parser = argparse.ArgumentParser(
        description="Run Hisemble on single-cell datasets. Process the predefined dataset list by default, or select datasets on the command line.",
        formatter_class=argparse.RawTextHelpFormatter # Format command-line help.
    )
    parser.add_argument(
        '--datanames',  # Use the plural argument name for multiple datasets.
        type=str, 
        nargs='+',      # nargs='+' accepts one or more values.
        required=False, # The argument is optional.
        help="One or more space-separated dataset identifiers.\nIf omitted, process the complete predefined dataset list."
    )
    args = parser.parse_args()
    
    # Fixed hyperparameters and random seed.
    K_NEIGHBORS = 15
    N_ITERATIONS = 20
    RANDOM_SEED = 42
    
    # Set the global random seed.
    np.random.seed(RANDOM_SEED)
    tf.random.set_seed(RANDOM_SEED)
    os.environ['PYTHONHASHSEED'] = str(RANDOM_SEED)

    # 2. Select datasets to process.
    # Complete predefined dataset list.
    all_datanames_predefined = [
    "Chen-2019-ATAC", "FCA_fetal_33184180_cerebrum", "FCA_fetal_33184180_cerebellum", "FCA_fetal_33184180_intestine", "FCA_fetal_33184180_kidney", "FCA_fetal_33184180_lung", "FCA_fetal_33184180_muscle", "FCA_fetal_33184180_pancreas", "FCA_fetal_33184180_stomach", "mca_BoneMarrow_62216", "mca_Cerebellum_62216", "mca_HeartA_62816", "mca_Kidney_62016", "mca_LargeIntestineA_62816", "mca_LargeIntestineB_62816", "mca_Liver_62016", "mca_Lung1_62216", "mca_Lung2_62216", "mca_SmallIntestine_62816", "mca_Testes_62016", "mca_WholeBrainA_62216", "mca_WholeBrainA_62816", "mouse_brain", "TME-All"
    ]

    # Select the datasets to process.
    if args.datanames:
        # Use dataset identifiers supplied on the command line.
        datanames_to_process = args.datanames
        print(f"Processing the {len(datanames_to_process)} datasets selected on the command line:")
        print(datanames_to_process)
    else:
        # Otherwise, use the complete predefined dataset list.
        datanames_to_process = all_datanames_predefined
        print(f"No datasets specified; processing all {len(datanames_to_process)} predefined datasets.")

    # 3. Manage paths.
    base_output_dir = '__HISEMBLE_RESULTS_ROOT__/Hisemble_abla_lsi/'
    os.makedirs(base_output_dir, exist_ok=True)
    save_path = os.path.join(base_output_dir, f'abla_lsi_results.csv')

    # 4. Process the selected datasets.
    for dataname in datanames_to_process:
        print(f"\n{'='*20} Processing dataset: {dataname} {'='*20}")
        print(f"Fixed parameters: K={K_NEIGHBORS}, iterations={N_ITERATIONS}, seed={RANDOM_SEED}")

        try:
            # Load input data.
            with open(f'__HISEMBLE_DATA_ROOT__/ADATA/{dataname}_scBasset.pkl', 'rb') as f:
                ad_scBasset = pickle.load(f)
            with open(f'__HISEMBLE_DATA_ROOT__/ADATA/{dataname}_snap.pkl', 'rb') as f:
                ad_snap = pickle.load(f)
            
            true_labels = ad_snap.obs.cell_type.to_numpy()
            true_cell_types = ad_snap.obs['cell_type'].nunique()
            
            # Core numerical workflow.
            # a. Construct the initial graphs.
            print("\n--- Constructing initial sparse affinity matrices ---")
            W1 = build_knn_graph_chunked_cpu(ad_scBasset.obsm['projection'], k_neighbors=K_NEIGHBORS, metric='gaussian', random_seed=RANDOM_SEED)
            W2 = build_knn_graph_chunked_cpu(ad_snap.obsm['X_spectral'], k_neighbors=K_NEIGHBORS, metric='gaussian', random_seed=RANDOM_SEED)
            W_list = [W1, W2]
            del W1, W2

            # b. Run Hisemble and monitor resources.
            with MemoryMonitor() as monitor:
                start_time = time.time()
                fused_network = Hisemble(W_list=W_list, k_neighbors=K_NEIGHBORS, n_iterations=N_ITERATIONS, random_seed=RANDOM_SEED)
                end_time = time.time()
            hisemble_runtime_s = end_time - start_time
            hisemble_peak_mem_mb = monitor.peak_mb
            del W_list
            
            # c. Clustering and evaluation.
            print("\n--- Downstream clustering and evaluation ---")
            # Clustering and evaluation follow the main workflow.
            sources, targets = fused_network.nonzero()
            weights = fused_network[sources, targets].A1
            net = ig.Graph(list(zip(sources, targets)), directed=False)
            net.es['weight'] = weights
            del fused_network, sources, targets, weights
            
            clusters = find_best_matrix(matrix=net, true_cell_types=true_cell_types, seed=RANDOM_SEED)
            del net
            
            ARI = adjusted_rand_score(true_labels, clusters)
            AMI = adjusted_mutual_info_score(true_labels, clusters)
            NMI = normalized_mutual_info_score(true_labels, clusters)
            HOM = homogeneity_score(true_labels, clusters)
            
            # Collect results.
            result_data = {
                'dataname': [dataname],
                'k_neighbors': [K_NEIGHBORS],
                'n_iterations': [N_ITERATIONS],
                'ARI': [ARI],
                'AMI': [AMI],
                'NMI': [NMI],
                'HOM': [HOM],
                'Hisemble_Time_s': [hisemble_runtime_s],
                'Hisemble_Peak_Mem_MB': [hisemble_peak_mem_mb]
            }
            result_df = pd.DataFrame(result_data)
            
            # Update the results CSV while preserving existing datasets.
            if os.path.exists(save_path):
                # Update existing CSV records using the main workflow.
                existing_df = pd.read_csv(save_path).set_index('dataname')
                new_row_df = result_df.set_index('dataname')
                existing_df.update(new_row_df)
                new_datanames = new_row_df.index.difference(existing_df.index)
                final_df = pd.concat([existing_df, new_row_df.loc[new_datanames]])
                final_df.reset_index(inplace=True)
            else:
                final_df = result_df

            final_df.to_csv(save_path, index=False)

            print(f"\n{'='*20} Dataset {dataname} completed {'='*20}")
            print("Results from this run:")
            print(result_df)
            print(f"\nAll results updated at: {save_path}")

        except FileNotFoundError as e:
            print(f"Data loading failed: {e}. Check the input paths.")
        except Exception as e:
            print(f"Error while processing dataset {dataname}: {e}")
            continue

if __name__ == '__main__':
    main()
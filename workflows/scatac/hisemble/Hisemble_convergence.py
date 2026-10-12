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
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings("ignore")
import tensorflow as tf
from sklearn.metrics.pairwise import rbf_kernel, euclidean_distances
from sklearn.metrics import adjusted_rand_score,adjusted_mutual_info_score,normalized_mutual_info_score, homogeneity_score
from scipy.spatial.distance import pdist, squareform
import igraph as ig
import leidenalg
from sklearn.metrics import pairwise_distances
from scipy.sparse import csr_matrix, identity as sparse_identity
from scipy.sparse.linalg import norm as sparse_norm
from sklearn.neighbors import NearestNeighbors
import time
import psutil
import threading
import argparse
import umap
import os



def find_best_matrix(matrix, true_cell_types, low=0.1, high=5.0, tolerance=0.01, seed=42):
    best_resolution = None
    clusters = None
    def remove_duplicates(lst):  
        return list(np.unique(lst))
        
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

def calculate_neighbor_gap(matrix, k):
    """Calculate the mean gap between the kth and (k+1)th largest entries in each row.
This summarizes separation at the neighbor-retention boundary.
"""
    gaps = []
    n_rows = matrix.shape[0]
    
    # Process the matrix row by row.
    # Dense row conversion may be expensive for very large matrices.
    # Include implicit zeros when finding the largest k+1 entries in sparse rows.
    
    for i in range(n_rows):
        if issparse(matrix):
            # Read all row entries, including implicit zeros.
            # Use a dense row to account for implicit zeros when few entries are stored.
            row = matrix[i, :].toarray().flatten()
        else:
            row = matrix[i, :]
            
        # Select the kth and (k+1)th largest values using one-based ranks.
        # Their sorted-array indices are -k and -(k+1).
        # Use partition to select the largest k+1 entries efficiently.
        if len(row) > k:
            # Partition at -(k+1) to isolate the largest k+1 entries.
            partitioned = np.partition(row, -(k + 1))
            top_k_plus_1 = partitioned[-(k + 1):]
            top_k_plus_1.sort() # Sort ascending: [(k+1)th largest, kth largest, ..., largest].
            
            # top_k_plus_1[0] is the (k+1)th largest entry.
            # top_k_plus_1[1] is the kth largest entry.
            gap = top_k_plus_1[1] - top_k_plus_1[0]
            gaps.append(gap)
        else:
            gaps.append(0.0)
            
    return np.mean(gaps) if gaps else 0.0


def Hisemble(W_list: list, k_neighbors: int = 20, n_iterations: int = 50, random_seed=42, dataname="dataset", checkpoint_iter: int = None):
    num_views = len(W_list)
    n_samples = W_list[0].shape[0]
    view_names = ["scBasset", "SnapATAC2", "LSI"]
    P_list = [normalize_sparse(W) for W in W_list]
    k_retained = k_neighbors * 2

    # Store the previous iteration's Q matrices.
    prev_Q_list = [None] * num_views
    convergence_data = []
    affinity_at_checkpoint = None

    for iter_num in range(n_iterations):
        print(f"  - Iteration {iter_num + 1}/{n_iterations}...")
        P_next_list = []
        
        # Calculate and monitor the consensus Q before updating P.
        for i in range(num_views):
            # A. Calculate the current consensus Q_t.
            sum_P_others = csr_matrix((n_samples, n_samples), dtype=np.float32)
            for j in range(num_views):
                if i != j:
                    sum_P_others += P_list[j]
            Q_t = sum_P_others / (num_views - 1)
        
            avg_P_others = sum_P_others / (num_views - 1)
            
            # Propagation uses the fixed initial graph, as in the main analysis.
            W_sparse = W_list[i]
            P_updated = W_sparse @ avg_P_others @ W_sparse.T 
            P_updated_sparsified = sparsify_matrix(P_updated, k_retained, random_seed=random_seed)
            P_updated_norm = normalize_sparse(P_updated_sparsified)
            
            diff = np.abs(P_updated_norm - P_list[i]).sum() / (n_samples * k_retained)
            # B. Compare Q_t with Q_{t-1}.
            delta_Q = 0.0
            hamming_dist = 0
            if prev_Q_list[i] is not None:
                # Frobenius norm: ||Q_t - Q_{t-1}||_F.
                delta_Q = sparse_norm(Q_t - prev_Q_list[i], ord='fro')
            
            # C. Record the current iteration.
            # Relative Change tracks the P update; Delta_Q preserves the pre-update recording convention.
            # Delta_Q measures consensus change; support diagnostics track graph topology.
            convergence_data.append({
                "Dataset": dataname,
                "Iteration": iter_num + 1,
                "View": view_names[i],
                "Relative Change": diff,
                "Delta_Q": delta_Q
            })
            
            # E. Store Q_t for the next iteration.
            prev_Q_list[i] = Q_t.copy()
            P_next_list.append(P_updated_norm)
            
        P_list = P_next_list

        # Capture clustering metrics at the requested checkpoint.
        if checkpoint_iter is not None and (iter_num + 1) == checkpoint_iter:
            print(f"  - Capturing the fused matrix at iteration {checkpoint_iter} for evaluation...")
            P_checkpoint_sum = csr_matrix((n_samples, n_samples), dtype=np.float32)
            for P in P_list:
                P_checkpoint_sum += P
            P_checkpoint_sum /= num_views
            affinity_at_checkpoint = (P_checkpoint_sum + P_checkpoint_sum.T) / 2

    # Construct the final fused matrix after all iterations.
    final_P = csr_matrix((n_samples, n_samples), dtype=np.float32)
    for P in P_list:
        final_P += P
    final_P /= num_views
    final_affinity = (final_P + final_P.T) / 2
    
    # Use the final matrix if the requested checkpoint was not reached.
    if affinity_at_checkpoint is None and checkpoint_iter is not None:
        print(f"  Warning: checkpoint {checkpoint_iter} was not reached. Evaluating the final matrix instead.")
        affinity_at_checkpoint = final_affinity

    # Return the final matrix, checkpoint matrix and history.
    return final_affinity, affinity_at_checkpoint, pd.DataFrame(convergence_data)



# Main execution block.
def main():
    parser = argparse.ArgumentParser(
        description="Run Hisemble on single-cell datasets. Process the predefined dataset list by default, or select datasets on the command line.",
        formatter_class=argparse.RawTextHelpFormatter # Format command-line help.
    )
    parser.add_argument(
        '--datanames',
        type=str, 
        nargs='+',
        required=False,
        help="One or more space-separated dataset identifiers.\nIf omitted, process the complete predefined dataset list."
    )
    args = parser.parse_args()
    
    # Fixed hyperparameters and random seed.
    K_NEIGHBORS = 15
    N_ITERATIONS = 50 # Total number of iterations.
    METRICS_CHECKPOINT_ITER = 20 # Evaluate clustering after this iteration.
    RANDOM_SEED = 42
    
    # Set the global random seed.
    np.random.seed(RANDOM_SEED)
    tf.random.set_seed(RANDOM_SEED)
    os.environ['PYTHONHASHSEED'] = str(RANDOM_SEED)

    all_datanames_predefined = [
        'Chen-2019-ATAC'
    ]

    all_convergence_dfs = []
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
    base_output_dir = '__HISEMBLE_RESULTS_ROOT__/Hisemble/'

    os.makedirs(base_output_dir, exist_ok=True)
    metrics_save_path = os.path.join(base_output_dir, 'all_datasets_metrics_summary.csv')
    conv_save_path = os.path.join(base_output_dir, 'all_convergence_results.csv')
    
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
            with open(f'__HISEMBLE_DATA_ROOT__/lsi_results/ADATA/{dataname}_lsi.pkl', 'rb') as f:
                ad_lsi = pickle.load(f)
            
            true_labels = ad_snap.obs.cell_type.to_numpy()
            true_cell_types = ad_snap.obs['cell_type'].nunique()
            
            # Core numerical workflow.
            # a. Construct the initial graphs.
            print("\n--- Constructing initial sparse affinity matrices ---")
            W1 = build_knn_graph_chunked_cpu(ad_scBasset.obsm['projection'], k_neighbors=K_NEIGHBORS, metric='gaussian', random_seed=RANDOM_SEED)
            W2 = build_knn_graph_chunked_cpu(ad_snap.obsm['X_spectral'], k_neighbors=K_NEIGHBORS, metric='gaussian', random_seed=RANDOM_SEED)
            W3 = build_knn_graph_chunked_cpu(ad_lsi.obsm['X_lsi'], k_neighbors=K_NEIGHBORS, metric='gaussian', random_seed=RANDOM_SEED)
            W_list = [W1, W2, W3]
            del W1, W2, W3
            
            _, fused_network, conv_df = Hisemble(
                W_list=W_list, 
                k_neighbors=K_NEIGHBORS, 
                n_iterations=N_ITERATIONS, 
                random_seed=RANDOM_SEED, 
                dataname=dataname,
                checkpoint_iter=METRICS_CHECKPOINT_ITER
            )
            
            del W_list
            
            if os.path.exists(conv_save_path):
                existing_conv = pd.read_csv(conv_save_path)
                # Replace prior records for this dataset to avoid duplicates on reruns.
                existing_conv = existing_conv[existing_conv['Dataset'] != dataname]
                final_conv_df = pd.concat([existing_conv, conv_df], ignore_index=True)
            else:
                final_conv_df = conv_df
            final_conv_df.to_csv(conv_save_path, index=False)
            print(f"  Convergence records updated at: {conv_save_path}")
            
            # c. Clustering and evaluation.
            print("\n--- Downstream clustering and evaluation ---")
            # Clustering and evaluation follow the main workflow.
            sources, targets = fused_network.nonzero()
            weights = fused_network[sources, targets].A1
            net = ig.Graph(list(zip(sources, targets)), directed=False)
            net.es['weight'] = weights
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
                'n_iterations_total': [N_ITERATIONS],
                'n_iterations_metrics': [METRICS_CHECKPOINT_ITER], # Record the iteration used for evaluation.
                'ARI': [ARI],
                'AMI': [AMI],
                'NMI': [NMI],
                'HOM': [HOM]
            }
            result_df = pd.DataFrame(result_data)

            if os.path.exists(metrics_save_path):
                existing_metrics = pd.read_csv(metrics_save_path)
                if dataname in existing_metrics['dataname'].values:
                    existing_metrics.set_index('dataname', inplace=True)
                    existing_metrics.update(result_df.set_index('dataname'))
                    final_metrics_df = existing_metrics.reset_index()
                else:
                    final_metrics_df = pd.concat([existing_metrics, result_df], ignore_index=True)
            else:
                final_metrics_df = result_df
            final_metrics_df.to_csv(metrics_save_path, index=False)
            print(f"  Metrics updated at: {metrics_save_path}")
            
            if all_convergence_dfs:
                full_convergence_df = pd.concat(all_convergence_dfs)
                convergence_save_path = os.path.join(base_output_dir, "all_convergence_data.csv")
                full_convergence_df.to_csv(convergence_save_path, index=False)
                print(f"\nConvergence records for all views saved to: {convergence_save_path}")

        except FileNotFoundError as e:
            print(f"Data loading failed: {e}. Check the input paths.")
        except Exception as e:
            print(f"Error while processing dataset {dataname}: {e}")
            continue


if __name__ == '__main__':
    main()
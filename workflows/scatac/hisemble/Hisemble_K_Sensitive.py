import pandas as pd
import numpy as np
import sklearn
from sklearn import metrics
import scipy
import scipy.sparse as sp
from scipy.sparse import issparse
import scanpy as sc
import anndata
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings("ignore")
import tensorflow as tf
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
import igraph as ig
import leidenalg
from scipy.sparse import csr_matrix
import argparse
import os

# Core helper functions.

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

    # Limit progress logging.
    # Optional progress message: building the k-NN graph.
    for i in range(0, n_samples, chunk_size):
        start_idx = i
        end_idx = min(i + chunk_size, n_samples)
        chunk_embedding = embedding[start_idx:end_idx]
        if metric == 'gaussian':
            sim_chunk = rbf_kernel(chunk_embedding, embedding, gamma=gamma)
        elif metric == 'cosine':
            sim_chunk = cosine_similarity(chunk_embedding, embedding)
            sim_chunk = np.maximum(sim_chunk, 0)

        np.random.seed(random_seed) 
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
    np.random.seed(random_seed)
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

def Hisemble(W_list: list, k_neighbors: int = 20, n_iterations: int = 20, random_seed=42) -> csr_matrix:
    num_views = len(W_list)
    n_samples = W_list[0].shape[0]
    P_list = [normalize_sparse(W) for W in W_list]
    k_retained = k_neighbors * 2
    for iter_num in range(n_iterations):
        # Optional iteration progress logging.
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

# Main neighborhood-sensitivity workflow.
def main():
    parser = argparse.ArgumentParser(description="Hisemble neighborhood-size sensitivity analysis.")
    parser.add_argument('--datanames', type=str, nargs='+', required=False, help="Dataset identifiers to process.")
    args = parser.parse_args()
    
    # Sensitivity-analysis settings.
    k_values_to_test = [10, 15, 20, 25, 30, 40, 50]
    N_ITERATIONS = 20
    RANDOM_SEED = 42
    
    # Global random seed.
    np.random.seed(RANDOM_SEED)
    tf.random.set_seed(RANDOM_SEED)
    os.environ['PYTHONHASHSEED'] = str(RANDOM_SEED)

    # Predefined dataset list.
    all_datanames_predefined = ['Chen-2019-ATAC']
    datanames_to_process = args.datanames if args.datanames else all_datanames_predefined
    
    # Configure paths.
    base_output_dir = '__HISEMBLE_RESULTS_ROOT__/Hisemble_final/'
    os.makedirs(base_output_dir, exist_ok=True)
    # Use a separate sensitivity-analysis filename to preserve benchmark results.
    save_path = os.path.join(base_output_dir, 'hisemble_sensitivity_k_neighbors.csv')

    print(f"Starting sensitivity analysis.")
    print(f"Neighborhood sizes to evaluate: {k_values_to_test}")
    print(f"Results will be saved to: {save_path}")

    # Outer loop: datasets.
    for dataname in datanames_to_process:
        print(f"\n{'#'*30}")
        print(f"Loading dataset: {dataname}")
        print(f"{'#'*30}")

        try:
            # 1. Load each dataset once to avoid repeated I/O.
            with open(f'__HISEMBLE_DATA_ROOT__/ADATA/{dataname}_scBasset.pkl', 'rb') as f:
                ad_scBasset = pickle.load(f)
            with open(f'__HISEMBLE_DATA_ROOT__/ADATA/{dataname}_snap.pkl', 'rb') as f:
                ad_snap = pickle.load(f)
            with open(f'__HISEMBLE_DATA_ROOT__/lsi_results/ADATA/{dataname}_lsi.pkl', 'rb') as f:
                ad_lsi = pickle.load(f)
            
            true_labels = ad_snap.obs.cell_type.to_numpy()
            true_cell_types = ad_snap.obs['cell_type'].nunique()

            # Inner loop: neighborhood sizes.
            for k in k_values_to_test:
                print(f"\n--- Running with K_NEIGHBORS = {k} ---")
                
                # a. Construct graphs for the current neighborhood size.
                W1 = build_knn_graph_chunked_cpu(ad_scBasset.obsm['projection'], k_neighbors=k, metric='gaussian', random_seed=RANDOM_SEED)
                W2 = build_knn_graph_chunked_cpu(ad_snap.obsm['X_spectral'], k_neighbors=k, metric='gaussian', random_seed=RANDOM_SEED)
                W3 = build_knn_graph_chunked_cpu(ad_lsi.obsm['X_lsi'], k_neighbors=k, metric='cosine', random_seed=RANDOM_SEED)
                W_list = [W1, W2, W3]
                del W1, W2, W3

                # b. Run Hisemble with the current neighborhood size.
                fused_network = Hisemble(W_list=W_list, k_neighbors=k, n_iterations=N_ITERATIONS, random_seed=RANDOM_SEED)
                del W_list

                # c. Clustering and evaluation.
                sources, targets = fused_network.nonzero()
                weights = fused_network[sources, targets].A1
                net = ig.Graph(list(zip(sources, targets)), directed=False)
                net.es['weight'] = weights
                
                # Search for the clustering resolution.
                clusters = find_best_matrix(matrix=net, true_cell_types=true_cell_types, seed=RANDOM_SEED)
                del net
                
                # Calculate evaluation metrics.
                ARI = adjusted_rand_score(true_labels, clusters)
                NMI = normalized_mutual_info_score(true_labels, clusters)
                
                print(f"K={k} -> ARI: {ARI:.4f}, NMI: {NMI:.4f}")

                # d. Save each result immediately.
                result_row = {
                    'dataname': dataname,
                    'k_neighbors': k,
                    'ARI': ARI,
                    'NMI': NMI
                }
                
                new_df = pd.DataFrame([result_row])
                
                if os.path.exists(save_path):
                    # Append rows without repeating the header.
                    new_df.to_csv(save_path, mode='a', header=False, index=False)
                else:
                    # Write the header when creating the file.
                    new_df.to_csv(save_path, mode='w', header=True, index=False)

        except Exception as e:
            print(f"Error while processing {dataname}: {e}")
            continue

    print(f"\nSensitivity analysis completed. Results saved to: {save_path}")

if __name__ == '__main__':
    main()
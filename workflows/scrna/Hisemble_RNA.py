import pandas as pd
import numpy as np
import sklearn
import scipy.sparse as sp
import scanpy as sc
import matplotlib.pyplot as plt
import os
import time
import psutil
import threading
import argparse
import warnings

# Suppress nonessential warnings.
warnings.filterwarnings("ignore")
import tensorflow as tf
from sklearn.metrics.pairwise import rbf_kernel, cosine_similarity
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score, normalized_mutual_info_score, homogeneity_score
import igraph as ig
import leidenalg
from scipy.sparse import csr_matrix

# ==============================================================================
# Core utilities and fusion algorithm
# ==============================================================================

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

# Helper: calculate clustering metrics.
def calculate_metrics(labels_true, labels_pred):
    return {
        'ARI': adjusted_rand_score(labels_true, labels_pred),
        'AMI': adjusted_mutual_info_score(labels_true, labels_pred),
        'NMI': normalized_mutual_info_score(labels_true, labels_pred),
        'HOM': homogeneity_score(labels_true, labels_pred)
    }
    
# ==============================================================================
# Main scRNA workflow using PCA and scVI views.
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Run Hisemble on PCA and scVI embeddings from scRNA-seq data and evaluate the individual views as benchmarks.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        '--tissues',  
        type=str, 
        nargs='+',      
        required=False, 
        help="Space-separated Tabula Sapiens tissue names, such as Pancreas Liver.\nIf omitted, process the six default tissues."
    )
    args = parser.parse_args()
    
    # Fixed hyperparameters.
    K_NEIGHBORS = 5
    N_ITERATIONS = 20
    RANDOM_SEED = 42
    
    np.random.seed(RANDOM_SEED)
    tf.random.set_seed(RANDOM_SEED)
    os.environ['PYTHONHASHSEED'] = str(RANDOM_SEED)

    # Input and output paths.
    data_dir = "Tabula_Sapiens"
    base_output_dir = "results/Hisemble_RNA"
    os.makedirs(base_output_dir, exist_ok=True)
    
    save_path = os.path.join(base_output_dir, 'hisemble_scRNA_benchmark_results.csv')
    save_folder_umap = os.path.join(base_output_dir, 'UMAP_plots')
    os.makedirs(save_folder_umap, exist_ok=True)
    
    print(f"UMAP figures will be saved to: {save_folder_umap}")

    # Select tissues to process.
    all_tissues_predefined = ["Pancreas", "Liver"]
    tissues_to_process = args.tissues if args.tissues else all_tissues_predefined

    for tissue in tissues_to_process:
        print(f"\n{'='*25} Processing tissue: {tissue} {'='*25}")
        
        file_path = os.path.join(data_dir, f"TS_{tissue}_30k_processed.h5ad")
        
        try:
            # 1. Load data.
            print(f"Loading data: {file_path}")
            adata = sc.read_h5ad(file_path)
            
            # Check that reference labels are available.
            if 'cell_type' not in adata.obs:
                raise ValueError(f"Dataset {tissue} lacks cell_type annotations required for evaluation.")
                
            true_labels = adata.obs['cell_type'].astype(str).to_numpy()
            true_cell_types = adata.obs['cell_type'].nunique()
            print(f"Found {adata.n_obs} cells and {true_cell_types} annotated cell types.")

            # 2. Construct initial graphs for both views.
            print("\n--- Constructing view 1: PCA ---")
            W1 = build_knn_graph_chunked_cpu(adata.obsm['X_pca'], k_neighbors=K_NEIGHBORS, metric='gaussian', random_seed=RANDOM_SEED)
            
            print("\n--- Constructing view 2: scVI ---")
            W2 = build_knn_graph_chunked_cpu(adata.obsm['X_scvi'], k_neighbors=K_NEIGHBORS, metric='cosine', random_seed=RANDOM_SEED)

            # Evaluate individual-view clustering benchmarks.
            print("\n--- Evaluating individual-view clustering: PCA ---")
            sources_pca, targets_pca = W1.nonzero()
            net_pca = ig.Graph(list(zip(sources_pca, targets_pca)), directed=False)
            net_pca.es['weight'] = W1[sources_pca, targets_pca].A1
            clusters_pca = find_best_matrix(matrix=net_pca, true_cell_types=true_cell_types, seed=RANDOM_SEED)
            del net_pca

            print("\n--- Evaluating individual-view clustering: scVI ---")
            sources_scvi, targets_scvi = W2.nonzero()
            net_scvi = ig.Graph(list(zip(sources_scvi, targets_scvi)), directed=False)
            net_scvi.es['weight'] = W2[sources_scvi, targets_scvi].A1
            clusters_scvi = find_best_matrix(matrix=net_scvi, true_cell_types=true_cell_types, seed=RANDOM_SEED)
            del net_scvi

            # 3. Run Hisemble fusion.
            W_list = [W1, W2]
            print("\n--- Running Hisemble network fusion ---")
            with MemoryMonitor() as monitor:
                start_time = time.time()
                fused_network = Hisemble(W_list=W_list, k_neighbors=K_NEIGHBORS, n_iterations=N_ITERATIONS, random_seed=RANDOM_SEED)
                end_time = time.time()
                
            hisemble_runtime_s = end_time - start_time
            hisemble_peak_mem_mb = monitor.peak_mb
            del W_list, W1, W2 # Release unused memory.

            # 4. Leiden clustering of the fused graph.
            print("\n--- Running Leiden clustering on the fused graph ---")
            sources, targets = fused_network.nonzero()
            weights = fused_network[sources, targets].A1
            net = ig.Graph(list(zip(sources, targets)), directed=False)
            net.es['weight'] = weights
            clusters_hisemble = find_best_matrix(matrix=net, true_cell_types=true_cell_types, seed=RANDOM_SEED)
            del net

            # 5. Update AnnData and UMAP.
            fused_network = fused_network.astype(np.float32)
            adata.obsp['connectivities'] = fused_network
            distances = adata.obsp['connectivities'].copy()
            distances.data = 1.0 - distances.data
            distances.eliminate_zeros()
            adata.obsp['distances'] = distances

            adata.uns['neighbors'] = {
                'params': {'method': 'custom', 'custom_graph': True, 'n_neighbors': K_NEIGHBORS},
                'connectivities_key': 'connectivities',
                'distances_key': 'distances'
            }

            print("Computing UMAP from the fused graph...")
            sc.tl.umap(adata, random_state=RANDOM_SEED)
            
            # Store all clustering results in AnnData.
            adata.obs['PCA_Cluster'] = pd.Categorical(clusters_pca)
            adata.obs['scVI_Cluster'] = pd.Categorical(clusters_scvi)
            adata.obs['Hisemble_Cluster'] = pd.Categorical(clusters_hisemble)

            # 6. Plot and save outputs.
            plt.rcParams.update({
                'font.size': 14,
                'axes.titlesize': 18,
                'axes.labelsize': 16,
                'axes.titleweight': 'bold',
                'axes.labelweight': 'bold',
                'text.color': 'black',
                'axes.labelcolor': 'black',
                'xtick.color': 'black',
                'ytick.color': 'black',
            })
            
            plot_targets = [
                ('cell_type', 'True Cell Types'),
                ('PCA_Cluster', 'PCA Clusters'),
                ('scVI_Cluster', 'scVI Clusters'),
                ('Hisemble_Cluster', 'Hisemble Clusters')
            ]
            
            for col, title in plot_targets:
                sc.pl.umap(adata, color=[col], show=False, legend_fontsize=12, title=f"{tissue} - {title}")
                plt.savefig(os.path.join(save_folder_umap, f'UMAP_{tissue}_{col}.pdf'), dpi=300, format="pdf", bbox_inches='tight')
                plt.close()

            # 7. Collect metrics for all views.
            metrics_pca = calculate_metrics(true_labels, clusters_pca)
            metrics_scvi = calculate_metrics(true_labels, clusters_scvi)
            metrics_hisemble = calculate_metrics(true_labels, clusters_hisemble)
            
            results = [
                {'Tissue': tissue, 'Views': 'PCA_only', 'K': K_NEIGHBORS, **metrics_pca, 'Time_s': 0, 'Peak_Mem_MB': 0},
                {'Tissue': tissue, 'Views': 'scVI_only', 'K': K_NEIGHBORS, **metrics_scvi, 'Time_s': 0, 'Peak_Mem_MB': 0},
                {'Tissue': tissue, 'Views': 'PCA+scVI', 'K': K_NEIGHBORS, **metrics_hisemble, 'Time_s': hisemble_runtime_s, 'Peak_Mem_MB': hisemble_peak_mem_mb}
            ]
            result_df = pd.DataFrame(results)
            
            # Update the CSV and deduplicate Tissue/Views records.
            if os.path.exists(save_path):
                existing_df = pd.read_csv(save_path)
                final_df = pd.concat([existing_df, result_df]).drop_duplicates(subset=['Tissue', 'Views'], keep='last')
            else:
                final_df = result_df

            final_df.to_csv(save_path, index=False)

            # Save AnnData.
            h5ad_save_path = os.path.join(base_output_dir, f'TS_{tissue}_Hisemble_Result.h5ad')
            adata.write(h5ad_save_path, compression="gzip")
            
            print(f"\n{tissue} completed.")
            print(f"Benchmark comparison:")
            print(f"   - PCA_only : ARI={metrics_pca['ARI']:.4f}, NMI={metrics_pca['NMI']:.4f}")
            print(f"   - scVI_only: ARI={metrics_scvi['ARI']:.4f}, NMI={metrics_scvi['NMI']:.4f}")
            print(f"   - Hisemble : ARI={metrics_hisemble['ARI']:.4f}, NMI={metrics_hisemble['NMI']:.4f}")
            print(f"Data saved to: {h5ad_save_path}")

        except FileNotFoundError:
            print(f"Data loading failed: {file_path} was not found. Complete preprocessing first.")
        except Exception as e:
            print(f"Error while processing {tissue}: {e}")
            import traceback
            traceback.print_exc()
            continue

if __name__ == '__main__':
    main()
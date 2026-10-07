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

# 禁用不必要的警告
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
        print(f"主算法峰值内存使用: {self.peak_mb:.2f} MB")

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
    print(f"最终使用分辨率: {best_resolution:.4f}, 产生 {len(remove_duplicates(final_clusters))} 个簇。")
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

    print(f"开始分块构建K-NN图 (chunk_size={chunk_size}, metric={metric})...")
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
    print("分块构建完成！")
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
        print(f"  - 第 {iter_num + 1}/{n_iterations} 次迭代...")
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

# 辅助函数：计算各类聚类指标
def calculate_metrics(labels_true, labels_pred):
    return {
        'ARI': adjusted_rand_score(labels_true, labels_pred),
        'AMI': adjusted_mutual_info_score(labels_true, labels_pred),
        'NMI': normalized_mutual_info_score(labels_true, labels_pred),
        'HOM': homogeneity_score(labels_true, labels_pred)
    }
    
# ==============================================================================
# 针对 scRNA (PCA + scVI) 的主执行逻辑
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="为 scRNA-seq 数据集运行 Hisemble 双视图融合聚类 (PCA + scVI)，同时评估单一视图作为 Benchmark。",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        '--tissues',  
        type=str, 
        nargs='+',      
        required=False, 
        help="要处理的 Tabula Sapiens 组织名称（用空格隔开，如: Pancreas Liver）。\n未提供则处理默认的 6 个组织。"
    )
    args = parser.parse_args()
    
    # --- 固定超参数 ---
    K_NEIGHBORS = 5
    N_ITERATIONS = 20
    RANDOM_SEED = 42
    
    np.random.seed(RANDOM_SEED)
    tf.random.set_seed(RANDOM_SEED)
    os.environ['PYTHONHASHSEED'] = str(RANDOM_SEED)

    # --- 数据输入输出路径 ---
    data_dir = "Tabula_Sapiens"
    base_output_dir = "results/Hisemble_RNA"
    os.makedirs(base_output_dir, exist_ok=True)
    
    save_path = os.path.join(base_output_dir, 'hisemble_scRNA_benchmark_results.csv')
    save_folder_umap = os.path.join(base_output_dir, 'UMAP_plots')
    os.makedirs(save_folder_umap, exist_ok=True)
    
    print(f"UMAP 图像将被保存到: {save_folder_umap}")

    # --- 决定处理列表 ---
    all_tissues_predefined = ["Pancreas", "Liver"]
    tissues_to_process = args.tissues if args.tissues else all_tissues_predefined

    for tissue in tissues_to_process:
        print(f"\n{'='*25} 开始处理组织: {tissue} {'='*25}")
        
        file_path = os.path.join(data_dir, f"TS_{tissue}_30k_processed.h5ad")
        
        try:
            # 1. 数据加载
            print(f"正在加载数据: {file_path}")
            adata = sc.read_h5ad(file_path)
            
            # 确保含有真实标签
            if 'cell_type' not in adata.obs:
                raise ValueError(f"数据 {tissue} 缺少 'cell_type' 标注，无法进行评估！")
                
            true_labels = adata.obs['cell_type'].astype(str).to_numpy()
            true_cell_types = adata.obs['cell_type'].nunique()
            print(f"发现 {adata.n_obs} 个细胞, {true_cell_types} 种真实细胞类型。")

            # 2. 构建双视图初始图
            print("\n--- 构建视图 1: 基于 PCA ---")
            W1 = build_knn_graph_chunked_cpu(adata.obsm['X_pca'], k_neighbors=K_NEIGHBORS, metric='gaussian', random_seed=RANDOM_SEED)
            
            print("\n--- 构建视图 2: 基于 scVI ---")
            W2 = build_knn_graph_chunked_cpu(adata.obsm['X_scvi'], k_neighbors=K_NEIGHBORS, metric='cosine', random_seed=RANDOM_SEED)

            # --- 单一模态聚类评估 (Benchmark) ---
            print("\n--- 评估单一视图聚类: PCA ---")
            sources_pca, targets_pca = W1.nonzero()
            net_pca = ig.Graph(list(zip(sources_pca, targets_pca)), directed=False)
            net_pca.es['weight'] = W1[sources_pca, targets_pca].A1
            clusters_pca = find_best_matrix(matrix=net_pca, true_cell_types=true_cell_types, seed=RANDOM_SEED)
            del net_pca

            print("\n--- 评估单一视图聚类: scVI ---")
            sources_scvi, targets_scvi = W2.nonzero()
            net_scvi = ig.Graph(list(zip(sources_scvi, targets_scvi)), directed=False)
            net_scvi.es['weight'] = W2[sources_scvi, targets_scvi].A1
            clusters_scvi = find_best_matrix(matrix=net_scvi, true_cell_types=true_cell_types, seed=RANDOM_SEED)
            del net_scvi

            # 3. 运行 Hisemble 融合
            W_list = [W1, W2]
            print("\n--- 开始执行 Hisemble 网络融合 ---")
            with MemoryMonitor() as monitor:
                start_time = time.time()
                fused_network = Hisemble(W_list=W_list, k_neighbors=K_NEIGHBORS, n_iterations=N_ITERATIONS, random_seed=RANDOM_SEED)
                end_time = time.time()
                
            snf_runtime_s = end_time - start_time
            snf_peak_mem_mb = monitor.peak_mb
            del W_list, W1, W2 # 及时清理内存

            # 4. Leiden 聚类 (融合网络)
            print("\n--- 启动 Leiden 聚类 (融合网络) ---")
            sources, targets = fused_network.nonzero()
            weights = fused_network[sources, targets].A1
            net = ig.Graph(list(zip(sources, targets)), directed=False)
            net.es['weight'] = weights
            clusters_hisemble = find_best_matrix(matrix=net, true_cell_types=true_cell_types, seed=RANDOM_SEED)
            del net

            # 5. 更新 Adata 与 UMAP
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

            print("正在计算融合后的 UMAP...")
            sc.tl.umap(adata, random_state=RANDOM_SEED)
            
            # 将所有聚类结果保存到 Adata
            adata.obs['PCA_Cluster'] = pd.Categorical(clusters_pca)
            adata.obs['scVI_Cluster'] = pd.Categorical(clusters_scvi)
            adata.obs['Hisemble_Cluster'] = pd.Categorical(clusters_hisemble)

            # 6. 画图并保存
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

            # 7. 计算并汇总所有视图的指标
            metrics_pca = calculate_metrics(true_labels, clusters_pca)
            metrics_scvi = calculate_metrics(true_labels, clusters_scvi)
            metrics_hisemble = calculate_metrics(true_labels, clusters_hisemble)
            
            results = [
                {'Tissue': tissue, 'Views': 'PCA_only', 'K': K_NEIGHBORS, **metrics_pca, 'Time_s': 0, 'Peak_Mem_MB': 0},
                {'Tissue': tissue, 'Views': 'scVI_only', 'K': K_NEIGHBORS, **metrics_scvi, 'Time_s': 0, 'Peak_Mem_MB': 0},
                {'Tissue': tissue, 'Views': 'PCA+scVI', 'K': K_NEIGHBORS, **metrics_hisemble, 'Time_s': snf_runtime_s, 'Peak_Mem_MB': snf_peak_mem_mb}
            ]
            result_df = pd.DataFrame(results)
            
            # 更新 CSV 记录表 (使用 concat 并去重，防止相同 Tissue 和 Views 产生重复数据)
            if os.path.exists(save_path):
                existing_df = pd.read_csv(save_path)
                final_df = pd.concat([existing_df, result_df]).drop_duplicates(subset=['Tissue', 'Views'], keep='last')
            else:
                final_df = result_df

            final_df.to_csv(save_path, index=False)

            # 保存 AnnData
            h5ad_save_path = os.path.join(base_output_dir, f'TS_{tissue}_Hisemble_Result.h5ad')
            adata.write(h5ad_save_path, compression="gzip")
            
            print(f"\n🎉 {tissue} 处理完成！")
            print(f"🏆 【Benchmark 对比】:")
            print(f"   - PCA_only : ARI={metrics_pca['ARI']:.4f}, NMI={metrics_pca['NMI']:.4f}")
            print(f"   - scVI_only: ARI={metrics_scvi['ARI']:.4f}, NMI={metrics_scvi['NMI']:.4f}")
            print(f"   - Hisemble : ARI={metrics_hisemble['ARI']:.4f}, NMI={metrics_hisemble['NMI']:.4f}")
            print(f"💾 数据已保存至: {h5ad_save_path}")

        except FileNotFoundError:
            print(f"❌ 数据加载失败: 找不到 {file_path}。请确保预处理步骤已完成！")
        except Exception as e:
            print(f"❌ 处理 {tissue} 时发生错误: {e}")
            import traceback
            traceback.print_exc()
            continue

if __name__ == '__main__':
    main()
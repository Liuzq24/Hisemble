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

        np.random.seed(random_seed) # 确保argsort在平局时行为一致
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
    np.random.seed(random_seed) # 确保argpartition行为一致
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
    """
    计算矩阵每一行第 k 大和第 k+1 大元素的差值的均值。
    用于衡量近邻结构的“陡峭度” (Eigengap/Spectral Gap 类似概念)。
    """
    gaps = []
    n_rows = matrix.shape[0]
    
    # 为了处理方便，逐行检查。
    # 注意：如果矩阵非常巨大，转 dense 可能会慢，但在一般分析场景可接受
    # 如果是稀疏矩阵，需要考虑隐式的 0 是否在前 k+1 大范围内
    
    for i in range(n_rows):
        if issparse(matrix):
            # 获取该行所有数据（含隐式0）
            # 如果非零元素远少于 k，为了准确性还是建议看作 dense 向量处理排序
            row = matrix[i, :].toarray().flatten()
        else:
            row = matrix[i, :]
            
        # 我们需要第 k 大和第 k+1 大 (1-based index)
        # 也就是排序后索引为 -(k) 和 -(k+1) 的元素
        # 使用 partition 提高效率，只找最大的 k+1 个
        if len(row) > k:
            # partition 将第 -(k+1) 大的元素放在该位置，右边是最大的 k+1 个元素
            partitioned = np.partition(row, -(k + 1))
            top_k_plus_1 = partitioned[-(k + 1):]
            top_k_plus_1.sort() # 升序: [第k+1大, 第k大, ..., 第1大]
            
            # top_k_plus_1[0] 是第 k+1 大
            # top_k_plus_1[1] 是第 k 大
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

    # 用于存储上一轮的 Q 矩阵
    prev_Q_list = [None] * num_views
    convergence_data = []
    affinity_at_checkpoint = None

    for iter_num in range(n_iterations):
        print(f"  - 第 {iter_num + 1}/{n_iterations} 次迭代...")
        P_next_list = []
        
        # --- 步骤 1 & 2 & 3：在更新前，先计算共识 Q 并监控其变化 ---
        for i in range(num_views):
            # A. 计算当前时刻 t 的共识 Q_t
            sum_P_others = csr_matrix((n_samples, n_samples), dtype=np.float32)
            for j in range(num_views):
                if i != j:
                    sum_P_others += P_list[j]
            Q_t = sum_P_others / (num_views - 1)
        
            avg_P_others = sum_P_others / (num_views - 1)
            
            P_updated = P_list[i] @ avg_P_others @ P_list[i].T 
            P_updated_sparsified = sparsify_matrix(P_updated, k_retained, random_seed=random_seed)
            P_updated_norm = normalize_sparse(P_updated_sparsified)
            
            diff = np.abs(P_updated_norm - P_list[i]).sum() / (n_samples * k_retained)
            # B. 计算监控指标：对比 Q_t 和 Q_{t-1}
            delta_Q = 0.0
            hamming_dist = 0
            if prev_Q_list[i] is not None:
                # Frobenius 范数: ||Q_t - Q_{t-1}||_F
                delta_Q = sparse_norm(Q_t - prev_Q_list[i], ord='fro')
            
            # C. 记录本轮数据
            # 这里的 Relative Change 记录的是上一轮更新带来的影响 (P_t vs P_{t-1} 的概念)
            # 但关键是 Delta_Q 和 Hamming 完全对应公式 ||Q_t - Q_{t-1}||
            convergence_data.append({
                "Dataset": dataname,
                "Iteration": iter_num + 1,
                "View": view_names[i],
                "Relative Change": diff,
                "Delta_Q": delta_Q
            })
            
            # E. 为下一轮存下当前的 Q_t
            prev_Q_list[i] = Q_t.copy()
            P_next_list.append(P_updated_norm)
            
        P_list = P_next_list

        # <<< NEW: 检查是否到达了需要记录指标的迭代次数 >>>
        if checkpoint_iter is not None and (iter_num + 1) == checkpoint_iter:
            print(f"  - 在第 {checkpoint_iter} 次迭代处捕获融合矩阵用于评估...")
            P_checkpoint_sum = csr_matrix((n_samples, n_samples), dtype=np.float32)
            for P in P_list:
                P_checkpoint_sum += P
            P_checkpoint_sum /= num_views
            affinity_at_checkpoint = (P_checkpoint_sum + P_checkpoint_sum.T) / 2

    # 最终融合 (在所有迭代结束后)
    final_P = csr_matrix((n_samples, n_samples), dtype=np.float32)
    for P in P_list:
        final_P += P
    final_P /= num_views
    final_affinity = (final_P + final_P.T) / 2
    
    # <<< MODIFIED: 如果检查点从未被触发（例如 checkpoint_iter > n_iterations），则默认使用最终矩阵 >>>
    if affinity_at_checkpoint is None and checkpoint_iter is not None:
        print(f"  - 警告: 指定的检查点 {checkpoint_iter} 未达到。将使用最终矩阵进行评估。")
        affinity_at_checkpoint = final_affinity

    # <<< MODIFIED: 返回三个值 >>>
    return final_affinity, affinity_at_checkpoint, pd.DataFrame(convergence_data)



# --- 主执行逻辑 (最终优化版) ---
def main():
    parser = argparse.ArgumentParser(
        description="为单细胞数据集运行SNF算法。默认处理代码中定义的完整列表，也可通过命令行指定部分数据集。",
        formatter_class=argparse.RawTextHelpFormatter # 美化帮助信息格式
    )
    parser.add_argument(
        '--datanames',
        type=str, 
        nargs='+',
        required=False,
        help="要处理的一个或多个数据集的名称（用空格隔开）。\n如果未提供此参数，将默认处理代码中定义的完整列表。"
    )
    args = parser.parse_args()
    
    # 固定超参数和随机种子
    K_NEIGHBORS = 15
    N_ITERATIONS = 50 # 总迭代次数
    METRICS_CHECKPOINT_ITER = 20 # 在这次迭代后计算聚类指标
    RANDOM_SEED = 42
    
    # 设置全局随机种子
    np.random.seed(RANDOM_SEED)
    tf.random.set_seed(RANDOM_SEED)
    os.environ['PYTHONHASHSEED'] = str(RANDOM_SEED)

    all_datanames_predefined = [
        'Chen-2019-ATAC'
    ]

    all_convergence_dfs = []
    # 决定最终要运行的列表
    if args.datanames:
        # 如果用户从命令行提供了名称，则使用用户提供的列表
        datanames_to_process = args.datanames
        print(f"检测到命令行输入，将只处理以下 {len(datanames_to_process)} 个数据集:")
        print(datanames_to_process)
    else:
        # 否则，使用代码中预定义的完整列表
        datanames_to_process = all_datanames_predefined
        print(f"未指定数据集，将处理代码中预定义的 {len(datanames_to_process)} 个数据集。")

    # --- 3. 路径管理 ---
    base_output_dir = '__HISEMBLE_RESULTS_ROOT__/Hisemble/'

    os.makedirs(base_output_dir, exist_ok=True)
    metrics_save_path = os.path.join(base_output_dir, 'all_datasets_metrics_summary.csv')
    conv_save_path = os.path.join(base_output_dir, 'all_convergence_results.csv')
    
    # --- 4. 循环处理选定的数据集 ---
    for dataname in datanames_to_process:
        print(f"\n{'='*20} 开始处理数据集: {dataname} {'='*20}")
        print(f"固定参数: K={K_NEIGHBORS}, N_iterations={N_ITERATIONS}, Seed={RANDOM_SEED}")

        try:
            # --- 数据加载 ---
            with open(f'__HISEMBLE_DATA_ROOT__/ADATA/{dataname}_scBasset.pkl', 'rb') as f:
                ad_scBasset = pickle.load(f)
            with open(f'__HISEMBLE_DATA_ROOT__/ADATA/{dataname}_snap.pkl', 'rb') as f:
                ad_snap = pickle.load(f)
            with open(f'__HISEMBLE_DATA_ROOT__/lsi_results/ADATA/{dataname}_lsi.pkl', 'rb') as f:
                ad_lsi = pickle.load(f)
            
            true_labels = ad_snap.obs.cell_type.to_numpy()
            true_cell_types = ad_snap.obs['cell_type'].nunique()
            
            # --- 核心计算流程 (此处代码与上一版完全相同) ---
            # a) 构建初始图
            print("\n--- 构建初始稀疏亲和度矩阵 ---")
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
                # 移除当前数据集旧的数据（如果有），防止重复跑导致行数无限增加
                existing_conv = existing_conv[existing_conv['Dataset'] != dataname]
                final_conv_df = pd.concat([existing_conv, conv_df], ignore_index=True)
            else:
                final_conv_df = conv_df
            final_conv_df.to_csv(conv_save_path, index=False)
            print(f"  - 收敛数据已增量更新至: {conv_save_path}")
            
            # c) 聚类和评估
            print("\n--- 后续聚类与评估 ---")
            # ... (此处省略了聚类和评估的详细代码，与上一版相同) ...
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
            
            # --- 结果汇总 ---
            result_data = {
                'dataname': [dataname],
                'k_neighbors': [K_NEIGHBORS],
                'n_iterations_total': [N_ITERATIONS],
                'n_iterations_metrics': [METRICS_CHECKPOINT_ITER], # <<< NEW: 记录用于评估的迭代次数
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
            print(f"  - 指标数据已增量更新至: {metrics_save_path}")
            
            if all_convergence_dfs:
                full_convergence_df = pd.concat(all_convergence_dfs)
                convergence_save_path = os.path.join(base_output_dir, "all_convergence_data.csv")
                full_convergence_df.to_csv(convergence_save_path, index=False)
                print(f"\n所有视图的收敛过程数据已保存至: {convergence_save_path}")

        except FileNotFoundError as e:
            print(f"!!!!!! 数据加载失败: {e}。请检查文件路径。!!!!!!")
        except Exception as e:
            print(f"!!!!!! 处理数据集 {dataname} 时发生严重错误: {e} !!!!!!")
            continue


if __name__ == '__main__':
    main()
import pandas as pd
import numpy as np
import sklearn
import scipy.sparse as sp
from scipy.sparse import issparse, csr_matrix
import scanpy as sc
import anndata
import matplotlib.pyplot as plt
import os
import time
import psutil
import threading
import argparse
import warnings
from joblib import Parallel, delayed

# 禁用不必要的警告
warnings.filterwarnings("ignore")
import tensorflow as tf
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score, normalized_mutual_info_score, homogeneity_score
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

# ==============================================================================
# SC3 核心工具类与算法
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
            try:
                mem = self.process.memory_info().rss
                # 获取 joblib 子进程的内存
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
        print(f"算法峰值内存使用: {self.peak_mb:.2f} MB")


def sparsify_matrix_from_dense(dense_matrix, k_neighbors: int) -> csr_matrix:
    """将密集的共识矩阵转换为稀疏矩阵，仅保留每行最大的K个值，用于 UMAP"""
    n_samples = dense_matrix.shape[0]
    indices = np.argsort(-dense_matrix, axis=1)[:, :k_neighbors]
    row_indices = np.repeat(np.arange(n_samples), k_neighbors)
    col_indices = indices.flatten()
    data = dense_matrix[np.arange(n_samples)[:, None], indices].flatten()
    sparse_mat = csr_matrix((data, (row_indices, col_indices)), shape=(n_samples, n_samples))
    return (sparse_mat + sparse_mat.T) / 2


def run_one_kmeans(data, k, seed):
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=seed, algorithm='elkan')
    return kmeans.fit_predict(data)


def run_sc3_workflow(embeddings, n_clusters, n_runs=20, k_neighbors_for_graph=15):
    """
    通用 SC3 工作流：
    如果 embeddings 列表只有一个元素，就是单视图 SC3；
    如果有多个元素，就是多视图融合 SC3。
    """
    N = embeddings[0].shape[0]
    num_views = len(embeddings)
    print(f"  > 视图数={num_views}, 细胞数 N={N}, 目标簇数 k={n_clusters}, 每次运行={n_runs}")

    # 1. 并行 K-Means
    print(f"  - 正在运行并行 K-Means (总任务数={num_views * n_runs})...")
    tasks = []
    for i, emb in enumerate(embeddings):
        for r in range(n_runs):
            tasks.append((emb, r + 42))
    
    results = Parallel(n_jobs=-1)(delayed(run_one_kmeans)(emb, n_clusters, seed) for emb, seed in tasks)
    label_matrix = np.array(results).T 
    
    # 2. 共识矩阵 (Consensus Matrix)
    print("  - 正在构建共识矩阵...")
    consensus_matrix = np.zeros((N, N), dtype=np.float32)
    n_total_runs = label_matrix.shape[1]
    
    for i in range(n_total_runs):
        lbl = label_matrix[:, i]
        mat = (lbl[:, None] == lbl[None, :])
        consensus_matrix += mat
    
    consensus_matrix /= n_total_runs

    # 3. 层次聚类 (Hierarchical Clustering)
    print("  - 正在进行层次聚类...")
    dist_mat = 1.0 - consensus_matrix
    dist_mat = np.clip(dist_mat, 0, 1) 
    dist_vec = squareform(dist_mat, checks=False)
    
    Z = linkage(dist_vec, method='average')
    labels = fcluster(Z, t=n_clusters, criterion='maxclust')
    
    # 4. 提取 UMAP 可视化所需稀疏图
    print("  - 生成稀疏图用于可视化...")
    sparse_graph = sparsify_matrix_from_dense(consensus_matrix, k_neighbors=k_neighbors_for_graph)
    
    return labels, sparse_graph


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
        description="为 scRNA-seq 数据集运行 SC3 双视图融合聚类 (PCA + scVI)，同时评估单一视图作为 Benchmark。",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        '--tissues',  
        type=str, 
        nargs='+',      
        required=False, 
        help="要处理的 Tabula Sapiens 组织名称（用空格隔开）。\n未提供则处理默认的 6 个组织。"
    )
    args = parser.parse_args()
    
    # --- 固定超参数 ---
    N_RUNS = 20        
    K_NEIGHBORS = 15   
    RANDOM_SEED = 42
    
    np.random.seed(RANDOM_SEED)
    tf.random.set_seed(RANDOM_SEED)
    os.environ['PYTHONHASHSEED'] = str(RANDOM_SEED)

    # --- 数据输入输出路径 ---
    data_dir = "Tabula_Sapiens"
    base_output_dir = "results/SC3_RNA"
    os.makedirs(base_output_dir, exist_ok=True)
    
    save_path = os.path.join(base_output_dir, 'sc3_scRNA_benchmark_results.csv')
    save_folder_umap = os.path.join(base_output_dir, 'UMAP_plots')
    os.makedirs(save_folder_umap, exist_ok=True)
    
    print(f"UMAP 图像将被保存到: {save_folder_umap}")

    # --- 决定处理列表 ---
    all_tissues_predefined = ["Pancreas", "Liver"]
    tissues_to_process = args.tissues if args.tissues else all_tissues_predefined

    for tissue in tissues_to_process:
        print(f"\n{'='*25} 开始处理组织: {tissue} (SC3 模型) {'='*25}")
        
        file_path = os.path.join(data_dir, f"TS_{tissue}_30k_processed.h5ad")
        
        try:
            # 1. 数据加载
            print(f"正在加载数据: {file_path}")
            adata = sc.read_h5ad(file_path)
            
            if 'cell_type' not in adata.obs:
                raise ValueError(f"数据 {tissue} 缺少 'cell_type' 标注，无法进行评估！")
                
            true_labels = adata.obs['cell_type'].astype(str).to_numpy()
            true_cell_types = adata.obs['cell_type'].nunique()
            print(f"发现 {adata.n_obs} 个细胞, {true_cell_types} 种真实细胞类型。")

            emb_pca = adata.obsm['X_pca']
            emb_scvi = adata.obsm['X_scvi']

            # 2. 单一模态聚类评估 (Benchmark)
            print("\n--- [SC3 单视图]: PCA ---")
            with MemoryMonitor() as monitor_pca:
                start_pca = time.time()
                clusters_pca, _ = run_sc3_workflow([emb_pca], true_cell_types, N_RUNS, K_NEIGHBORS)
                time_pca = time.time() - start_pca
                mem_pca = monitor_pca.peak_mb

            print("\n--- [SC3 单视图]: scVI ---")
            with MemoryMonitor() as monitor_scvi:
                start_scvi = time.time()
                clusters_scvi, _ = run_sc3_workflow([emb_scvi], true_cell_types, N_RUNS, K_NEIGHBORS)
                time_scvi = time.time() - start_scvi
                mem_scvi = monitor_scvi.peak_mb

            # 3. 运行 SC3 双视图融合
            print("\n--- [SC3 融合网络]: PCA + scVI ---")
            with MemoryMonitor() as monitor_fused:
                start_fused = time.time()
                clusters_sc3, consensus_graph = run_sc3_workflow([emb_pca, emb_scvi], true_cell_types, N_RUNS, K_NEIGHBORS)
                time_fused = time.time() - start_fused
                mem_fused = monitor_fused.peak_mb

            # 4. 更新 Adata 与 UMAP (基于融合后的图)
            consensus_graph = consensus_graph.astype(np.float32)
            adata.obsp['connectivities'] = consensus_graph
            
            distances = consensus_graph.copy()
            distances.data = 1.0 - distances.data
            distances.data[distances.data < 0] = 0
            adata.obsp['distances'] = distances

            # 欺骗 Scanpy
            adata.uns['neighbors'] = {
                'params': {'n_neighbors': K_NEIGHBORS, 'method': 'umap', 'metric': 'precomputed'},
                'connectivities_key': 'connectivities',
                'distances_key': 'distances'
            }

            print("\n正在计算基于 SC3 共识矩阵的 UMAP...")
            sc.tl.umap(adata, min_dist=0.5, spread=1.0, random_state=RANDOM_SEED)
            
            # 记录类别为 Categorical (绘图表现更好)
            adata.obs['PCA_SC3_Cluster'] = pd.Categorical(clusters_pca)
            adata.obs['scVI_SC3_Cluster'] = pd.Categorical(clusters_scvi)
            adata.obs['Fused_SC3_Cluster'] = pd.Categorical(clusters_sc3)

            # 5. 画图并保存
            plt.rcParams.update({
                'font.size': 14, 'axes.titlesize': 18, 'axes.labelsize': 16,
                'axes.titleweight': 'bold', 'axes.labelweight': 'bold',
                'text.color': 'black', 'axes.labelcolor': 'black',
                'xtick.color': 'black', 'ytick.color': 'black',
            })
            
            plot_targets = [
                ('cell_type', 'True Cell Types'),
                ('PCA_SC3_Cluster', 'PCA (SC3) Clusters'),
                ('scVI_SC3_Cluster', 'scVI (SC3) Clusters'),
                ('Fused_SC3_Cluster', 'Fused SC3 Clusters')
            ]
            
            for col, title in plot_targets:
                sc.pl.umap(adata, color=[col], show=False, legend_fontsize=12, title=f"{tissue} - {title}")
                plt.savefig(os.path.join(save_folder_umap, f'UMAP_{tissue}_{col}.pdf'), dpi=300, format="pdf", bbox_inches='tight')
                plt.close()

            # 6. 计算并汇总所有视图的指标
            metrics_pca = calculate_metrics(true_labels, clusters_pca)
            metrics_scvi = calculate_metrics(true_labels, clusters_scvi)
            metrics_fused = calculate_metrics(true_labels, clusters_sc3)
            
            results = [
                {'Tissue': tissue, 'Algorithm': 'SC3', 'Views': 'PCA_only', 'Runs': N_RUNS, **metrics_pca, 'Time_s': time_pca, 'Peak_Mem_MB': mem_pca},
                {'Tissue': tissue, 'Algorithm': 'SC3', 'Views': 'scVI_only', 'Runs': N_RUNS, **metrics_scvi, 'Time_s': time_scvi, 'Peak_Mem_MB': mem_scvi},
                {'Tissue': tissue, 'Algorithm': 'SC3', 'Views': 'PCA+scVI', 'Runs': N_RUNS * 2, **metrics_fused, 'Time_s': time_fused, 'Peak_Mem_MB': mem_fused}
            ]
            result_df = pd.DataFrame(results)
            
            # 更新 CSV 记录表 
            if os.path.exists(save_path):
                existing_df = pd.read_csv(save_path)
                final_df = pd.concat([existing_df, result_df]).drop_duplicates(subset=['Tissue', 'Algorithm', 'Views'], keep='last')
            else:
                final_df = result_df

            final_df.to_csv(save_path, index=False)

            # 保存 AnnData
            h5ad_save_path = os.path.join(base_output_dir, f'TS_{tissue}_SC3_Result.h5ad')
            adata.write(h5ad_save_path, compression="gzip")
            
            print(f"\n🎉 {tissue} SC3 处理完成！")
            print(f"🏆 【SC3 Benchmark 对比】:")
            print(f"   - PCA_only : ARI={metrics_pca['ARI']:.4f}, NMI={metrics_pca['NMI']:.4f} (耗时: {time_pca:.1f}s)")
            print(f"   - scVI_only: ARI={metrics_scvi['ARI']:.4f}, NMI={metrics_scvi['NMI']:.4f} (耗时: {time_scvi:.1f}s)")
            print(f"   - Fused SC3: ARI={metrics_fused['ARI']:.4f}, NMI={metrics_fused['NMI']:.4f} (耗时: {time_fused:.1f}s)")
            print(f"💾 数据已保存至: {h5ad_save_path}")

        except Exception as e:
            print(f"❌ 处理 {tissue} 时发生错误: {e}")
            import traceback
            traceback.print_exc()
            continue

if __name__ == '__main__':
    main()
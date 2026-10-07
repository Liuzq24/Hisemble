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
# --- 资源监控类 (保持不变) ---
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
                # 尝试获取子进程内存（因为SC3用了joblib并行）
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

# --- 辅助函数：用于将SC3的共识矩阵转为稀疏图以便画UMAP ---
def sparsify_matrix_from_dense(dense_matrix, k_neighbors: int) -> csr_matrix:
    """
    将密集的共识矩阵转换为稀疏矩阵，仅保留每行最大的K个值。
    """
    n_samples = dense_matrix.shape[0]
    # 使用 argsort 找到每行最大的 k 个索引
    # 注意：SC3共识矩阵值越大越相似
    indices = np.argsort(-dense_matrix, axis=1)[:, :k_neighbors]
    
    row_indices = np.repeat(np.arange(n_samples), k_neighbors)
    col_indices = indices.flatten()
    data = dense_matrix[np.arange(n_samples)[:, None], indices].flatten()
    
    sparse_mat = csr_matrix((data, (row_indices, col_indices)), shape=(n_samples, n_samples))
    return (sparse_mat + sparse_mat.T) / 2

# --- SC3 核心逻辑 ---
def run_one_kmeans(data, k, seed):
    # n_init=10 这里的n_init可以适当降低以提高速度，原始SC3非常慢
    # algorithm='elkan' 稍微快一点
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=seed, algorithm='elkan')
    return kmeans.fit_predict(data)

def run_sc3_workflow(embeddings, n_clusters, n_runs=20, k_neighbors_for_graph=15):
    N = embeddings[0].shape[0]
    print(f"  > [Debug] 细胞数量 N={N}, 目标簇数 k={n_clusters}")

    # 1. K-Means
    print(f"  - 正在运行 K-Means (Runs={n_runs})...")
    tasks = []
    for i, emb in enumerate(embeddings):
        # 简单检查 embedding 形状
        if emb.shape[0] != N:
            raise ValueError(f"Embedding {i} 维度不匹配: {emb.shape}")
        for r in range(n_runs):
            tasks.append((emb, r + 42))
    
    results = Parallel(n_jobs=-1)(delayed(run_one_kmeans)(emb, n_clusters, seed) for emb, seed in tasks)
    label_matrix = np.array(results).T 
    
    # 2. Consensus Matrix
    print("  - 正在构建共识矩阵...")
    consensus_matrix = np.zeros((N, N), dtype=np.float32)
    n_total_runs = label_matrix.shape[1]
    
    for i in range(n_total_runs):
        lbl = label_matrix[:, i]
        mat = (lbl[:, None] == lbl[None, :])
        consensus_matrix += mat
    
    consensus_matrix /= n_total_runs
    
    # Debug: 检查共识矩阵是否有意义
    print(f"  > [Debug] 共识矩阵范围: Min={consensus_matrix.min():.4f}, Max={consensus_matrix.max():.4f}, Mean={consensus_matrix.mean():.4f}")
    if consensus_matrix.mean() < 0.001 or consensus_matrix.mean() > 0.999:
        print("  > [Warning] 共识矩阵似乎是空的或全满的，聚类可能失败！")

    # 3. Hierarchy
    print("  - 正在进行层次聚类...")
    # 距离矩阵：确保在 [0, 1] 之间
    dist_mat = 1.0 - consensus_matrix
    dist_mat = np.clip(dist_mat, 0, 1) 
    
    # Squareform: 将方阵转为 compressed vector，linkage 需要这个
    dist_vec = squareform(dist_mat, checks=False)
    
    Z = linkage(dist_vec, method='average')
    labels = fcluster(Z, t=n_clusters, criterion='maxclust')
    
    print(f"  > [Debug] 生成的簇标签种类数: {len(np.unique(labels))}")
    print(f"  > [Debug] 前20个标签: {labels[:20]}")

    # 4. Graph for UMAP
    print("  - 生成稀疏图用于可视化...")
    sparse_graph = sparsify_matrix_from_dense(consensus_matrix, k_neighbors=k_neighbors_for_graph)
    
    return labels, sparse_graph


# --- 主执行逻辑 ---
def main():
    # --- 1. 参数解析 ---
    parser = argparse.ArgumentParser(description="运行 SC3 (Consensus Clustering) 工作流")
    parser.add_argument('--datanames', type=str, nargs='+', required=False, 
                        help="要处理的数据集名称")
    args = parser.parse_args()
    
    # 固定超参数
    N_RUNS = 20        # 每个视图运行 K-means 的次数
    K_NEIGHBORS = 15   # 仅用于最后生成 UMAP 的稀疏图构建
    RANDOM_SEED = 42
    
    np.random.seed(RANDOM_SEED)
    os.environ['PYTHONHASHSEED'] = str(RANDOM_SEED)

    # 预定义数据集
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

    # --- 路径管理 ---
    # 修改输出路径，避免覆盖 Hisemble 的结果
    base_output_dir = '__HISEMBLE_RESULTS_ROOT__/SC3_final/'
    os.makedirs(base_output_dir, exist_ok=True)
    save_path = os.path.join(base_output_dir, f'sc3_results.csv')

    save_folder_umap = os.path.join(base_output_dir, 'UMAP_plots')
    os.makedirs(save_folder_umap, exist_ok=True)
    
    # --- 循环处理 ---
    for dataname in datanames_to_process:
        print(f"\n{'='*20} 处理: {dataname} {'='*20}")
        try:
            # --- 加载数据并检查对齐 (CRITICAL FIX) ---
            path_basset = f'__HISEMBLE_DATA_ROOT__/ADATA/{dataname}_scBasset.pkl'
            path_snap = f'__HISEMBLE_DATA_ROOT__/ADATA/{dataname}_snap.pkl'
            path_lsi = f'__HISEMBLE_DATA_ROOT__/lsi_results/ADATA/{dataname}_lsi.pkl'

            with open(path_basset, 'rb') as f: ad_scBasset = pickle.load(f)
            with open(path_snap, 'rb') as f: ad_snap = pickle.load(f)
            with open(path_lsi, 'rb') as f: ad_lsi = pickle.load(f)

            print(f"  > [Check] scBasset shape: {ad_scBasset.shape}")
            print(f"  > [Check] snap shape:     {ad_snap.shape}")
            print(f"  > [Check] lsi shape:      {ad_lsi.shape}")

            # 强校验：确保细胞名完全一致且顺序一致
            if not (ad_scBasset.obs_names.equals(ad_snap.obs_names) and ad_snap.obs_names.equals(ad_lsi.obs_names)):
                print("  !!!!!! 严重错误: 三个数据的细胞顺序(obs_names)不一致 !!!!!!")
                print("  这会导致共识矩阵计算出随机噪声，ARI=0。尝试自动对齐...")
                
                # 尝试取交集并对齐
                common_cells = ad_scBasset.obs_names.intersection(ad_snap.obs_names).intersection(ad_lsi.obs_names)
                print(f"  > 对齐后细胞数: {len(common_cells)}")
                ad_scBasset = ad_scBasset[common_cells].copy()
                ad_snap = ad_snap[common_cells].copy()
                ad_lsi = ad_lsi[common_cells].copy()
            else:
                print("  > 数据对齐检查通过。")

            true_labels = ad_snap.obs.cell_type.to_numpy()
            true_cell_types = ad_snap.obs['cell_type'].nunique()
            
            embeddings = [
                ad_scBasset.obsm['projection'],
                ad_snap.obsm['X_spectral'],
                ad_lsi.obsm['X_lsi']
            ]

            # --- 运行 SC3 ---
            with MemoryMonitor() as monitor:
                start_time = time.time()
                clusters, consensus_graph = run_sc3_workflow(
                    embeddings, true_cell_types, N_RUNS, K_NEIGHBORS
                )
                end_time = time.time()

            # --- 构造 Adata 用于绘图 (修复 UMAP 警告) ---
            # 重新读取一个模板 adata (为了保存 metadata)
            # 注意：如果上面做了对齐裁切，这里必须也要裁切
            adata_temp = sc.read_h5ad('__HISEMBLE_DATA_ROOT__/0.01_h5ad/%s_processed.h5ad'%dataname)
            adata = adata_temp[ad_snap.obs_names].copy() # 确保和计算用的细胞一致

            # 注入 SC3 图
            consensus_graph = consensus_graph.astype(np.float32)
            adata.obsp['connectivities'] = consensus_graph
            
            # 构造距离
            distances = consensus_graph.copy()
            distances.data = 1.0 - distances.data
            distances.data[distances.data < 0] = 0
            adata.obsp['distances'] = distances

            # 欺骗 Scanpy，让它以为这是一个标准的邻接图
            adata.uns['neighbors'] = {
                'params': {
                    'n_neighbors': K_NEIGHBORS, 
                    'method': 'umap', # 关键：欺骗 Scanpy
                    'metric': 'precomputed'
                },
                'connectivities_key': 'connectivities',
                'distances_key': 'distances'
            }

            # 运行 UMAP (现在应该会快很多，且不会有那个警告)
            print("  - 计算 UMAP...")
            sc.tl.umap(adata, min_dist=0.5, spread=1.0, random_state=RANDOM_SEED)

            adata.obs['SC3'] = clusters.astype(str) # 转为字符型方便画图
            
            # 保存图
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
            print(f"  > 结果: ARI={ARI:.4f}, NMI={NMI:.4f}, Time={end_time-start_time:.2f}s")

            output_columns = ['dataname', 'n_runs', 'ARI', 'AMI', 'NMI', 'HOM', 'Time_s', 'Mem_MB']
            
            # 2. 构造字典
            res_data = {
                'dataname': dataname, 
                'n_runs': N_RUNS,   # 之前缺失了这个字段
                'ARI': ARI, 
                'AMI': AMI,
                'NMI': NMI,
                'HOM': HOM,
                'Time_s': end_time - start_time, 
                'Mem_MB': monitor.peak_mb
            }
            
            res_df = pd.DataFrame([res_data])
            # 3. 强制重排列，确保顺序一致
            res_df = res_df[output_columns]

            # 4. 使用文件锁安全写入 (防止多进程同时写乱文件)
            with open(save_path, 'a') as f:
                fcntl.flock(f, fcntl.LOCK_EX) # 加锁
                try:
                    # 检查文件是否为空来决定是否写Header
                    is_empty = os.path.getsize(save_path) == 0
                    res_df.to_csv(f, header=is_empty, index=False)
                finally:
                    fcntl.flock(f, fcntl.LOCK_UN) # 解锁
            
            # 保存 h5ad
            adata.write(os.path.join(base_output_dir, f'{dataname}_result.h5ad'), compression='gzip')

        except Exception as e:
            print(f"!!!!!! 出错: {e}")
            import traceback
            traceback.print_exc()

if __name__ == '__main__':
    main()
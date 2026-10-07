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
from scipy.linalg import eigh
from scipy.cluster.vq import kmeans2
import time
import psutil
import threading
import argparse
import os
from joblib import Parallel, delayed
import fcntl

# --- 资源监控类 ---
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

# --- 辅助函数：稀疏化矩阵用于 UMAP ---
def sparsify_matrix_from_dense(dense_matrix, k_neighbors: int) -> csr_matrix:
    n_samples = dense_matrix.shape[0]
    # 相似度矩阵：值越大越相似，取负号argsort
    indices = np.argsort(-dense_matrix, axis=1)[:, :k_neighbors]
    row_indices = np.repeat(np.arange(n_samples), k_neighbors)
    col_indices = indices.flatten()
    data = dense_matrix[np.arange(n_samples)[:, None], indices].flatten()
    sparse_mat = csr_matrix((data, (row_indices, col_indices)), shape=(n_samples, n_samples))
    return (sparse_mat + sparse_mat.T) / 2

# --- CHAI 核心组件 ---

def dense_snf(wall_list, k=20, t=20):
    """
    CHAI (SNF) 使用的密集矩阵融合算法
    """
    def normalize(w):
        w = w.copy()
        row_sums = w.sum(axis=1)
        row_sums[row_sums == 0] = 1 
        p = w / row_sums[:, np.newaxis]
        p = (p + p.T) / 2 
        return p

    def dominate_set(w, k):
        n = w.shape[0]
        s = np.zeros_like(w)
        for i in range(n):
            indices = np.argsort(w[i, :])[-k:]
            s[i, indices] = w[i, indices]
        return normalize(s)

    P = [normalize(w) for w in wall_list]
    S = [dominate_set(w, k) for w in wall_list]
    num_views = len(wall_list)
    
    for _ in range(t):
        P_next = []
        for v in range(num_views):
            others = [P[i] for i in range(num_views) if i != v]
            P_sum = sum(others) / (num_views - 1)
            P_new = S[v].dot(P_sum).dot(S[v].T)
            P_next.append(P_new)
        P = P_next

    final_matrix = sum(P) / num_views
    
    # [Fix] 这里的数值清理非常关键，防止NaN传入后续步骤
    final_matrix = np.nan_to_num(final_matrix, nan=0.0, posinf=0.0, neginf=0.0)
    # 归一化到 0-1 之间
    max_val = final_matrix.max()
    if max_val > 0:
        final_matrix /= max_val
        
    return final_matrix

def run_spectral_clustering_njw(affinity_matrix, n_clusters):
    """
    NJW (2002) 谱聚类算法
    """
    row_sums = np.array(affinity_matrix.sum(axis=1)).flatten()
    row_sums[row_sums == 0] = 1e-10
    d_inv_sqrt = np.power(row_sums, -0.5)
    
    # L = D^-1/2 * W * D^-1/2
    L = affinity_matrix * d_inv_sqrt[:, np.newaxis] * d_inv_sqrt[np.newaxis, :]
    
    N = L.shape[0]
    # eigh 针对对称矩阵，获取最大的 n_clusters 个特征值
    evals, evecs = eigh(L, subset_by_index=[N-n_clusters, N-1])
    
    # 行归一化
    row_norms = np.linalg.norm(evecs, axis=1)
    row_norms[row_norms == 0] = 1e-10
    X_normalized = evecs / row_norms[:, np.newaxis]
    
    # K-Means on eigenvectors
    centroids, labels = kmeans2(X_normalized, n_clusters, minit='++')
    
    return labels

def run_chai_workflow(embeddings, n_clusters, method='AvgSim', k_neighbors_for_graph=15):
    """
    CHAI 工作流:
    1. 对每个 Embedding 做 K-Means
    2. 构建二进制关联矩阵
    3. 融合 (AvgSim 或 SNF)
    4. 谱聚类
    """
    N = embeddings[0].shape[0]
    print(f"  > [CHAI] Method={method}, Cells={N}, Clusters={n_clusters}")

    # 1. 对每个视图运行 K-Means
    # CHAI 默认逻辑是对每个模态产生一个分区
    labels_list = []
    print("  - 正在对各个视图运行 K-Means...")
    
    # 并行运行以加速
    def run_k_means_wrapper(emb, k):
        return KMeans(n_clusters=k, n_init=10, algorithm='elkan').fit_predict(emb)
    
    labels_list = Parallel(n_jobs=-1)(
        delayed(run_k_means_wrapper)(emb, n_clusters) for emb in embeddings
    )

    # 2. 构建二进制相似度矩阵 (N x N)
    print("  - 构建二进制关联矩阵...")
    binary_matrices = []
    for lbl in labels_list:
        # 广播计算：相等为1，不等为0
        mat = (lbl[:, None] == lbl[None, :]).astype(np.float32)
        binary_matrices.append(mat)
        
    # 3. 融合
    print(f"  - 执行融合 ({method})...")
    if method == 'AvgSim':
        # AvgSim: 直接平均
        fused_matrix = sum(binary_matrices) / len(binary_matrices)
    elif method == 'SNF':
        # SNF: 密集矩阵 SNF 迭代
        # 注意：这里调用 dense_snf，非常消耗计算资源
        fused_matrix = dense_snf(binary_matrices, k=20, t=20)
    else:
        raise ValueError(f"Unknown method: {method}")

    # 4. 谱聚类
    print("  - 运行 NJW 谱聚类...")
    final_labels = run_spectral_clustering_njw(fused_matrix, n_clusters=n_clusters)
    
    # 5. 生成用于 UMAP 的稀疏图
    print("  - 生成稀疏图用于可视化...")
    sparse_graph = sparsify_matrix_from_dense(fused_matrix, k_neighbors=k_neighbors_for_graph)

    return final_labels, sparse_graph

# --- 主执行逻辑 ---
def main():
    parser = argparse.ArgumentParser(description="运行 CHAI 工作流")
    parser.add_argument('--datanames', type=str, nargs='+', required=False, help="数据集名称列表")
    parser.add_argument('--method', type=str, default='AvgSim', choices=['AvgSim', 'SNF'], 
                        help="CHAI 融合方法: AvgSim 或 SNF")
    args = parser.parse_args()
    
    # 选定方法
    CHAI_METHOD = args.method
    
    # 固定参数
    K_NEIGHBORS = 15   
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

    # 输出路径配置
    base_output_dir = f'__HISEMBLE_RESULTS_ROOT__/CHAI_{CHAI_METHOD}/'
    os.makedirs(base_output_dir, exist_ok=True)
    save_path = os.path.join(base_output_dir, f'chai_{CHAI_METHOD}_results.csv')

    save_folder_umap = os.path.join(base_output_dir, 'UMAP_plots')
    os.makedirs(save_folder_umap, exist_ok=True)
    
    for dataname in datanames_to_process:
        print(f"\n{'='*20} 处理: {dataname} ({CHAI_METHOD}) {'='*20}")
        try:
            # 加载数据 (与 SC3 脚本一致)
            path_basset = f'__HISEMBLE_DATA_ROOT__/ADATA/{dataname}_scBasset.pkl'
            path_snap = f'__HISEMBLE_DATA_ROOT__/ADATA/{dataname}_snap.pkl'
            path_lsi = f'__HISEMBLE_DATA_ROOT__/lsi_results/ADATA/{dataname}_lsi.pkl'

            with open(path_basset, 'rb') as f: ad_scBasset = pickle.load(f)
            with open(path_snap, 'rb') as f: ad_snap = pickle.load(f)
            with open(path_lsi, 'rb') as f: ad_lsi = pickle.load(f)

            # 对齐检查
            if not (ad_scBasset.obs_names.equals(ad_snap.obs_names) and ad_snap.obs_names.equals(ad_lsi.obs_names)):
                print("  !!!!!! 警告: 数据未对齐，正在取交集对齐...")
                common_cells = ad_scBasset.obs_names.intersection(ad_snap.obs_names).intersection(ad_lsi.obs_names)
                ad_scBasset = ad_scBasset[common_cells].copy()
                ad_snap = ad_snap[common_cells].copy()
                ad_lsi = ad_lsi[common_cells].copy()
            
            true_labels = ad_snap.obs.cell_type.to_numpy()
            true_cell_types = ad_snap.obs['cell_type'].nunique()
            
            embeddings = [
                ad_scBasset.obsm['projection'],
                ad_snap.obsm['X_spectral'],
                ad_lsi.obsm['X_lsi']
            ]

            # --- 运行 CHAI ---
            with MemoryMonitor() as monitor:
                start_time = time.time()
                # 关键调用: 传入 method 参数
                clusters, fused_graph = run_chai_workflow(
                    embeddings, true_cell_types, method=CHAI_METHOD, k_neighbors_for_graph=K_NEIGHBORS
                )
                end_time = time.time()

            # --- 绘图与保存 ---
            adata_temp = sc.read_h5ad('__HISEMBLE_DATA_ROOT__/0.01_h5ad/%s_processed.h5ad'%dataname)
            adata = adata_temp[ad_snap.obs_names].copy()

            fused_graph = fused_graph.astype(np.float32)
            adata.obsp['connectivities'] = fused_graph
            
            distances = fused_graph.copy()
            distances.data = 1.0 - distances.data
            distances.data[distances.data < 0] = 0
            adata.obsp['distances'] = distances

            adata.uns['neighbors'] = {
                'params': {'n_neighbors': K_NEIGHBORS, 'method': 'umap', 'metric': 'precomputed'},
                'connectivities_key': 'connectivities',
                'distances_key': 'distances'
            }

            print("  - 计算 UMAP...")
            sc.tl.umap(adata, min_dist=0.5, spread=1.0, random_state=RANDOM_SEED, init_pos='random')

            adata.obs['CHAI'] = clusters.astype(str)
            
            sc.pl.umap(adata, color=['cell_type'], show=False, title=f'{dataname} True Labels')
            plt.savefig(os.path.join(save_folder_umap, f'{dataname}_True.pdf'), bbox_inches='tight')
            plt.close()

            sc.pl.umap(adata, color=['CHAI'], show=False, title=f'{dataname} CHAI({CHAI_METHOD}) Labels')
            plt.savefig(os.path.join(save_folder_umap, f'{dataname}_CHAI.pdf'), bbox_inches='tight')
            plt.close()

            # 计算指标
            ARI = adjusted_rand_score(true_labels, clusters)
            AMI = adjusted_mutual_info_score(true_labels, clusters)
            NMI = normalized_mutual_info_score(true_labels, clusters)
            HOM = homogeneity_score(true_labels, clusters)
            print(f"  > 结果: ARI={ARI:.4f}, NMI={NMI:.4f}, Time={end_time-start_time:.2f}s")

            output_columns = ['dataname', 'Method', 'ARI', 'AMI', 'NMI', 'HOM', 'Time_s', 'Mem_MB']
            res_data = {
                'dataname': dataname, 
                'Method': f'CHAI-{CHAI_METHOD}',
                'ARI': ARI, 'AMI': AMI, 'NMI': NMI, 'HOM': HOM,
                'Time_s': end_time - start_time, 
                'Mem_MB': monitor.peak_mb
            }
            
            res_df = pd.DataFrame([res_data])[output_columns]

            with open(save_path, 'a') as f:
                fcntl.flock(f, fcntl.LOCK_EX)
                is_empty = os.path.getsize(save_path) == 0
                res_df.to_csv(f, header=is_empty, index=False)
                fcntl.flock(f, fcntl.LOCK_UN)
            
            adata.write(os.path.join(base_output_dir, f'{dataname}_result.h5ad'), compression='gzip')

        except Exception as e:
            print(f"!!!!!! 出错: {e}")
            import traceback
            traceback.print_exc()

if __name__ == '__main__':
    main()
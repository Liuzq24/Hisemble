import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering
import time
import tracemalloc
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
import os
import pandas as pd
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
from utils import cluster_evaluation, overcorrection_score #从utils里面导入
from sklearn.metrics.pairwise import rbf_kernel, euclidean_distances
from sklearn.metrics import adjusted_rand_score,adjusted_mutual_info_score,normalized_mutual_info_score, homogeneity_score
from scipy.spatial.distance import pdist, squareform
import igraph as ig
import leidenalg
from sklearn.metrics import pairwise_distances
from scipy.sparse import csr_matrix, identity as sparse_identity
from sklearn.neighbors import NearestNeighbors
import psutil
import threading
import argparse
from utils import find_best_matrix, perform_leiden_clustering, remove_duplicates, build_knn_graph_chunked_cpu, normalize_sparse, sparsify_matrix
from joblib import Parallel, delayed

K_NEIGHBORS = 15
N_ITERATIONS = 20
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
os.environ['PYTHONHASHSEED'] = str(RANDOM_SEED)

base_output_dir = '__HISEMBLE_RESULTS_ROOT__/downsampling/'
os.makedirs(base_output_dir, exist_ok=True)
save_path = os.path.join(base_output_dir, 'downsampling_SC3.csv')

dataname = 'FCA_fetal_33184180_liver'
with open(f'__HISEMBLE_DATA_ROOT__/ADATA/{dataname}_scBasset.pkl', 'rb') as f:
    ad_scBasset = pickle.load(f)
with open(f'__HISEMBLE_DATA_ROOT__/ADATA/{dataname}_snap.pkl', 'rb') as f:
    ad_snap = pickle.load(f)
with open(f'__HISEMBLE_DATA_ROOT__/lsi_results/ADATA/{dataname}_lsi.pkl', 'rb') as f:
    ad_lsi = pickle.load(f)

FULL_EMB_1 = ad_scBasset.obsm['projection']
FULL_EMB_2 = ad_snap.obsm['X_spectral']
FULL_EMB_3 = ad_lsi.obsm['X_lsi']

true_labels = ad_snap.obs.cell_type.to_numpy()
true_cell_types = ad_snap.obs['cell_type'].nunique()

class MemoryMonitor:
    def __init__(self, interval=0.1):
        self.interval = interval
        self.peak_memory = 0
        self.running = False
        self.process = psutil.Process(os.getpid())

    def _monitor(self):
        while self.running:
            try:
                # 获取当前进程及其子进程的内存
                mem = self.process.memory_info().rss
                # 尝试获取子进程内存（对于并行任务很重要）
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
        self.peak_memory = self.process.memory_info().rss
        self.thread = threading.Thread(target=self._monitor)
        self.thread.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.running = False
        self.thread.join()
        self.peak_mb = self.peak_memory / (1024 ** 2)


def SNF(W_list: list, k_neighbors: int = 15, n_iterations: int = 20, random_seed=42) -> csr_matrix:
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

def run_hisemble_workflow(embeddings):
    graphs = []
    graphs.append(build_knn_graph_chunked_cpu(embeddings[0], k_neighbors=K_NEIGHBORS, metric='gaussian', random_seed=RANDOM_SEED))
    graphs.append(build_knn_graph_chunked_cpu(embeddings[1], k_neighbors=K_NEIGHBORS, metric='gaussian', random_seed=RANDOM_SEED))
    graphs.append(build_knn_graph_chunked_cpu(embeddings[2], k_neighbors=K_NEIGHBORS, metric='cosine', random_seed=RANDOM_SEED))

    fused_network = SNF(W_list=graphs, k_neighbors=K_NEIGHBORS, n_iterations=N_ITERATIONS, random_seed=RANDOM_SEED)
    del graphs

    sources, targets = fused_network.nonzero()
    weights = fused_network[sources, targets].A1
    net = ig.Graph(list(zip(sources, targets)), directed=False)
    net.es['weight'] = weights
    clusters = find_best_matrix(matrix=net, true_cell_types=true_cell_types, seed=RANDOM_SEED)
    del net

    return clusters

def run_one_kmeans(data, k, seed):
    # n_init=1000 是为了严格复现 SC3 的高精度要求，但这很慢。
    kmeans = KMeans(n_clusters=k, n_init=1000, random_state=seed, algorithm='elkan')
    return kmeans.fit_predict(data)

def run_sc3_workflow(embeddings, n_clusters=true_cell_types, n_runs=20):
    """
    模拟 SC3 的核心瓶颈：
    1. 对每个嵌入运行多次 K-Means (并行)
    2. 构建 N x N 的密集共识矩阵 (内存杀手)
    """
    N = embeddings[0].shape[0]
    all_labels = []

    # 1. 并行运行 K-means
    # 对三个嵌入，每个跑 n_runs 次，总共 3 * n_runs 次
    tasks = []
    for emb in embeddings:
        for i in range(n_runs):
            tasks.append((emb, i))
    
    results = Parallel(n_jobs=-1)(
        delayed(run_one_kmeans)(emb, n_clusters, seed) 
        for emb, seed in tasks
    )
    label_matrix = np.array(results).T # Shape: (N, 3*n_runs)

    # 2. 构建共识矩阵 (这里是内存爆炸的地方)
    consensus_matrix = np.zeros((N, N), dtype=np.float32)
    
    # 逐列计算以节省一点内存，但最终必须持有 N x N 矩阵
    n_total_runs = label_matrix.shape[1]
    for i in range(n_total_runs):
        lbl = label_matrix[:, i]
        # 广播计算：相等为1，不等为0
        # 这一步会生成临时的 N x N bool 矩阵
        mat = (lbl[:, None] == lbl[None, :])
        consensus_matrix += mat
    
    consensus_matrix /= n_total_runs
    
    # 3. 层次聚类 (需要距离矩阵)
    
    # SC3 使用 1-consensus 作为距离
    
    Z = linkage(squareform(1 - consensus_matrix, checks=False), method='complete')
    labels = fcluster(Z, t=n_clusters, criterion='maxclust')
    
    return labels

MAX_CELLS = 183175
sample_sizes = [10000, 20000, 50000, 80000, 100000, 150000, 180000]
results = []
for n in sample_sizes:
    # 下采样
    indices = np.random.choice(MAX_CELLS, n, replace=False)
    embs_subset = [FULL_EMB_1[indices], FULL_EMB_2[indices], FULL_EMB_3[indices]]
    try:
        print("  Running SC3 (Consensus)...")
        with MemoryMonitor() as monitor:
            start = time.time()
            run_sc3_workflow(embs_subset) 
            end = time.time()
        res = {
            'Method': 'Consensus(SC3)', 'Cells': n, 
            'Time(s)': end - start, 'Memory(MB)': monitor.peak_mb
        }
    except MemoryError:
        print("    SC3: Memory Error (OOM)!")
        # 记录一个特殊值用于画图，表示失败
        results.append({'Method': 'Consensus(SC3)', 'Cells': n, 'Time(s)': None, 'Memory(MB)': None})
    except Exception as e:
        print(f"    SC3 Error: {e}")
        if "Array is too big" in str(e): # numpy 的报错
             results.append({'Method': 'Consensus(SC3)', 'Cells': n, 'Time(s)': None, 'Memory(MB)': None})


    pd.DataFrame([res]).to_csv(save_path, mode='a', header=not os.path.exists(save_path), index=False)
print("\nAll done.")
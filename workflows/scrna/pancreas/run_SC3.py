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
import matplotlib as mpl
from matplotlib import pyplot as plt
import seaborn as sns
import pickle
import anndata as ad
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings("ignore")
from utils import cluster_evaluation, overcorrection_score #从utils里面导入
from sklearn.metrics.pairwise import rbf_kernel, euclidean_distances
from sklearn.metrics import adjusted_rand_score,adjusted_mutual_info_score,normalized_mutual_info_score, homogeneity_score
from scipy.spatial.distance import pdist, squareform
import igraph as ig
import leidenalg
from sklearn.metrics import pairwise_distances
from scipy.sparse import csr_matrix, identity as sparse_identity
from sklearn.neighbors import NearestNeighbors
import gc

FIG_SIZE = (6, 6)   
DOT_SIZE = 12       
TITLE_SIZE = 18     

plt.rcParams['font.family'] = 'Arial'
mpl.rcParams['pdf.fonttype'] = 42 
mpl.rcParams['ps.fonttype'] = 42
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['axes.linewidth'] = 1.2

def plot_pixel_perfect_umap(adata, color_key, title, filename, ari_score=None):
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    sc.pl.umap(
        adata,
        color=color_key,
        show=False,
        ax=ax,
        size=DOT_SIZE,
        legend_loc='none', 
        title=title,
        frameon=False,
    )
    
    if ari_score is not None:
        ax.text(
            0.95, 0.95, 
            f'ARI: {ari_score:.2f}', 
            transform=ax.transAxes, 
            ha='right', 
            va='top', 
            fontsize=TITLE_SIZE,
            color='black',
            fontweight='bold'
        )
    
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_title(title, fontsize=TITLE_SIZE, pad=10, fontweight='bold')
    
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)

    plt.savefig(filename, dpi=300, format="pdf", bbox_inches='tight', transparent=True)
    plt.close()
    print(f"Saved UMAP: {filename}")
# =================================================================

adata = sc.read_h5ad("../data/bestpancreas_hisemble_results.h5ad")
true_labels = adata.obs.cell_type.to_numpy()
true_cell_types = adata.obs['cell_type'].nunique()
embedding = adata.obsm['X_pca']

def run_sc3_proxy(embedding, n_clusters, n_runs=20):
    """
    Python 实现的 SC3 核心逻辑代理 (Consensus Clustering)
    
    Args:
        embedding: (n_cells, n_features) 的低维矩阵 (如 PCA)
        n_clusters: 目标聚类数 (k)
        n_runs: 模拟 SC3 的多次运行次数 (默认20次足够模拟共识效果)
    
    Returns:
        labels: 最终的聚类标签
    """
    n_cells = embedding.shape[0]
    print(f"--- 开始运行 SC3 Proxy (Cells={n_cells}, K={n_clusters}) ---")

    # 1. 多次运行 K-Means (模拟 SC3 的集成步骤)
    print(f"Step 1: Running K-Means {n_runs} times...")
    labels_matrix = np.zeros((n_cells, n_runs), dtype=int)
    
    for i in range(n_runs):
        # 改变 random_state 以获得不同的初始化结果
        kmeans = KMeans(n_clusters=n_clusters, n_init=1, random_state=i, max_iter=300)
        labels_matrix[:, i] = kmeans.fit_predict(embedding)
    
    # 2. 构建共识矩阵 (Consensus Matrix) - 内存瓶颈所在！
    print("Step 2: Building Consensus Matrix (Dense)...")
    # Consensus_ij = (i 和 j 在多少次运行中被分到了同一类)
    # 这是一个 N x N 的密集矩阵
    consensus_matrix = np.zeros((n_cells, n_cells), dtype=np.float32)
    
    for i in range(n_runs):
        lbl = labels_matrix[:, i]
        # 广播计算：如果 lbl[u] == lbl[v] 则为 1
        # 这一步内存消耗巨大，如果 n_cells > 3万，Mac可能会卡
        match_matrix = (lbl[:, None] == lbl[None, :]).astype(np.float32)
        consensus_matrix += match_matrix
        
        # 手动回收垃圾，防止内存溢出
        del match_matrix
        gc.collect()
        
    consensus_matrix /= n_runs
    
    # 3. 层次聚类 (Hierarchical Clustering)
    print("Step 3: Hierarchical Clustering...")
    # SC3 使用 Complete Linkage
    # linkage 需要距离矩阵，我们用 1 - consensus 作为距离
    # squareform 将 N x N 矩阵压缩为距离向量 (这是 scipy 的要求)
    try:
        # 注意：这里也可能爆内存，因为 squareform 需要拷贝数据
        dist_vec = squareform(1.0 - consensus_matrix, checks=False)
        Z = linkage(dist_vec, method='complete')
        
        # 获取最终标签
        final_labels = fcluster(Z, t=n_clusters, criterion='maxclust')
        
        # 标签标准化 (从0开始)
        return (final_labels - 1).astype(str)
        
    except MemoryError:
        print("!!! Memory Error in Hierarchical Clustering !!!")
        return None
    
labels = run_sc3_proxy(embedding, n_clusters=true_cell_types)

adata.obs['SC3'] = labels
adata.obs['SC3'] = adata.obs['SC3'].astype('category')
unique_types = np.unique(true_cell_types)

adata.obs['cell_type'] = adata.obs['cell_type'].astype('category')
plot_pixel_perfect_umap(
    adata, 
    color_key='cell_type', 
    title='SC3 embedding (ground truth)', 
    filename='../UMAP/bestumap_cell_type(pancreas_SC3).pdf',
    ari_score=None
)
plot_pixel_perfect_umap(
    adata, 
    color_key='SC3', 
    title='SC3 clusters', 
    filename='../UMAP/bestumap_SC3(pancreas_SC3).pdf'
)
ARI = adjusted_rand_score(true_labels, labels)
AMI = adjusted_mutual_info_score(true_labels, labels)
NMI = normalized_mutual_info_score(true_labels, labels)
HOM = homogeneity_score(true_labels, labels)
result_data = {
    'ARI': [ARI],
    'AMI': [AMI],
    'NMI': [NMI],
    'HOM': [HOM]
}
result_df = pd.DataFrame(result_data)

result_df.to_csv('../metrics/bestmetrics_sc3_pancreas.csv', index=False)
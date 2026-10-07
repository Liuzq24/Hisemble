import scanpy as sc
import pandas as pd
import scipy.io
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import font_manager
import igraph as ig
import leidenalg
import numpy as np
from utils import find_best_matrix, perform_leiden_clustering, remove_duplicates, build_knn_graph_chunked_cpu, normalize_sparse, sparsify_matrix
from sklearn.metrics.pairwise import rbf_kernel, euclidean_distances
from sklearn.metrics import adjusted_rand_score,adjusted_mutual_info_score,normalized_mutual_info_score, homogeneity_score
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import pairwise_distances
from scipy.sparse import csr_matrix, identity as sparse_identity
from sklearn.neighbors import NearestNeighbors
import math
# UMAP figure settings
FIG_SIZE = (6, 6)   # 单个 UMAP 画布大小
DOT_SIZE = 12       # 散点大小
TITLE_SIZE = 18     # 标题字体大小

plt.rcParams['font.family'] = 'Arial'
mpl.rcParams['pdf.fonttype'] = 42 
mpl.rcParams['ps.fonttype'] = 42
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['axes.linewidth'] = 1.2

def plot_pixel_perfect_umap(adata, color_key, title, filename, ari_score=None):
    """标准化的无边框 UMAP 绘图函数"""
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    sc.pl.umap(
        adata,
        color=color_key,
        show=False,
        ax=ax,
        size=DOT_SIZE,
        legend_loc='none', # 隐藏自带图例
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

def save_horizontal_legend(adata, color_key, filename):
    """生成横向排布的独立图例，强制排为 4 行"""
    cats = adata.obs[color_key].cat.categories
    colors = adata.uns[f'{color_key}_colors']
    global_ctype_palette = dict(zip(cats, colors))

    # 宽度依然是 18 (3倍 UMAP 宽)，高度从 2 增加到 4，给 4 行图例留出充足的呼吸空间
    fig_leg, ax_leg = plt.subplots(figsize=(18, 4)) 
    
    legend_patches = []
    for ctype, color in global_ctype_palette.items():
        patch = mpatches.Patch(color=color, label=ctype)
        legend_patches.append(patch)

    # Arrange legend entries in four rows.
    n_cols = math.ceil(len(cats) / 4.0)

    leg = ax_leg.legend(
        handles=legend_patches, 
        loc='center', 
        ncol=n_cols,          
        frameon=False, 
        fontsize=14,
        markerscale=2,
        prop={'weight': 'bold', 'size': 14, 'family': 'Arial'} 
    )
    
    ax_leg.axis('off')
    plt.savefig(filename, dpi=300, format="pdf", bbox_inches='tight', transparent=True)
    plt.close()
    print(f"Saved Legend (4 rows): {filename}")
# =================================================================

K_NEIGHBORS = 5
N_ITERATIONS = 20
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
os.environ['PYTHONHASHSEED'] = str(RANDOM_SEED)

adata = sc.read_h5ad("../data/scRNA_pancreas.h5ad")
embedding_pca = adata.obsm['X_pca']
print(f"视图 1 (PCA) 准备就绪: {embedding_pca.shape}")
embedding_scvi = adata.obsm['X_scvi']
print(f"视图 2 (scVI) 准备就绪: {embedding_scvi.shape}")
true_labels = adata.obs.cell_type.to_numpy()
true_cell_types = adata.obs['cell_type'].nunique()

W_pca = build_knn_graph_chunked_cpu(embedding_pca, k_neighbors=K_NEIGHBORS, metric='gaussian') 
W_scvi = build_knn_graph_chunked_cpu(embedding_scvi, k_neighbors=K_NEIGHBORS, metric='gaussian') 
W_list = [W_pca, W_scvi]

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

fused_network = SNF(W_list, k_neighbors=K_NEIGHBORS, n_iterations=N_ITERATIONS, random_seed=RANDOM_SEED)
sources, targets = fused_network.nonzero()
weights = fused_network[sources, targets].A1
net = ig.Graph(list(zip(sources, targets)), directed=False)
net.es['weight'] = weights
clusters = find_best_matrix(matrix=net, true_cell_types=true_cell_types, seed=RANDOM_SEED)
fused_network = fused_network.astype(np.float32)
adata.obsp['connectivities'] = fused_network
distances = adata.obsp['connectivities'].copy()
distances.data = 1.0 - distances.data
distances.eliminate_zeros() # 移除那些 similarity=1.0 导致的 0
adata.obsp['distances'] = distances

adata.uns['neighbors'] = {}
adata.uns['neighbors']['params'] = {'method': 'custom', 'custom_graph': True}
adata.uns['neighbors']['connectivities_key'] = 'connectivities' 
adata.uns['neighbors']['distances_key'] = 'distances'  

sc.tl.umap(adata, random_state=RANDOM_SEED)
if len(clusters) != adata.n_obs:
    raise ValueError(f"聚类列表的长度 ({len(clusters)}) "
                        f"与 adata.n_obs ({adata.n_obs}) 不匹配")
adata.obs['Hisemble'] = clusters
adata.obs['Hisemble'] = adata.obs['Hisemble'].astype('category')
adata.obs['cell_type'] = adata.obs['cell_type'].astype('category')
plot_pixel_perfect_umap(
    adata, 
    color_key='cell_type', 
    title='Hisemble embedding (ground truth)', 
    filename='../UMAP/bestumap_cell_type(Pancreas_Hisemble).pdf',
    ari_score=None
)
save_horizontal_legend(
    adata, 
    color_key='cell_type', 
    filename='../UMAP/Pancreas_Unified_Legend.pdf'
)
plot_pixel_perfect_umap(
    adata, 
    color_key='Hisemble', 
    title='Hisemble clusters', 
    filename='../UMAP/bestumap_Hisemble(Pancreas_Hisemble).pdf'
)

ARI = adjusted_rand_score(true_labels, clusters)
AMI = adjusted_mutual_info_score(true_labels, clusters)
NMI = normalized_mutual_info_score(true_labels, clusters)
HOM = homogeneity_score(true_labels, clusters)

result_data = {
    'k_neighbors': [K_NEIGHBORS],
    'n_iterations': [N_ITERATIONS],
    'ARI': [ARI],
    'AMI': [AMI],
    'NMI': [NMI],
    'HOM': [HOM]
}
result_df = pd.DataFrame(result_data)

result_df.to_csv('../metrics/bestmetrics_pancreas.csv', index=False)

adata.write_h5ad("../data/bestpancreas_hisemble_results.h5ad")
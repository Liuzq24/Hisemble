import os
import warnings
import pandas as pd
import numpy as np
import scipy.sparse as sp
from scipy.sparse import issparse, csr_matrix
import scanpy as sc
import anndata
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib import font_manager
import igraph as ig
import leidenalg
from sklearn.metrics.pairwise import rbf_kernel, cosine_similarity
from sklearn.metrics import adjusted_rand_score
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.decomposition import TruncatedSVD

# ==============================================================================
# 0. 导入 GraphST 与 PyTorch
# 安装要求: pip install GraphST torch torch_geometric
# ==============================================================================
import torch
from GraphST.GraphST import GraphST
warnings.filterwarnings("ignore")

# ==============================================================================
# 1. 全局排版风格配置
# ==============================================================================
font_dir = "../"
arial_regular_path = os.path.join(font_dir, "Arial.ttf")
arial_bold_path = os.path.join(font_dir, "Arial Bold.ttf")

if os.path.exists(arial_regular_path):
    font_manager.fontManager.addfont(arial_regular_path)
if os.path.exists(arial_bold_path):
    font_manager.fontManager.addfont(arial_bold_path)

plt.rcParams['font.family'] = 'Arial'
mpl.rcParams['pdf.fonttype'] = 42 
mpl.rcParams['ps.fonttype'] = 42
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['axes.labelweight'] = 'bold'

# ==============================================================================
# 2. 配置路径与四阶段切片任务 (请根据你的实际文件名调整)
# ==============================================================================
DATA_DIR = "__HISEMBLE_SPATIAL_ROOT__"

# 整理你需要跑的四个切片阶段
TASKS = [
    {"file": "E11_0-S1_expr_preprocessed.h5ad", "mode": "RNA"},
    {"file": "E13_5-S1_expr_preprocessed.h5ad", "mode": "RNA"}, 
    {"file": "E15_5-S1_expr_preprocessed.h5ad", "mode": "RNA"},
    {"file": "E18_5-S1_expr_preprocessed.h5ad", "mode": "RNA"}
]

K_NEIGHBORS = 20    
SEED = 42           

# 设定设备 (自动调用 GPU)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(SEED)
np.random.seed(SEED)

# ==============================================================================
# 3. Clustering and graph operations
# ==============================================================================
def find_best_matrix(matrix, true_cell_types, low=0.1, high=5.0, tolerance=0.01, seed=42):
    sources, targets = matrix.nonzero()
    weights = matrix[sources, targets].A1
    g = ig.Graph(list(zip(sources, targets)), directed=False)
    g.es['weight'] = weights

    best_resolution = None
    def remove_duplicates(lst):  
        return list(np.unique(lst))
        
    low_n_clusters = len(remove_duplicates(perform_leiden_clustering(g, resolution=low, seed=seed)))
    high_n_clusters = len(remove_duplicates(perform_leiden_clustering(g, resolution=high, seed=seed)))

    if high_n_clusters <= true_cell_types:
        best_resolution = high
    elif low_n_clusters >= true_cell_types:
        best_resolution = low
    else:
        while high - low > tolerance:
            mid = (low + high) / 2
            current_clusters = perform_leiden_clustering(g, resolution=mid, seed=seed)
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
    final_clusters = perform_leiden_clustering(g, resolution=best_resolution, seed=seed)
    return np.array(final_clusters).astype(str)

def perform_leiden_clustering(net, resolution, seed=42):
    partition = leidenalg.find_partition(net, leidenalg.RBConfigurationVertexPartition, 
                                         weights=net.es['weight'], 
                                         resolution_parameter=resolution, seed=seed)
    return partition.membership

def build_knn_graph_chunked_cpu(embedding: np.ndarray, k_neighbors: int, metric: str, chunk_size: int = 2048, random_seed=42) -> csr_matrix:
    n_samples = embedding.shape[0]
    all_rows, all_cols, all_data = [], [], []
    gamma = 1.0 / (embedding.shape[1] * embedding.var() + 1e-12) if metric == 'gaussian' else None

    for i in range(0, n_samples, chunk_size):
        start_idx = i
        end_idx = min(i + chunk_size, n_samples)
        chunk_embedding = embedding[start_idx:end_idx]
        if metric == 'gaussian':
            sim_chunk = rbf_kernel(chunk_embedding, embedding, gamma=gamma)
        elif metric == 'cosine':
            sim_chunk = np.maximum(cosine_similarity(chunk_embedding, embedding), 0)

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
    return (W_sparse + W_sparse.T) / 2

# ==============================================================================
# 4. 核心绘图模块 (底置图例)
# ==============================================================================
def draw_combined_plots(adata, ari_base, ari_graphst, mode, prefix):
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(21, 8), gridspec_kw={'wspace': 0.05})
    
    base_title = f'PCA (RNA) (ARI={ari_base:.2f})' if mode == 'RNA' else f'ATAC (LSI) (ARI={ari_base:.2f})'
    titles = ['Ground Truth', base_title, f'GraphST (ARI={ari_graphst:.2f})']
    columns = ['Ground_Truth', 'Baseline', 'GraphST_Cluster']
    
    for i, ax in enumerate(axes):
        sc.pl.spatial(adata, color=columns[i], title=titles[i], spot_size=1, show=False, ax=ax, frameon=False)
        
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.title.set_fontsize(20) 
        ax.title.set_fontweight('bold')
        
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color('black')
            spine.set_linewidth(1.2)
            
        legend = ax.get_legend()
        if legend:
            try:
                handles = legend.legend_handles
            except AttributeError:
                handles = legend.legendHandles
            labels = [t.get_text() for t in legend.get_texts()]
            
            legend.remove()
            ncols = 2 if i == 0 else 4
            
            new_legend = ax.legend(handles, labels, 
                                   loc='upper center', 
                                   bbox_to_anchor=(0.5, -0.05), 
                                   ncol=ncols, 
                                   frameon=False, 
                                   prop={'size': 14, 'weight': 'bold'}) 
            try:
                for handle in handles:
                    handle.set_sizes([100])
            except:
                pass

    out_file = f"Combined_{mode}_{prefix}_GraphST.pdf"
    plt.savefig(out_file, dpi=300, format="pdf", bbox_inches='tight')
    plt.close()
    print(f"✅ 成功生成 21:8 合并大图：{out_file}\n")

# ==============================================================================
# 5. GraphST 数据处理流水线
# ==============================================================================
def process_pipeline(filename, mode):
    print(f"\n{'='*60}")
    print(f"🚀 开始处理 {mode} 模态数据: {filename}")
    print(f"{'='*60}")
    
    file_path = os.path.join(DATA_DIR, filename)
    if not os.path.exists(file_path):
        print(f"❌ 错误：文件不存在 {file_path}")
        return

    prefix = filename.replace(f'_{mode.lower()}_preprocessed.h5ad', '').replace('_expr_preprocessed.h5ad', '')
    
    print(">>> 1. Loading Spatial Data...")
    adata = sc.read_h5ad(file_path)
    
    # 获取空间坐标
    if 'array_col' in adata.obs.columns and 'array_row' in adata.obs.columns:
        X_spatial = adata.obs[['array_col', 'array_row']].values
    elif 'x' in adata.obs.columns and 'y' in adata.obs.columns:
        X_spatial = adata.obs[['x', 'y']].values
    
    adata.obsm['spatial'] = X_spatial
    true_labels = adata.obs['Annotation_for_Combined'].values
    adata.obs['Ground_Truth'] = pd.Categorical(true_labels)
    n_true_clusters = len(np.unique(true_labels))
    
    print(f">>> 2. Baseline Embedding ({'PCA' if mode == 'RNA' else 'LSI'})...")
    # 因为 GraphST 对特征输入有要求，这里我们为 GraphST 的模型输入创造一个适配版
    if mode == 'RNA':
        # Baseline PCA
        sc.tl.pca(adata, n_comps=30, svd_solver='arpack', random_state=SEED)
        X_base = adata.obsm['X_pca']
        adata_graphst = adata.copy() 
    else: 
        # Baseline ATAC (LSI)
        tfidf = TfidfTransformer(norm='l2', use_idf=True)
        X_tfidf = tfidf.fit_transform(adata.X)
        svd = TruncatedSVD(n_components=30, random_state=SEED)
        X_base = svd.fit_transform(X_tfidf)[:, 1:] 
        
        # ⚠️ ATAC 适配：由于 GraphST 是 GNN 且专为 RNA 设计，稀疏的 peak 矩阵可能导致不收敛
        # 最稳妥的方法是：将 ATAC 降维后的 LSI 稠密特征赋值给 X 给 GraphST 训练
        adata_graphst = anndata.AnnData(X=X_base)
        adata_graphst.obsm['spatial'] = X_spatial
        adata_graphst.obs = adata.obs
        
    print(">>> 3. Running GraphST Model (Self-supervised Contrastive Learning)...")
    # 实例化模型 (参数遵循默认教程)
    model = GraphST(adata_graphst, device=device)    
    # 训练模型 (GraphST 会自动建图并进行对比学习)
    adata_graphst = model.train()
    
    # 提取学习到的隐层表征 (GraphST 默认将特征存在 adata.obsm['emb'])
    X_graphst = adata_graphst.obsm['emb']

    print(">>> 4. Clustering (Auto-resolving Leiden)...")
    # 4.1 Baseline 聚类
    P_base = build_knn_graph_chunked_cpu(X_base, k_neighbors=K_NEIGHBORS, metric='cosine', random_seed=SEED)
    labels_base = find_best_matrix(P_base, n_true_clusters, seed=SEED)
    
    # 4.2 GraphST 聚类
    # 构建 KNN 图以兼容你写的 Leiden 二分查找法
    P_graphst = build_knn_graph_chunked_cpu(X_graphst, k_neighbors=K_NEIGHBORS, metric='cosine', random_seed=SEED)
    labels_graphst = find_best_matrix(P_graphst, n_true_clusters, seed=SEED)
    
    # 评估
    ari_base = adjusted_rand_score(true_labels, labels_base)
    ari_graphst = adjusted_rand_score(true_labels, labels_graphst)
    print(f"\n======== RESULTS for {prefix} ========")
    print(f"Baseline ARI = {ari_base:.4f} | GraphST ARI = {ari_graphst:.4f}\n")
    
    # 格式化
    base_int = labels_base.astype(int) + 1
    graphst_int = labels_graphst.astype(int) + 1
    
    cats_base = [str(x) for x in sorted(np.unique(base_int))]
    cats_graphst = [str(x) for x in sorted(np.unique(graphst_int))]
    
    adata.obs['Baseline'] = pd.Categorical(base_int.astype(str), categories=cats_base, ordered=True)
    adata.obs['GraphST_Cluster'] = pd.Categorical(graphst_int.astype(str), categories=cats_graphst, ordered=True)
    
    print(">>> 5. 正在生成排版大图...")
    draw_combined_plots(adata, ari_base, ari_graphst, mode, prefix)

# ==============================================================================
# 6. 批量启动器 (依次跑完 4 个切片)
# ==============================================================================
if __name__ == "__main__":
    for task in TASKS:
        try:
            process_pipeline(task['file'], mode=task['mode'])
        except Exception as e:
            print(f"❌ 运行 {task['file']} 时发生错误: {e}")
            
    print("\n🎉 所有四阶段切片任务处理完毕！")
import warnings
warnings.filterwarnings("ignore")
import tensorflow as tf
import os
import pandas as pd
import numpy as np
import sklearn
from sklearn import metrics
import scipy
from scipy.sparse import issparse
import scanpy as sc
import anndata
import episcanpy.api as epi
import matplotlib as mpl
from matplotlib import pyplot as plt
import seaborn as sns
import pickle
import scib
import random

from utils import cluster_evaluation, overcorrection_score
from scbasset.utils import make_model, get_cell_embedding 

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    print(f"全局随机种子已设置为: {seed}")

GLOBAL_SEED = 42
set_seed(GLOBAL_SEED)

def find_best_leiden_resolution(adata, true_cell_types, low=0.1, high=5.0, tolerance=0.01, key_prefix='leiden_res', random_seed=42):

    low_n_clusters = sc.tl.leiden(adata, resolution=low, copy=True, random_state=random_seed).obs['leiden'].nunique()
    high_n_clusters = sc.tl.leiden(adata, resolution=high, copy=True, random_state=random_seed).obs['leiden'].nunique()

    # 检查边界情况
    if high_n_clusters <= true_cell_types:
        print(f"警告: resolution上限 {high} 产生的簇数 ({high_n_clusters}) 不大于目标簇数 ({true_cell_types})。直接使用 resolution={high}。")
        sc.tl.leiden(adata, resolution=high, key_added=f'{key_prefix}_best', random_state=random_seed)
        return high
    if low_n_clusters >= true_cell_types:
        print(f"警告: resolution下限 {low} 产生的簇数 ({low_n_clusters}) 不小于目标簇数 ({true_cell_types})。直接使用 resolution={low}。")
        sc.tl.leiden(adata, resolution=low, key_added=f'{key_prefix}_best', random_state=random_seed)
        return low

    best_res = low
    best_n_clust = low_n_clusters

    while high - low > tolerance:
        mid = (low + high) / 2
        sc.tl.leiden(adata, resolution=mid, key_added=f'leiden_{mid:.4f}', random_state=random_seed)
        num_clusters = adata.obs[f'leiden_{mid:.4f}'].nunique()

        # 记录与目标簇数最接近的分辨率
        if abs(num_clusters - true_cell_types) < abs(best_n_clust - true_cell_types):
            best_res = mid
            best_n_clust = num_clusters

        if num_clusters < true_cell_types:
            low = mid
        elif num_clusters > true_cell_types:
            high = mid
        else:
            print(f"找到精确匹配！最佳分辨率: {mid:.4f}，簇数: {num_clusters}")
            sc.tl.leiden(adata, resolution=mid, key_added=f'{key_prefix}_best', random_state=random_seed)
            return mid

    print(f"搜索结束。最佳分辨率为 {best_res:.4f}，产生 {best_n_clust} 个簇 (目标为 {true_cell_types})。")
    sc.tl.leiden(adata, resolution=best_res, key_added=f'{key_prefix}_best', random_state=random_seed)
    return best_res


def process_single_dataset(dataname, base_path='__HISEMBLE_DATA_ROOT__/'):
    """
    处理单个数据集的完整分析流程。
    """
    print(f"\n{'='*20} 开始处理数据集: {dataname} {'='*20}")

    # --- 1. 定义路径并创建目录 ---
    save_folder_umap = os.path.join(base_path, 'UMAP/')
    savedata_folder = os.path.join(base_path, 'ADATA/')
    savemetric_folder = os.path.join(base_path, 'metrics/scBasset/')
    scbasset_results_folder = os.path.join(base_path, f'scBasset_results/{dataname}/')

    for folder in [save_folder_umap, savedata_folder, savemetric_folder, scbasset_results_folder]:
        os.makedirs(folder, exist_ok=True)

    ad_file = os.path.join(base_path, f'0.01_processed/{dataname}_processed.h5ad')
    trained_model_file = os.path.join(base_path, f'scBasset_output/{dataname}/best_model.h5')

    # --- 2. 加载数据和模型，生成嵌入 ---
    print("加载数据和预训练模型...")
    if not os.path.exists(ad_file) or not os.path.exists(trained_model_file):
        print(f"错误: 数据文件或模型文件不存在，跳过 {dataname}。")
        return None

    ad = anndata.read_h5ad(ad_file)
    model = make_model(32, ad.shape[0], show_summary=False)
    model.load_weights(trained_model_file)
    proj = get_cell_embedding(model)

    proj_csv_path = os.path.join(scbasset_results_folder, 'projection_atac.csv')
    pd.DataFrame(proj).to_csv(proj_csv_path)
    ad.obsm['projection'] = proj

    # --- 3. UMAP 可视化 ---
    print("计算和绘制 UMAP 图...")
    sc.pp.neighbors(ad, use_rep='projection', n_neighbors=15, random_state=GLOBAL_SEED)
    sc.tl.umap(ad, random_state=GLOBAL_SEED)

    sc.pl.umap(ad, color=['cell_type'], show=False, color_map='viridis')
    plt.savefig(os.path.join(save_folder_umap, f'umap_cell_type({dataname}_scBasset).pdf'), dpi=300, format="pdf", bbox_inches='tight', transparent=True)
    plt.close() # 关闭图像，防止在循环中占用过多内存

    if 'batch' in ad.obs.columns:
        sc.pl.umap(ad, color=['batch'], show=False, color_map='viridis')
        plt.savefig(os.path.join(save_folder_umap, f'umap_batch({dataname}_scBasset).pdf'), dpi=300, format='pdf', bbox_inches='tight', transparent=True)
        plt.close()

    # --- 4. 聚类与评估 ---
    print("进行Leiden聚类...")
    true_cell_types = ad.obs['cell_type'].nunique()
    best_res = find_best_leiden_resolution(ad, true_cell_types, random_seed=GLOBAL_SEED) 
    leiden_key = f'leiden_res_best' # find_best_leiden_resolution 会创建这个key

    print("计算评估指标...")
    # 注意：cluster_evaluation被调用了多次，效率较低。优化一下。
    ami, ari, nmi, hom = cluster_evaluation(ad.obs, 'cell_type', leiden_key)

    # 计算批次校正和整合指标
    if 'batch' in ad.obs.columns:
        graph_clisi = scib.me.clisi_graph(ad, label_key="batch", type_="embed", use_rep="projection")
        batch_asw = scib.me.silhouette_batch(ad, batch_key="batch", label_key="cell_type", embed="projection", verbose=False)
        graph_connectivity = scib.me.graph_connectivity(ad, label_key="cell_type")
        ilisi_graph = scib.me.ilisi_graph(ad, batch_key="batch", type_="embed", use_rep="projection", scale=False)
    else:
        graph_clisi, batch_asw, graph_connectivity, ilisi_graph = 0, 0, 0, 0

    overcorrection = overcorrection_score(ad.obsm["projection"], ad.obs['cell_type'], n_neighbors=100, n_pools=100, n_samples_per_pool=100)

    # --- 5. 保存结果 ---
    print("保存结果...")
    # 保存处理后的 anndata 对象
    pkl_file_name = os.path.join(savedata_folder, f"{dataname}_scBasset.pkl")
    with open(pkl_file_name, "wb") as f:
        pickle.dump(ad, f)

    # 汇总指标
    metrics_dict = {
        f"{dataname}_scBasset": [ami, ari, nmi, hom, graph_clisi, batch_asw, graph_connectivity, ilisi_graph, overcorrection]
    }
    metrics_df = pd.DataFrame(metrics_dict, index=["AMI", "ARI", "NMI", "HOM", 'GraphcLISI', 'batchASW', 'Graph_connectivity', 'ilisi_graph', 'overcorrection'])

    # 保存指标到CSV文件
    # 采用更健壮的保存方式：每次都覆盖或先读后写，而不是追加模式'a'
    final_metrics_file = os.path.join(savemetric_folder, 'scBasset_metrics_summary.csv')
    if os.path.exists(final_metrics_file):
        all_metrics_df = pd.read_csv(final_metrics_file, index_col=0)
        # 更新或添加当前数据集的结果
        all_metrics_df[f"{dataname}_scBasset"] = metrics_df[f"{dataname}_scBasset"]
    else:
        all_metrics_df = metrics_df

    all_metrics_df.to_csv(final_metrics_file)

    print(f"--- 数据集 {dataname} 处理完成 ---")
    return metrics_df


# ================== 主程序入口 ==================
if __name__ == '__main__':
    # 1. 定义你的数据集列表
    dataname_list = [
        'mouse_brain'

    ]

    # 2. 循环处理每个数据集
    all_results = {}
    for dataname in dataname_list:
        result_df = process_single_dataset(dataname)
        if result_df is not None:
            all_results.update(result_df.to_dict())

    # 3. 汇总并打印最终结果
    if all_results:
        final_summary_df = pd.DataFrame(all_results)
        print("\n\n{'='*25} 所有任务完成，最终结果汇总 {'='*25}")
        print(final_summary_df)
    else:
        print("\n所有任务处理完毕，没有成功的结果。")
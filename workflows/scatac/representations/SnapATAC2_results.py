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
import time  # 引入时间库
import psutil # 引入psutil库
import matplotlib as mpl
from matplotlib import pyplot as plt
import seaborn as sns

from utils import cluster_evaluation
from utils import overcorrection_score,overcorrection_score_ot
from utils import batch_entropy_mixing_score
from utils import EarlyStopping
from utils import EarlyStopping_simple
from utils import TFIDF
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import snapatac2 as snap
from pathlib import Path

# --- 辅助函数 ---
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


def Snapatac(inputdata, dataname, umapsave, savedata, savemetric):
    """
    inputdata: 预处理后数据
    dataname: 对数据的称呼，用于保存数据时记录数据名称
    saveumap: 用于保存UMAP
    savedata: 保存处理后数据的文件夹
    savemetric: 保存指标的路径
    """ 
    ad = inputdata
    from scipy.sparse import csc_matrix
    ad.X = ad.X.tocsr()
    process = psutil.Process(os.getpid())
    mem_before_spectral_mb = process.memory_info().rss / (1024 * 1024)
    start_time_spectral = time.perf_counter()
    snap.tl.spectral(ad, features = None)
    end_time_spectral = time.perf_counter()
    # 记录 spectral 执行后的内存
    mem_after_spectral_mb = process.memory_info().rss / (1024 * 1024)

    # 计算该步骤的耗时和内存增量
    elapsed_time_spectral_s = end_time_spectral - start_time_spectral
    peak_memory_spectral_mb = mem_after_spectral_mb - mem_before_spectral_mb

    print(f"--- 性能监控 [snap.tl.spectral on {dataname}] ---")
    print(f"耗时: {elapsed_time_spectral_s:.2f} 秒")
    print(f"内存峰值增量: {peak_memory_spectral_mb:.2f} MB")
    print(f"-------------------------------------------")
    sc.pp.neighbors(ad, n_neighbors=15, use_rep="X_spectral")
    snap.tl.umap(ad, random_state=None)
    save_folder = umapsave
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    sc.pl.umap(ad, color=['cell_type'], show=False, color_map='viridis') 
    plt.savefig(os.path.join(save_folder, f'umap_cell_type({dataname}_snap).pdf'), dpi=300, format="pdf", bbox_inches='tight', transparent=True)
    if 'batch' in ad.obs.columns:
        sc.pl.umap(ad, color=['batch'], show=False, color_map='viridis')
        plt.savefig(os.path.join(save_folder, f'umap_batch({dataname}_snap).pdf'), dpi=300, format='pdf', bbox_inches='tight', transparent=True)
    #snap.pp.knn(ad)
    true_cell_types = ad.obs['cell_type'].nunique()
    best_res = find_best_leiden_resolution(ad, true_cell_types)
    print(f"最佳的Leiden聚类分辨率是：{best_res}")

    import pickle
    pkl_file_name = savedata+dataname+"_snap.pkl" 
    with open(pkl_file_name, "wb") as f:
        pickle.dump(ad, f)
    
    if not os.path.exists(savemetric):
        os.makedirs(savemetric)
    savemetric = os.path.join(savemetric, 'final.pkl')
    target = {}
    
    final_cluster_key = 'leiden_res_best' 
    metrics_tuple = cluster_evaluation(ad.obs, 'cell_type', final_cluster_key)
    AMI, ARI, NMI, HOM = metrics_tuple[0], metrics_tuple[1], metrics_tuple[2], metrics_tuple[3]
    
    import scib
    if 'batch' in ad.obs.columns:
        # 如果存在 'batch' 标签，计算各项指标
        GraphcLISI = scib.me.clisi_graph(ad, label_key="batch", type_="embed", use_rep="X_spectral")
        batchASW = scib.me.silhouette_batch(ad, batch_key="batch", label_key="cell_type", embed="X_spectral", verbose=False)
        Graph_connectivity = scib.me.graph_connectivity(ad, label_key="cell_type")
        ilisi_graph = scib.me.ilisi_graph(ad, batch_key="batch", type_="embed", use_rep="X_spectral", scale=False)
    else:
        # 如果不存在 'batch' 标签，将所有指标设置为 0
        GraphcLISI = 0
        batchASW = 0
        Graph_connectivity = 0
        ilisi_graph = 0

    overcorrection = overcorrection_score(ad.obsm["X_spectral"], ad.obs['cell_type'], n_neighbors=100, n_pools=100, n_samples_per_pool=100)
    target[f"{dataname}_snap"] = [AMI, ARI, NMI, HOM, GraphcLISI, batchASW, Graph_connectivity, ilisi_graph, overcorrection, elapsed_time_spectral_s, peak_memory_spectral_mb]
    
    return target


RANDOM_SEED = 42
os.environ['PYTHONHASHSEED'] = str(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

BASE_PATH = '__HISEMBLE_DATA_ROOT__/'
datanames = [
'Droplet',
'FCA_fetal_33184180_adrenal',
'FCA_fetal_33184180_cerebellum',
'FCA_fetal_33184180_cerebrum',
'FCA_fetal_33184180_eye',
'FCA_fetal_33184180_heart',
'FCA_fetal_33184180_intestine',
'FCA_fetal_33184180_kidney',
'FCA_fetal_33184180_liver',
'FCA_fetal_33184180_lung',
'FCA_fetal_33184180_muscle',
'FCA_fetal_33184180_pancreas',
'FCA_fetal_33184180_placenta',
'FCA_fetal_33184180_spleen',
'FCA_fetal_33184180_stomach',
'FCA_fetal_33184180_thymus',
'mca_Cerebellum_62216',
'mca_HeartA_62816',
'mca_Kidney_62016',
'mca_LargeIntestineA_62816',
'mca_LargeIntestineB_62816',
'mca_Liver_62016',
'mca_Lung1_62216',
'mca_Lung2_62216',
'mca_PreFrontalCortex_62216',
'mca_SmallIntestine_62816',
'mca_Spleen_62016',
'mca_Testes_62016',
'mca_Thymus_62016',
'mca_WholeBrainA_62216',
'mca_WholeBrainA_62816',
'mouse_brain',
'Muto-2021-ATAC',
'TME-All'
    ]
adatas = {}
for dataname in datanames:
    adatas[dataname] = anndata.read_h5ad('__HISEMBLE_DATA_ROOT__/0.01_h5ad/%s_processed.h5ad'%dataname)

# 更新行索引名称，使其更精确
index_names = [
    "AMI", "ARI", "NMI", "HOM", "GraphcLISI", "batchASW", 
    "Graph_connectivity", "ilisi_graph", "overcorrection",
    "Spectral_Time_s", "Spectral_Memory_MB" # 新名称
]

all_results_dict = {}

for dataname in datanames:
    umapsave =  f'__HISEMBLE_DATA_ROOT__/UMAP/' 
    savedata =  f"__HISEMBLE_DATA_ROOT__/ADATA/"
    savemetric_folder = '__HISEMBLE_RESULTS_ROOT__/snapatac/'
    
    # 确保目录存在
    if not os.path.exists(savedata):
        os.makedirs(savedata)
    if not os.path.exists(umapsave):
        os.makedirs(umapsave)
    if not os.path.exists(savemetric_folder):
        os.makedirs(savemetric_folder)
    
    # 调用修改后的函数
    result_dict = Snapatac(
        inputdata=adatas[dataname],
        dataname=dataname,
        umapsave=umapsave,
        savedata=savedata,
        savemetric=savemetric_folder
    )
    
    all_results_dict.update(result_dict)


if not all_results_dict:
    print("\n所有任务处理完毕，但没有成功的结果。")
else:
    index_names = [
        "AMI", "ARI", "NMI", "HOM", "GraphcLISI", "batchASW", 
        "Graph_connectivity", "ilisi_graph", "overcorrection",
        "Spectral_Time_s", "Spectral_Memory_MB"
    ]
    final_df = pd.DataFrame(all_results_dict, index=index_names)
    
    final_metrics_file = os.path.join(savemetric_folder, 'final_metrics_summary.csv')

    # 健壮的CSV保存逻辑
    if os.path.exists(final_metrics_file):
        print(f"正在更新已有的指标文件: {final_metrics_file}")
        try:
            existing_df = pd.read_csv(final_metrics_file, index_col=0)
            # 使用 update 方法，它会用新数据覆盖旧数据中的同名列
            existing_df.update(final_df)
            # 添加 final_df 中有但 existing_df 中没有的新列
            for col in final_df.columns:
                if col not in existing_df.columns:
                    existing_df[col] = final_df[col]
            final_df = existing_df
        except Exception as e:
            print(f"读取旧指标文件失败: {e}。将覆盖文件。")

    final_df.to_csv(final_metrics_file)

    print(f"\n\n{'='*25} 所有任务完成 {'='*25}")
    print("最终结果汇总:")
    print(final_df)
    print(f"\n所有指标已保存/更新至: {final_metrics_file}")

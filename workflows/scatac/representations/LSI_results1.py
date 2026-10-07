import scanpy as sc
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import time
import tracemalloc
import pickle
from utils import cluster_evaluation, overcorrection_score
# 导入必要的评估库
import scib
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.decomposition import TruncatedSVD
from scipy.stats import entropy
import random

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
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


def run_lsi(adata_paths, output_dir, cell_type_key='cell_type', batch_key='batch'):
    """
    对一系列单细胞数据集进行LSI处理、评估并保存结果。
    """
    # --- 1. 初始化 ---
    # 统一且清晰地定义所有输出路径
    umap_save_path = os.path.join(output_dir, 'UMAP')
    adata_save_path = os.path.join(output_dir, 'ADATA')
    metrics_save_path = os.path.join(output_dir, 'metrics')
    os.makedirs(umap_save_path, exist_ok=True)
    os.makedirs(adata_save_path, exist_ok=True)
    os.makedirs(metrics_save_path, exist_ok=True)
    
    # 用于存储所有数据集最终评估指标的列表
    all_metrics_list = []

    print(f"开始处理 {len(adata_paths)} 个数据集...")

    # --- 2. 循环处理每个数据集 ---
    for name, path in adata_paths.items():
        print(f"\n{'='*20} 正在处理: {name} {'='*20}")
        
        try:
            adata = sc.read_h5ad(path)
            
            # --- 2.1. LSI 流程 ---
            print("[1/4] LSI: Binarize -> TF-IDF -> SVD")
            adata.layers['counts'] = adata.X.copy()
            
            binarized_matrix = adata.X.copy()
            binarized_matrix[binarized_matrix > 0] = 1
            
            tfidf_transformer = TfidfTransformer(norm='l2', use_idf=True, smooth_idf=True)
            tfidf_matrix = tfidf_transformer.fit_transform(binarized_matrix)
            adata.X = tfidf_matrix
            
            # --- 性能监控开始 ---
            print("  [LSI] 正在执行 TruncatedSVD 并监控性能...")
            tracemalloc.start()
            t_start = time.perf_counter()
            
            svd_model = TruncatedSVD(n_components=50, random_state=42)
            X_lsi = svd_model.fit_transform(adata.X)
            
            t_end = time.perf_counter()
            current_mem, peak_mem = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            
            svd_time_s = t_end - t_start
            svd_mem_mb = peak_mem / (1024 * 1024)
            print(f"  [LSI] TruncatedSVD 完成。耗时: {svd_time_s:.2f} 秒, 峰值内存: {svd_mem_mb:.2f} MB")
            # --- 性能监控结束 ---

            adata.obsm['X_lsi'] = X_lsi
            adata.varm['LSI_components'] = svd_model.components_.T

            # --- 2.2. UMAP 可视化 ---
            print("[2/4] UMAP: 计算 Neighbors 和 UMAP...")
            sc.pp.neighbors(adata, use_rep='X_lsi', n_neighbors=15, random_state=42)
            sc.tl.umap(adata, random_state=42)

            print("  [UMAP] 绘制并保存 UMAP 图...")
            sc.pl.umap(adata, color=['cell_type'], show=False, color_map='viridis')
            plt.savefig(os.path.join(umap_save_path, f'umap_cell_type({name}_LSI).pdf'), dpi=300, format="pdf", bbox_inches='tight', transparent=True)
            plt.close()

            if 'batch' in adata.obs.columns:
                sc.pl.umap(adata, color=['batch'], show=False, color_map='viridis')
                plt.savefig(os.path.join(umap_save_path, f'umap_batch({name}_LSI).pdf'), dpi=300, format='pdf', bbox_inches='tight', transparent=True)
                plt.close()

            # --- 2.3. 评估 ---
            print("[3/4] Metrics: 聚类并计算所有指标...")
            # 聚类
            true_cell_types = adata.obs[cell_type_key].nunique()
            find_best_leiden_resolution(adata, true_cell_types)
            leiden_key = f'leiden_res_best'
            
            # 计算聚类指标
            ami, ari, nmi, hom = cluster_evaluation(adata.obs, cell_type_key, leiden_key)
        
            # 计算 scib 指标
            if batch_key in adata.obs.columns:
                graph_clisi = scib.me.clisi_graph(adata, label_key=batch_key, type_="embed", use_rep="X_lsi")
                batch_asw = scib.me.silhouette_batch(adata, batch_key=batch_key, label_key=cell_type_key, embed="X_lsi", verbose=False)
                graph_connectivity = scib.me.graph_connectivity(adata, label_key=cell_type_key)
                ilisi_graph = scib.me.ilisi_graph(adata, batch_key=batch_key, type_="embed", use_rep="X_lsi", scale=False)
            else: # 如果没有batch信息，则填充为 NaN
                graph_clisi, batch_asw, graph_connectivity, ilisi_graph = np.nan, np.nan, np.nan, np.nan
        
            overcorrection = overcorrection_score(adata.obsm["X_lsi"], adata.obs[cell_type_key])

            # --- 2.4. 保存 ---
            print("[4/4] Saving: 保存处理后的 AnnData 对象和指标...")
            
            pkl_file_name = os.path.join(adata_save_path, f"{name}_lsi.pkl")
            with open(pkl_file_name, "wb") as f:
                pickle.dump(adata, f)

            # 汇总当前数据集的所有指标
            metrics_data = {
                "AMI": ami, "ARI": ari, "NMI": nmi, "Homogeneity": hom,
                'Graph_cLISI': graph_clisi, 'Batch_ASW': batch_asw,
                'Graph_Connectivity': graph_connectivity, 'iLISI_graph': ilisi_graph,
                'Overcorrection': overcorrection,
                'SVD_Time_s': svd_time_s,
                'SVD_Mem_MB': svd_mem_mb
            }
            metrics_s = pd.Series(metrics_data, name=name)
            all_metrics_list.append(metrics_s)
            
        except Exception as e:
            print(f"!!!!!! 处理数据集 {name} 时发生严重错误: {e} !!!!!!")

    # --- 3. 汇总并保存所有数据集的指标 ---
    if all_metrics_list:
        final_metrics_df = pd.concat(all_metrics_list, axis=1)
        
        csv_path = os.path.join(metrics_save_path, 'LSI_metrics_summary.csv')
        final_metrics_df.to_csv(csv_path)
        
        print(f"\n{'='*25} 所有任务完成 {'='*25}")
        print(f"所有数据集的评估指标已成功汇总到: {csv_path}")
        print("\n最终指标汇总:")
        print(final_metrics_df)
    else:
        print("\n没有成功处理任何数据集，未生成指标文件。")

# ================== 主程序入口 ==================
if __name__ == '__main__':
    # Input h5ad paths
    adata_files = {
        'Chen-2019-ATAC': '__HISEMBLE_DATA_ROOT__/0.01_h5ad/Chen-2019-ATAC_processed.h5ad',
        'Droplet': '__HISEMBLE_DATA_ROOT__/0.01_h5ad/Droplet_processed.h5ad',
        'FCA_fetal_33184180_cerebrum': '__HISEMBLE_DATA_ROOT__/0.01_h5ad/FCA_fetal_33184180_cerebrum_processed.h5ad',
        'FCA_fetal_33184180_cerebellum': '__HISEMBLE_DATA_ROOT__/0.01_h5ad/FCA_fetal_33184180_cerebellum_processed.h5ad',
        'FCA_fetal_33184180_eye': '__HISEMBLE_DATA_ROOT__/0.01_h5ad/FCA_fetal_33184180_eye_processed.h5ad',
        'FCA_fetal_33184180_intestine': '__HISEMBLE_DATA_ROOT__/0.01_h5ad/FCA_fetal_33184180_intestine_processed.h5ad',
        'FCA_fetal_33184180_kidney': '__HISEMBLE_DATA_ROOT__/0.01_h5ad/FCA_fetal_33184180_kidney_processed.h5ad',
        'FCA_fetal_33184180_lung': '__HISEMBLE_DATA_ROOT__/0.01_h5ad/FCA_fetal_33184180_lung_processed.h5ad',
        'FCA_fetal_33184180_muscle': '__HISEMBLE_DATA_ROOT__/0.01_h5ad/FCA_fetal_33184180_muscle_processed.h5ad',
        'FCA_fetal_33184180_pancreas': '__HISEMBLE_DATA_ROOT__/0.01_h5ad/FCA_fetal_33184180_pancreas_processed.h5ad',
        'FCA_fetal_33184180_placenta': '__HISEMBLE_DATA_ROOT__/0.01_h5ad/FCA_fetal_33184180_placenta_processed.h5ad',
        'FCA_fetal_33184180_stomach': '__HISEMBLE_DATA_ROOT__/0.01_h5ad/FCA_fetal_33184180_stomach_processed.h5ad',
        'mca_BoneMarrow_62016': '__HISEMBLE_DATA_ROOT__/0.01_h5ad/mca_BoneMarrow_62016_processed.h5ad',
        'mca_BoneMarrow_62216': '__HISEMBLE_DATA_ROOT__/0.01_h5ad/mca_BoneMarrow_62216_processed.h5ad',
        'mca_Cerebellum_62216': '__HISEMBLE_DATA_ROOT__/0.01_h5ad/mca_Cerebellum_62216_processed.h5ad',
        'mca_HeartA_62816': '__HISEMBLE_DATA_ROOT__/0.01_h5ad/mca_HeartA_62816_processed.h5ad',
        'mca_Kidney_62016': '__HISEMBLE_DATA_ROOT__/0.01_h5ad/mca_Kidney_62016_processed.h5ad',
        'mca_LargeIntestineA_62816': '__HISEMBLE_DATA_ROOT__/0.01_h5ad/mca_LargeIntestineA_62816_processed.h5ad',
        'mca_LargeIntestineB_62816': '__HISEMBLE_DATA_ROOT__/0.01_h5ad/mca_LargeIntestineB_62816_processed.h5ad',
        'mca_Liver_62016': '__HISEMBLE_DATA_ROOT__/0.01_h5ad/mca_Liver_62016_processed.h5ad',
        'mca_Lung1_62216': '__HISEMBLE_DATA_ROOT__/0.01_h5ad/mca_Lung1_62216_processed.h5ad',
        'mca_Lung2_62216': '__HISEMBLE_DATA_ROOT__/0.01_h5ad/mca_Lung2_62216_processed.h5ad',
        'mca_PreFrontalCortex_62216': '__HISEMBLE_DATA_ROOT__/0.01_h5ad/mca_PreFrontalCortex_62216_processed.h5ad',
        'mca_SmallIntestine_62816': '__HISEMBLE_DATA_ROOT__/0.01_h5ad/mca_SmallIntestine_62816_processed.h5ad',
        'mca_Testes_62016': '__HISEMBLE_DATA_ROOT__/0.01_h5ad/mca_Testes_62016_processed.h5ad',
        'mca_Thymus_62016': '__HISEMBLE_DATA_ROOT__/0.01_h5ad/mca_Thymus_62016_processed.h5ad',
        'mca_WholeBrainA_62216': '__HISEMBLE_DATA_ROOT__/0.01_h5ad/mca_WholeBrainA_62216_processed.h5ad',
        'mca_WholeBrainA_62816': '__HISEMBLE_DATA_ROOT__/0.01_h5ad/mca_WholeBrainA_62816_processed.h5ad',
        'mouse_brain': '__HISEMBLE_DATA_ROOT__/0.01_h5ad/mouse_brain_processed.h5ad',
        'Muto-2021-ATAC': '__HISEMBLE_DATA_ROOT__/0.01_h5ad/Muto-2021-ATAC_processed.h5ad',
        'TME-All': '__HISEMBLE_DATA_ROOT__/0.01_h5ad/TME-All_processed.h5ad'
    }
    
    # --- 2. 配置输出目录 ---
    output_directory = '__HISEMBLE_DATA_ROOT__/'
    
    # --- 执行主函数 ---
    run_lsi(
        adata_paths=adata_files, 
        output_dir=output_directory
    )
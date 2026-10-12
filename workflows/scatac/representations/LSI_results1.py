import scanpy as sc
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import time
import tracemalloc
import pickle
from utils import cluster_evaluation, overcorrection_score
# Import evaluation libraries.
import scib
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.decomposition import TruncatedSVD
from scipy.stats import entropy
import random

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    print(f"Global random seed set to: {seed}")

GLOBAL_SEED = 42
set_seed(GLOBAL_SEED)

def find_best_leiden_resolution(adata, true_cell_types, low=0.1, high=5.0, tolerance=0.01, key_prefix='leiden_res', random_seed=42):

    low_n_clusters = sc.tl.leiden(adata, resolution=low, copy=True, random_state=random_seed).obs['leiden'].nunique()
    high_n_clusters = sc.tl.leiden(adata, resolution=high, copy=True, random_state=random_seed).obs['leiden'].nunique()

    # Check boundary cases.
    if high_n_clusters <= true_cell_types:
        print(f"Warning: the upper resolution {high} produces {high_n_clusters} clusters, no more than the target {true_cell_types}. Using resolution={high}.")
        sc.tl.leiden(adata, resolution=high, key_added=f'{key_prefix}_best', random_state=random_seed)
        return high
    if low_n_clusters >= true_cell_types:
        print(f"Warning: the lower resolution {low} produces {low_n_clusters} clusters, no fewer than the target {true_cell_types}. Using resolution={low}.")
        sc.tl.leiden(adata, resolution=low, key_added=f'{key_prefix}_best', random_state=random_seed)
        return low

    best_res = low
    best_n_clust = low_n_clusters

    while high - low > tolerance:
        mid = (low + high) / 2
        sc.tl.leiden(adata, resolution=mid, key_added=f'leiden_{mid:.4f}', random_state=random_seed)
        num_clusters = adata.obs[f'leiden_{mid:.4f}'].nunique()

        # Track the resolution with the closest cluster count to the target.
        if abs(num_clusters - true_cell_types) < abs(best_n_clust - true_cell_types):
            best_res = mid
            best_n_clust = num_clusters

        if num_clusters < true_cell_types:
            low = mid
        elif num_clusters > true_cell_types:
            high = mid
        else:
            print(f"Exact match found. Best resolution: {mid:.4f}; clusters: {num_clusters}.")
            sc.tl.leiden(adata, resolution=mid, key_added=f'{key_prefix}_best', random_state=random_seed)
            return mid

    print(f"Search completed. Best resolution: {best_res:.4f}; clusters: {best_n_clust}; target: {true_cell_types}.")
    sc.tl.leiden(adata, resolution=best_res, key_added=f'{key_prefix}_best', random_state=random_seed)
    return best_res


def run_lsi(adata_paths, output_dir, cell_type_key='cell_type', batch_key='batch'):
    """Run LSI processing and evaluation for multiple single-cell datasets, then save results."""
    # 1. Initialize the workflow.
    # Define all output paths in one place.
    umap_save_path = os.path.join(output_dir, 'UMAP')
    adata_save_path = os.path.join(output_dir, 'ADATA')
    metrics_save_path = os.path.join(output_dir, 'metrics')
    os.makedirs(umap_save_path, exist_ok=True)
    os.makedirs(adata_save_path, exist_ok=True)
    os.makedirs(metrics_save_path, exist_ok=True)
    
    # Collect evaluation metrics across datasets.
    all_metrics_list = []

    print(f"Processing {len(adata_paths)} datasets...")

    # 2. Process each dataset.
    for name, path in adata_paths.items():
        print(f"\n{'='*20} Processing: {name} {'='*20}")
        
        try:
            adata = sc.read_h5ad(path)
            
            # 2.1. LSI workflow.
            print("[1/4] LSI: Binarize -> TF-IDF -> SVD")
            adata.layers['counts'] = adata.X.copy()
            
            binarized_matrix = adata.X.copy()
            binarized_matrix[binarized_matrix > 0] = 1
            
            tfidf_transformer = TfidfTransformer(norm='l2', use_idf=True, smooth_idf=True)
            tfidf_matrix = tfidf_transformer.fit_transform(binarized_matrix)
            adata.X = tfidf_matrix
            
            # Start performance monitoring.
            print("  [LSI] Running TruncatedSVD with performance monitoring...")
            tracemalloc.start()
            t_start = time.perf_counter()
            
            svd_model = TruncatedSVD(n_components=50, random_state=42)
            X_lsi = svd_model.fit_transform(adata.X)
            
            t_end = time.perf_counter()
            current_mem, peak_mem = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            
            svd_time_s = t_end - t_start
            svd_mem_mb = peak_mem / (1024 * 1024)
            print(f"  [LSI] TruncatedSVD completed. Time: {svd_time_s:.2f} s; peak memory: {svd_mem_mb:.2f} MB")
            # Finish performance monitoring.

            adata.obsm['X_lsi'] = X_lsi
            adata.varm['LSI_components'] = svd_model.components_.T

            # 2.2. UMAP visualization.
            print("[2/4] UMAP: computing neighbors and UMAP...")
            sc.pp.neighbors(adata, use_rep='X_lsi', n_neighbors=15, random_state=42)
            sc.tl.umap(adata, random_state=42)

            print("  [UMAP] Plotting and saving UMAP...")
            sc.pl.umap(adata, color=['cell_type'], show=False, color_map='viridis')
            plt.savefig(os.path.join(umap_save_path, f'umap_cell_type({name}_LSI).pdf'), dpi=300, format="pdf", bbox_inches='tight', transparent=True)
            plt.close()

            if 'batch' in adata.obs.columns:
                sc.pl.umap(adata, color=['batch'], show=False, color_map='viridis')
                plt.savefig(os.path.join(umap_save_path, f'umap_batch({name}_LSI).pdf'), dpi=300, format='pdf', bbox_inches='tight', transparent=True)
                plt.close()

            # 2.3. Evaluation.
            print("[3/4] Metrics: clustering and computing evaluation metrics...")
            # Cluster cells.
            true_cell_types = adata.obs[cell_type_key].nunique()
            find_best_leiden_resolution(adata, true_cell_types)
            leiden_key = f'leiden_res_best'
            
            # Calculate clustering metrics.
            ami, ari, nmi, hom = cluster_evaluation(adata.obs, cell_type_key, leiden_key)
        
            # Calculate scib integration metrics.
            if batch_key in adata.obs.columns:
                graph_clisi = scib.me.clisi_graph(adata, label_key=batch_key, type_="embed", use_rep="X_lsi")
                batch_asw = scib.me.silhouette_batch(adata, batch_key=batch_key, label_key=cell_type_key, embed="X_lsi", verbose=False)
                graph_connectivity = scib.me.graph_connectivity(adata, label_key=cell_type_key)
                ilisi_graph = scib.me.ilisi_graph(adata, batch_key=batch_key, type_="embed", use_rep="X_lsi", scale=False)
            else: # Use NaN when batch information is unavailable.
                graph_clisi, batch_asw, graph_connectivity, ilisi_graph = np.nan, np.nan, np.nan, np.nan
        
            overcorrection = overcorrection_score(adata.obsm["X_lsi"], adata.obs[cell_type_key])

            # 2.4. Save outputs.
            print("[4/4] Saving: writing the processed AnnData object and metrics...")
            
            pkl_file_name = os.path.join(adata_save_path, f"{name}_lsi.pkl")
            with open(pkl_file_name, "wb") as f:
                pickle.dump(adata, f)

            # Collect all metrics for the current dataset.
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
            print(f"Error while processing dataset {name}: {e}")

    # 3. Collect and save metrics across datasets.
    if all_metrics_list:
        final_metrics_df = pd.concat(all_metrics_list, axis=1)
        
        csv_path = os.path.join(metrics_save_path, 'LSI_metrics_summary.csv')
        final_metrics_df.to_csv(csv_path)
        
        print(f"\n{'='*25} All tasks completed {'='*25}")
        print(f"Dataset evaluation metrics saved to: {csv_path}")
        print("\nFinal metrics summary:")
        print(final_metrics_df)
    else:
        print("\nNo dataset was processed successfully; no metrics file was generated.")

# Main entry point.
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
    
    # 2. Configure output directories.
    output_directory = '__HISEMBLE_DATA_ROOT__/'
    
    # Run the main function.
    run_lsi(
        adata_paths=adata_files, 
        output_dir=output_directory
    )
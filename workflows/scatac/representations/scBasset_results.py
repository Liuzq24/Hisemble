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


def process_single_dataset(dataname, base_path='__HISEMBLE_DATA_ROOT__/'):
    """Run the complete analysis workflow for one dataset."""
    print(f"\n{'='*20} Processing dataset: {dataname} {'='*20}")

    # 1. Define paths and create directories.
    save_folder_umap = os.path.join(base_path, 'UMAP/')
    savedata_folder = os.path.join(base_path, 'ADATA/')
    savemetric_folder = os.path.join(base_path, 'metrics/scBasset/')
    scbasset_results_folder = os.path.join(base_path, f'scBasset_results/{dataname}/')

    for folder in [save_folder_umap, savedata_folder, savemetric_folder, scbasset_results_folder]:
        os.makedirs(folder, exist_ok=True)

    ad_file = os.path.join(base_path, f'0.01_processed/{dataname}_processed.h5ad')
    trained_model_file = os.path.join(base_path, f'scBasset_output/{dataname}/best_model.h5')

    # 2. Load data and the model, then generate embeddings.
    print("Loading data and the pretrained model...")
    if not os.path.exists(ad_file) or not os.path.exists(trained_model_file):
        print(f"Error: a data or model file is missing. Skipping {dataname}.")
        return None

    ad = anndata.read_h5ad(ad_file)
    model = make_model(32, ad.shape[0], show_summary=False)
    model.load_weights(trained_model_file)
    proj = get_cell_embedding(model)

    proj_csv_path = os.path.join(scbasset_results_folder, 'projection_atac.csv')
    pd.DataFrame(proj).to_csv(proj_csv_path)
    ad.obsm['projection'] = proj

    # 3. UMAP visualization.
    print("Computing and plotting UMAP...")
    sc.pp.neighbors(ad, use_rep='projection', n_neighbors=15, random_state=GLOBAL_SEED)
    sc.tl.umap(ad, random_state=GLOBAL_SEED)

    sc.pl.umap(ad, color=['cell_type'], show=False, color_map='viridis')
    plt.savefig(os.path.join(save_folder_umap, f'umap_cell_type({dataname}_scBasset).pdf'), dpi=300, format="pdf", bbox_inches='tight', transparent=True)
    plt.close() # Close the figure to release memory between datasets.

    if 'batch' in ad.obs.columns:
        sc.pl.umap(ad, color=['batch'], show=False, color_map='viridis')
        plt.savefig(os.path.join(save_folder_umap, f'umap_batch({dataname}_scBasset).pdf'), dpi=300, format='pdf', bbox_inches='tight', transparent=True)
        plt.close()

    # 4. Clustering and evaluation.
    print("Running Leiden clustering...")
    true_cell_types = ad.obs['cell_type'].nunique()
    best_res = find_best_leiden_resolution(ad, true_cell_types, random_seed=GLOBAL_SEED) 
    leiden_key = f'leiden_res_best' # find_best_leiden_resolution creates this key.

    print("Computing evaluation metrics...")
    # Calculate cluster_evaluation once to avoid repeated work.
    ami, ari, nmi, hom = cluster_evaluation(ad.obs, 'cell_type', leiden_key)

    # Calculate batch-correction and integration metrics.
    if 'batch' in ad.obs.columns:
        graph_clisi = scib.me.clisi_graph(ad, label_key="batch", type_="embed", use_rep="projection")
        batch_asw = scib.me.silhouette_batch(ad, batch_key="batch", label_key="cell_type", embed="projection", verbose=False)
        graph_connectivity = scib.me.graph_connectivity(ad, label_key="cell_type")
        ilisi_graph = scib.me.ilisi_graph(ad, batch_key="batch", type_="embed", use_rep="projection", scale=False)
    else:
        graph_clisi, batch_asw, graph_connectivity, ilisi_graph = 0, 0, 0, 0

    overcorrection = overcorrection_score(ad.obsm["projection"], ad.obs['cell_type'], n_neighbors=100, n_pools=100, n_samples_per_pool=100)

    # 5. Save results.
    print("Saving results...")
    # Save the processed AnnData object.
    pkl_file_name = os.path.join(savedata_folder, f"{dataname}_scBasset.pkl")
    with open(pkl_file_name, "wb") as f:
        pickle.dump(ad, f)

    # Collect evaluation metrics.
    metrics_dict = {
        f"{dataname}_scBasset": [ami, ari, nmi, hom, graph_clisi, batch_asw, graph_connectivity, ilisi_graph, overcorrection]
    }
    metrics_df = pd.DataFrame(metrics_dict, index=["AMI", "ARI", "NMI", "HOM", 'GraphcLISI', 'batchASW', 'Graph_connectivity', 'ilisi_graph', 'overcorrection'])

    # Save metrics to CSV.
    # Read and update existing results rather than appending duplicate rows.
    final_metrics_file = os.path.join(savemetric_folder, 'scBasset_metrics_summary.csv')
    if os.path.exists(final_metrics_file):
        all_metrics_df = pd.read_csv(final_metrics_file, index_col=0)
        # Update or add results for the current dataset.
        all_metrics_df[f"{dataname}_scBasset"] = metrics_df[f"{dataname}_scBasset"]
    else:
        all_metrics_df = metrics_df

    all_metrics_df.to_csv(final_metrics_file)

    print(f"--- Dataset {dataname} completed ---")
    return metrics_df


# Main entry point.
if __name__ == '__main__':
    # 1. Define the dataset list.
    dataname_list = [
        'mouse_brain'

    ]

    # 2. Process each dataset.
    all_results = {}
    for dataname in dataname_list:
        result_df = process_single_dataset(dataname)
        if result_df is not None:
            all_results.update(result_df.to_dict())

    # 3. Collect and print the final results.
    if all_results:
        final_summary_df = pd.DataFrame(all_results)
        print("\n\n{'='*25} All tasks completed; final results summary {'='*25}")
        print(final_summary_df)
    else:
        print("\nAll tasks completed without successful results.")
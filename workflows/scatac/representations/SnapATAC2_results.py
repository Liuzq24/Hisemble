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
import time  # Import timing utilities.
import psutil # Import process-memory monitoring utilities.
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

# Helper functions.
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


def Snapatac(inputdata, dataname, umapsave, savedata, savemetric):
    """Process preprocessed input data and save dataset-specific outputs.

Args:
    inputdata: Preprocessed data.
    dataname: Dataset identifier used in output filenames.
    saveumap: UMAP output directory.
    savedata: Processed-data output directory.
    savemetric: Metrics output path.
""" 
    ad = inputdata
    from scipy.sparse import csc_matrix
    ad.X = ad.X.tocsr()
    process = psutil.Process(os.getpid())
    mem_before_spectral_mb = process.memory_info().rss / (1024 * 1024)
    start_time_spectral = time.perf_counter()
    snap.tl.spectral(ad, features = None)
    end_time_spectral = time.perf_counter()
    # Record memory usage after spectral embedding.
    mem_after_spectral_mb = process.memory_info().rss / (1024 * 1024)

    # Calculate elapsed time and the memory increase for this step.
    elapsed_time_spectral_s = end_time_spectral - start_time_spectral
    peak_memory_spectral_mb = mem_after_spectral_mb - mem_before_spectral_mb

    print(f"--- Performance monitoring [snap.tl.spectral on {dataname}] ---")
    print(f"Elapsed time: {elapsed_time_spectral_s:.2f} s")
    print(f"Peak memory increase: {peak_memory_spectral_mb:.2f} MB")
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
    print(f"Best Leiden clustering resolution: {best_res}")

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
        # Calculate integration metrics when batch labels are available.
        GraphcLISI = scib.me.clisi_graph(ad, label_key="batch", type_="embed", use_rep="X_spectral")
        batchASW = scib.me.silhouette_batch(ad, batch_key="batch", label_key="cell_type", embed="X_spectral", verbose=False)
        Graph_connectivity = scib.me.graph_connectivity(ad, label_key="cell_type")
        ilisi_graph = scib.me.ilisi_graph(ad, batch_key="batch", type_="embed", use_rep="X_spectral", scale=False)
    else:
        # Set batch-related metrics to zero when batch labels are unavailable.
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

# Use descriptive row-index labels.
index_names = [
    "AMI", "ARI", "NMI", "HOM", "GraphcLISI", "batchASW", 
    "Graph_connectivity", "ilisi_graph", "overcorrection",
    "Spectral_Time_s", "Spectral_Memory_MB" # Updated labels.
]

all_results_dict = {}

for dataname in datanames:
    umapsave =  f'__HISEMBLE_DATA_ROOT__/UMAP/' 
    savedata =  f"__HISEMBLE_DATA_ROOT__/ADATA/"
    savemetric_folder = '__HISEMBLE_RESULTS_ROOT__/snapatac/'
    
    # Ensure the output directory exists.
    if not os.path.exists(savedata):
        os.makedirs(savedata)
    if not os.path.exists(umapsave):
        os.makedirs(umapsave)
    if not os.path.exists(savemetric_folder):
        os.makedirs(savemetric_folder)
    
    # Call the processing function.
    result_dict = Snapatac(
        inputdata=adatas[dataname],
        dataname=dataname,
        umapsave=umapsave,
        savedata=savedata,
        savemetric=savemetric_folder
    )
    
    all_results_dict.update(result_dict)


if not all_results_dict:
    print("\nAll tasks completed, but none produced a successful result.")
else:
    index_names = [
        "AMI", "ARI", "NMI", "HOM", "GraphcLISI", "batchASW", 
        "Graph_connectivity", "ilisi_graph", "overcorrection",
        "Spectral_Time_s", "Spectral_Memory_MB"
    ]
    final_df = pd.DataFrame(all_results_dict, index=index_names)
    
    final_metrics_file = os.path.join(savemetric_folder, 'final_metrics_summary.csv')

    # Update the metrics CSV while preserving existing results.
    if os.path.exists(final_metrics_file):
        print(f"Updating the existing metrics file: {final_metrics_file}")
        try:
            existing_df = pd.read_csv(final_metrics_file, index_col=0)
            # Update existing columns with the new values.
            existing_df.update(final_df)
            # Add columns present in final_df but absent from existing_df.
            for col in final_df.columns:
                if col not in existing_df.columns:
                    existing_df[col] = final_df[col]
            final_df = existing_df
        except Exception as e:
            print(f"Could not read the existing metrics file: {e}. The file will be replaced.")

    final_df.to_csv(final_metrics_file)

    print(f"\n\n{'='*25} All tasks completed {'='*25}")
    print("Final results summary:")
    print(final_df)
    print(f"\nAll metrics have been saved or updated at: {final_metrics_file}")

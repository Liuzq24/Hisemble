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
from utils import cluster_evaluation, overcorrection_score # Import shared utilities.
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
save_path = os.path.join(base_output_dir, 'downsampling_CHAI.csv')

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
                # Measure memory used by the current process and its children.
                mem = self.process.memory_info().rss
                # Include child-process memory for parallel tasks.
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


def Hisemble(W_list: list, k_neighbors: int = 15, n_iterations: int = 20, random_seed=42) -> csr_matrix:
    num_views = len(W_list)
    n_samples = W_list[0].shape[0]
    P_list = [normalize_sparse(W) for W in W_list]
    k_retained = k_neighbors * 2
    for iter_num in range(n_iterations):
        print(f"  - Iteration {iter_num + 1}/{n_iterations}...")
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

    fused_network = Hisemble(W_list=graphs, k_neighbors=K_NEIGHBORS, n_iterations=N_ITERATIONS, random_seed=RANDOM_SEED)
    del graphs

    sources, targets = fused_network.nonzero()
    weights = fused_network[sources, targets].A1
    net = ig.Graph(list(zip(sources, targets)), directed=False)
    net.es['weight'] = weights
    clusters = find_best_matrix(matrix=net, true_cell_types=true_cell_types, seed=RANDOM_SEED)
    del net

    return clusters

def run_one_kmeans(data, k, seed):
    # n_init=1000 uses many k-means initializations and can be slow.
    kmeans = KMeans(n_clusters=k, n_init=1000, random_state=seed, algorithm='elkan')
    return kmeans.fit_predict(data)

def run_sc3_workflow(embeddings, n_clusters=true_cell_types, n_runs=20):
    """SC3-style consensus baseline on precomputed embeddings.
1. Run repeated k-means on each embedding in parallel.
2. Construct the dense N x N co-association matrix.
"""
    N = embeddings[0].shape[0]
    all_labels = []

    # 1. Run k-means in parallel.
    # Run n_runs partitions per embedding, giving 3 * n_runs for three views.
    tasks = []
    for emb in embeddings:
        for i in range(n_runs):
            tasks.append((emb, i))
    
    results = Parallel(n_jobs=-1)(
        delayed(run_one_kmeans)(emb, n_clusters, seed) 
        for emb, seed in tasks
    )
    label_matrix = np.array(results).T # Shape: (N, 3*n_runs)

    # 2. Construct the dense consensus matrix, which dominates memory use.
    consensus_matrix = np.zeros((N, N), dtype=np.float32)
    
    # Accumulate partitions one at a time; the final N x N matrix remains dense.
    n_total_runs = label_matrix.shape[1]
    for i in range(n_total_runs):
        lbl = label_matrix[:, i]
        # Broadcast label equality: one for a shared cluster and zero otherwise.
        # This step creates a temporary N x N Boolean matrix.
        mat = (lbl[:, None] == lbl[None, :])
        consensus_matrix += mat
    
    consensus_matrix /= n_total_runs
    
    # 3. Hierarchical clustering requires a distance matrix.
    
    # Use 1 - consensus as the clustering distance.
    
    Z = linkage(squareform(1 - consensus_matrix, checks=False), method='complete')
    labels = fcluster(Z, t=n_clusters, criterion='maxclust')
    
    return labels

def dense_snf(wall_list, k=20, t=20):
    # 1. Normalize the affinity matrices to obtain P.
    def normalize(w):
        row_sums = w.sum(axis=1)
        row_sums[row_sums == 0] = 1 # Avoid division by zero.
        p = w / row_sums[:, np.newaxis]
        p = (p + p.T) / 2 # Preserve symmetry.
        return p

    # 2. Construct the local k-NN matrix S.
    def dominate_set(w, k):
        n = w.shape[0]
        s = np.zeros_like(w)
        # Retain the largest k values in each row.
        for i in range(n):
            indices = np.argsort(w[i, :])[-k:]
            s[i, indices] = w[i, indices]
        # Normalize S.
        return normalize(s)

    # Initialize the matrices.
    P = [normalize(w) for w in wall_list]
    S = [dominate_set(w, k) for w in wall_list]
    num_views = len(wall_list)
    
    # Iterative cross-diffusion updates.
    for _ in range(t):
        P_next = []
        for v in range(num_views):
            # Average the P matrices from the other views.
            others = [P[i] for i in range(num_views) if i != v]
            P_sum = sum(others) / (num_views - 1)
            
            # Update formula: P_new = S * P_others * S^T.
            # Dense matrix multiplication can be computationally expensive.
            P_new = S[v].dot(P_sum).dot(S[v].T)
            P_next.append(P_new)
        P = P_next

    # Construct the final fused matrix.
    return sum(P) / num_views

def run_spectral_clustering_njw(affinity_matrix, n_clusters=15):
    """Run the NJW (2002) spectral clustering algorithm used by CHAI.
Input: affinity_matrix, a dense N x N matrix.
"""
    import numpy as np
    from scipy.cluster.vq import kmeans2
    # Use eigh for the symmetric eigendecomposition.
    from scipy.linalg import eigh 

    # 1. Calculate the inverse square root of the degree matrix.
    # D_ii = sum(W_ij)
    row_sums = np.array(affinity_matrix.sum(axis=1)).flatten()
    # Avoid division by zero.
    row_sums[row_sums == 0] = 1e-10
    d_inv_sqrt = np.power(row_sums, -0.5)
    
    # An explicit N x N diagonal matrix would require substantial memory.
    # Use broadcasting to compute D^-1/2 * W * D^-1/2.
    # L_ij = W_ij * d_inv_sqrt[i] * d_inv_sqrt[j]
    
    # 2. Calculate the normalized affinity matrix L.
    # This step performs elementwise operations on an N x N matrix.
    L = affinity_matrix * d_inv_sqrt[:, np.newaxis] * d_inv_sqrt[np.newaxis, :]
    
    # 3. Eigendecomposition is a major computational bottleneck.
    # Select eigenvectors corresponding to the largest k eigenvalues.
    # subset_by_index=[N-k, N-1] selects the largest k eigenvalues.
    N = L.shape[0]
    # eigh exploits symmetry but remains expensive for large matrices.
    # Dense eigendecomposition can require O(N^3) time.
    evals, evecs = eigh(L, subset_by_index=[N-n_clusters, N-1])
    
    # 4. Normalize rows to unit length.
    # Normalize each row of the eigenvector matrix.
    row_norms = np.linalg.norm(evecs, axis=1)
    row_norms[row_norms == 0] = 1e-10
    X_normalized = evecs / row_norms[:, np.newaxis]
    
    # 5. K-means clustering.
    # Cluster the normalized eigenvectors.
    # minit='++' uses k-means++ initialization.
    centroids, labels = kmeans2(X_normalized, n_clusters, minit='++')
    
    return labels

def run_chai_proxy(embeddings, method='AvgSim'):
    N = embeddings[0].shape[0]
    labels_list = []
    for emb in embeddings:
        labels = KMeans(n_clusters=true_cell_types, n_init=10).fit_predict(emb)
        labels_list.append(labels)
        
    # 2. Construct binary co-association matrices for CHAI.
    binary_matrices = []
    for lbl in labels_list:
        # This step creates a dense N x N matrix.
        # Boolean storage reduces constants but still requires O(N^2) memory.
        mat = (lbl[:, None] == lbl[None, :])
        binary_matrices.append(mat.astype(np.float32))
        
    # 3. Fuse the matrices.
    if method == 'AvgSim':
        fused_matrix = sum(binary_matrices) / len(binary_matrices)
    elif method == 'SNF':
        fused_matrix = dense_snf(binary_matrices, k=20, t=20)
            
    final_labels = run_spectral_clustering_njw(fused_matrix, n_clusters=true_cell_types)
    
    return final_labels



MAX_CELLS = 183175
sample_sizes = [10000, 20000, 50000, 80000, 100000, 150000, 180000]
results = []
for n in sample_sizes:
    try:
        # Downsample cells.
        indices = np.random.choice(MAX_CELLS, n, replace=False)
        embs_subset = [FULL_EMB_1[indices], FULL_EMB_2[indices], FULL_EMB_3[indices]]
        print("  Running CHAI(AvgSim)...")
        with MemoryMonitor() as monitor:
            start = time.time()
            run_chai_proxy(embs_subset)
            end = time.time()
        res_avg = {
            'Method': 'CHAI(AvgSim)', 'Cells': n, 
            'Time(s)': end - start, 'Memory(MB)': monitor.peak_mb
        }
        results.append(res_avg)
        
        # Save successful results immediately.
        pd.DataFrame([res_avg]).to_csv(save_path, mode='a', header=not os.path.exists(save_path), index=False)
        print(f"  [Saved] CHAI(AvgSim) result for {n} cells.")
    except MemoryError:
        print("    CHAI(AvgSim): Memory Error (OOM)!")
        err_res = {'Method': 'CHAI(AvgSim)', 'Cells': n, 'Time(s)': None, 'Memory(MB)': None}
        results.append(err_res)
        # Record failures as well as successful runs.
        pd.DataFrame([err_res]).to_csv(save_path, mode='a', header=not os.path.exists(save_path), index=False)
    except Exception as e:
        print(f"    CHAI(AvgSim) Error: {e}")
        if "Array is too big" in str(e): # Handle NumPy errors.
             err_res = {'Method': 'CHAI(AvgSim)', 'Cells': n, 'Time(s)': None, 'Memory(MB)': None}
             results.append(err_res)
             # Record failures as well as successful runs.
             pd.DataFrame([err_res]).to_csv(save_path, mode='a', header=not os.path.exists(save_path), index=False)
            
    try:
        print("  Running CHAI(SNF)...")
        with MemoryMonitor() as monitor:
            start = time.time()
            run_chai_proxy(embs_subset, method = 'SNF') # Reduce the number of runs for a faster test.
            end = time.time()
        res_snf = {
            'Method': 'CHAI(SNF)', 'Cells': n, 
            'Time(s)': end - start, 'Memory(MB)': monitor.peak_mb
        }
        results.append(res_snf)

        # Save successful results immediately.
        pd.DataFrame([res_snf]).to_csv(save_path, mode='a', header=not os.path.exists(save_path), index=False)
        print(f"  [Saved] CHAI(SNF) result for {n} cells.")

    except MemoryError:
        print("    CHAI(SNF): Memory Error (OOM)!")
        err_res = {'Method': 'CHAI(SNF)', 'Cells': n, 'Time(s)': None, 'Memory(MB)': None}
        results.append(err_res)
        # Record failures as well as successful runs.
        pd.DataFrame([err_res]).to_csv(save_path, mode='a', header=not os.path.exists(save_path), index=False)

    except Exception as e:
        print(f"    CHAI(SNF) Error: {e}")
        if "Array is too big" in str(e): # Handle NumPy errors.
             err_res = {'Method': 'CHAI(SNF)', 'Cells': n, 'Time(s)': None, 'Memory(MB)': None}
             results.append(err_res)
             # Record failures as well as successful runs.
             pd.DataFrame([err_res]).to_csv(save_path, mode='a', header=not os.path.exists(save_path), index=False)
print("\nAll done.")
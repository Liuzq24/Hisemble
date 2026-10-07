import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.feature_extraction.text import TfidfTransformer
import scipy
import sklearn
from sklearn.neighbors import NearestNeighbors, KNeighborsRegressor
from sklearn.metrics import silhouette_score
from sklearn import metrics
from sklearn.preprocessing import OneHotEncoder, LabelEncoder,MaxAbsScaler
from sklearn.neighbors import NearestNeighbors, KNeighborsRegressor
import os
import pandas as pd
import numpy as np
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
from sklearn.metrics.pairwise import rbf_kernel, euclidean_distances
from sklearn.metrics import adjusted_rand_score,adjusted_mutual_info_score,normalized_mutual_info_score, homogeneity_score
from scipy.spatial.distance import pdist, squareform
import igraph as ig
import leidenalg
from sklearn.metrics import pairwise_distances
from scipy.sparse import csr_matrix, identity as sparse_identity
from sklearn.neighbors import NearestNeighbors
import time
import psutil
import threading

def cluster_evaluation(adata_obs, label_key, cluster_key):
    '''
    Clustering Performance Evaluation

    Args:
        adata_obs: polars.internals.frame.DataFrame.
        label_key: e.g. 'cell type', 'cell_type'
        cluster_key: e.g. 'mc_Dleiden'

    Returns:
        evaluation

    '''
    print(cluster_key)
    AMI = sklearn.metrics.adjusted_mutual_info_score(adata_obs[label_key], adata_obs[cluster_key])
    ARI = sklearn.metrics.adjusted_rand_score(adata_obs[cluster_key], adata_obs[label_key])
    NMI = sklearn.metrics.normalized_mutual_info_score(adata_obs[cluster_key], adata_obs[label_key])
    HOM = sklearn.metrics.homogeneity_score(adata_obs[label_key], adata_obs[cluster_key])

    
    print('AMI:%.3f\tARI:%.3f\tNMI:%.3f\tHOM:%.3f\t'%(AMI,ARI,NMI,HOM))
    return AMI,ARI,NMI,HOM

def overcorrection_score_ot(emb, celltype, n_neighbors=50, accept_ratio=0.8):
    n_neighbors = min(n_neighbors, len(emb)-1)
    nne = NearestNeighbors(n_neighbors=1+n_neighbors)
    nne.fit(emb)
    kmatrix = nne.kneighbors_graph(emb) -scipy.sparse.identity(emb.shape[0])

    score = 0
    celltype_ = np.unique(celltype)
    celltype_dict = celltype.value_counts().to_dict()
    
    N_celltype = len(celltype_)
    for i in range(len(celltype)):
        cell_type_num=celltype_dict[celltype[i]]    
        sub_num=np.sum( celltype[  kmatrix[i].nonzero()[1]  ] == celltype[i]  )
        if sub_num>accept_ratio*cell_type_num:
            sub_num= n_neighbors 
                  
        score += (n_neighbors -sub_num)/(sub_num+1)

    return score / float(len(celltype)) #negative index


def overcorrection_score(emb, celltype, n_neighbors=100, n_pools=100, n_samples_per_pool=100, seed=124):
    n_neighbors = min(n_neighbors, len(emb) - 1)
    nne = NearestNeighbors(n_neighbors=1 + n_neighbors, n_jobs=8)
    nne.fit(emb)
    kmatrix = nne.kneighbors_graph(emb) - scipy.sparse.identity(emb.shape[0])

    score = 0
    celltype_ = np.unique(celltype)
    celltype_dict = celltype.value_counts().to_dict()
    
    N_celltype = len(celltype_)

    for t in range(n_pools):
        indices = np.random.choice(np.arange(emb.shape[0]), size=n_samples_per_pool, replace=False)
        score += np.mean([np.mean(celltype[kmatrix[i].nonzero()[1]][:min(celltype_dict[celltype[i]], n_neighbors)] == celltype[i]) for i in indices])
    return 1-score / float(n_pools)



def find_best_matrix(matrix, true_cell_types, low=0.1, high=5.0, tolerance=0.01, seed=42):
    best_resolution = None
    clusters = None
    low_n_clusters = len(remove_duplicates(perform_leiden_clustering(matrix, resolution=low, seed=seed)))
    high_n_clusters = len(remove_duplicates(perform_leiden_clustering(matrix, resolution=high, seed=seed)))

    if high_n_clusters <= true_cell_types:
        best_resolution = high
    elif low_n_clusters >= true_cell_types:
        best_resolution = low
    else:
        while high - low > tolerance:
            mid = (low + high) / 2
            current_clusters = perform_leiden_clustering(matrix, resolution=mid, seed=seed)
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
            
    final_clusters = perform_leiden_clustering(matrix, resolution=best_resolution, seed=seed)
    print(f"最终使用分辨率: {best_resolution:.4f}, 产生 {len(remove_duplicates(final_clusters))} 个簇。")
    return final_clusters

def perform_leiden_clustering(net, resolution, seed=42):
    partition = leidenalg.find_partition(net, leidenalg.RBConfigurationVertexPartition, 
                                         weights=net.es['weight'], 
                                         resolution_parameter=resolution,
                                         seed=seed)
    clusters = partition.membership
    return clusters

def remove_duplicates(lst):  
    return list(np.unique(lst))

def build_knn_graph_chunked_cpu(embedding: np.ndarray, k_neighbors: int, metric: str, chunk_size: int = 2048, random_seed=42):
    n_samples = embedding.shape[0]
    all_rows, all_cols, all_data = [], [], []
    gamma = None
    if metric == 'gaussian':
        variance = embedding.var()
        if variance == 0:
            variance += 1e-12
        gamma = 1.0 / (embedding.shape[1] * variance)

    print(f"开始分块构建K-NN图 (chunk_size={chunk_size}, metric={metric})...")
    for i in range(0, n_samples, chunk_size):
        start_idx = i
        end_idx = min(i + chunk_size, n_samples)
        chunk_embedding = embedding[start_idx:end_idx]
        if metric == 'gaussian':
            sim_chunk = rbf_kernel(chunk_embedding, embedding, gamma=gamma)
        elif metric == 'cosine':
            sim_chunk = cosine_similarity(chunk_embedding, embedding)
            sim_chunk = np.maximum(sim_chunk, 0)

        np.random.seed(random_seed) # 确保argsort在平局时行为一致
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
    W_symmetric = (W_sparse + W_sparse.T) / 2
    print("分块构建完成！")
    return W_symmetric

def normalize_sparse(W_sparse: csr_matrix):
    n_samples = W_sparse.shape[0]
    row_sums = W_sparse.sum(axis=1).A.flatten()
    denominators = 2.0 * row_sums
    denominators[denominators == 0] = 1e-9
    inv_denominators = 1.0 / denominators
    normalizer = sp.diags(inv_denominators)
    P_non_diag = normalizer @ W_sparse
    P_sparse = P_non_diag + sp.diags(np.full(n_samples, 0.5))
    return P_sparse.tocsr()

def sparsify_matrix(sparse_matrix: csr_matrix, k_neighbors: int, random_seed=42):
    n_samples = sparse_matrix.shape[0]
    all_rows, all_cols, all_data = [], [], []
    np.random.seed(random_seed) # 确保argpartition行为一致
    for i in range(n_samples):
        row_slice = sparse_matrix.getrow(i)
        if row_slice.nnz > k_neighbors:
            data = row_slice.data
            indices = row_slice.indices
            top_k_indices_in_slice = np.argpartition(-data, k_neighbors)[:k_neighbors]
            top_data = data[top_k_indices_in_slice]
            top_cols = indices[top_k_indices_in_slice]
            all_rows.append(np.full(k_neighbors, i, dtype=np.int32))
            all_cols.append(top_cols)
            all_data.append(top_data)
        else:
            all_rows.append(np.full(row_slice.nnz, i, dtype=np.int32))
            all_cols.append(row_slice.indices)
            all_data.append(row_slice.data)
    final_rows = np.concatenate(all_rows)
    final_cols = np.concatenate(all_cols)
    final_data = np.concatenate(all_data)
    return csr_matrix((final_data, (final_rows, final_cols)), shape=(n_samples, n_samples))


def batch_entropy_mixing_score(adata, batches,use_rep, n_neighbors=50, n_pools=100, n_samples_per_pool=30):
    """
    Calculate batch entropy mixing score
    
    Algorithm
    -----
        * 1. Calculate the regional mixing entropies at the location of 100 randomly chosen cells from all batches
        * 2. Define 100 nearest neighbors for each randomly chosen cell
        * 3. Calculate the mean mixing entropy as the mean of the regional entropies
        * 4. Repeat above procedure for 100 iterations with different randomly chosen cells.
    
    Parameters
    ----------
    data
        np.array of shape nsamples x nfeatures.
    batches
        batch labels of nsamples.
    n_neighbors
        The number of nearest neighbors for each randomly chosen cell. By default, n_neighbors=100.
    n_samples_per_pool
        The number of randomly chosen cells from all batches per iteration. By default, n_samples_per_pool=100.
    n_pools
        The number of iterations with different randomly chosen cells. By default, n_pools=100.
        
    Returns
    -------
    Batch entropy mixing score
    """
#     print("Start calculating Entropy mixing score")
    data=adata.obsm[use_rep]
    def entropy(batches):
        p = np.zeros(N_batches)
        adapt_p = np.zeros(N_batches)
        a = 0
        for i in range(N_batches):
            p[i] = np.mean(batches == batches_[i])
            a = a + p[i]/P[i]
        entropy = 0
        for i in range(N_batches):
            adapt_p[i] = (p[i]/P[i])/a
            entropy = entropy - adapt_p[i]*np.log(adapt_p[i]+10**-8)
        return entropy

    n_neighbors = min(n_neighbors, len(data) - 1)
    nne = NearestNeighbors(n_neighbors=1 + n_neighbors, n_jobs=8)
    nne.fit(data)
    kmatrix = nne.kneighbors_graph(data) - scipy.sparse.identity(data.shape[0])

    score = 0
    batches_ = np.unique(batches)
    N_batches = len(batches_)
    if N_batches < 2:
        raise ValueError("Should be more than one cluster for batch mixing")
    P = np.zeros(N_batches)
    for i in range(N_batches):
            P[i] = np.mean(batches == batches_[i])
    for t in range(n_pools):
        indices = np.random.choice(np.arange(data.shape[0]), size=n_samples_per_pool)
        score += np.mean([entropy(batches[kmatrix[indices].nonzero()[1]
                                                 [kmatrix[indices].nonzero()[0] == i]])
                          for i in range(n_samples_per_pool)])
    Score = score / float(n_pools)
    return Score / float(np.log2(N_batches))



def onehot(y, n):
    """
    Make the input tensor one hot tensors
    
    Parameters
    ----------
    y
        input tensors
    n
        number of classes
        
    Return
    ------
    Tensor
    """
    if (y is None) or (n<2):
        return None
    assert torch.max(y).item() < n
    y = y.view(y.size(0), 1)
    y_cat = torch.zeros(y.size(0), n).to(y.device)
    y_cat.scatter_(1, y.data, 1)
    return y_cat

def token_map(x,j,n_domain,device):
    #token is a one hot vector
    batch_token=(j*torch.ones(x.size(0))).to(torch.int64).to(device)
    batch_token=F.one_hot(batch_token,num_classes=n_domain).to(torch.float32)
    return torch.cat((x,batch_token),1)

def TFIDF(adata,tfidf):
    # Perform TF-IDF (count_mat: peak*cell)
    def tfidf1(count_mat): 
        nfreqs = 1.0 * count_mat / np.tile(np.sum(count_mat,axis=0), (count_mat.shape[0],1))
        tfidf_mat = np.multiply(nfreqs, np.tile(np.log(1 + 1.0 * count_mat.shape[1] / np.sum(count_mat,axis=1)).reshape(-1,1), (1,count_mat.shape[1])))
        return scipy.sparse.csr_matrix(tfidf_mat)

    # Perform Signac TF-IDF (count_mat: peak*cell)
    def tfidf2(count_mat): 
        tf_mat = 1.0 * count_mat / np.tile(np.sum(count_mat,axis=0), (count_mat.shape[0],1))
        signac_mat = np.log(1 + np.multiply(1e4*tf_mat,  np.tile((1.0 * count_mat.shape[1] / np.sum(count_mat,axis=1)).reshape(-1,1), (1,count_mat.shape[1]))))
        return scipy.sparse.csr_matrix(signac_mat)
 
    def tfidf3(count_mat): 
        model = TfidfTransformer(smooth_idf=False, norm="l2")
        model = model.fit(np.transpose(count_mat))
        model.idf_ -= 1
        tf_idf = np.transpose(model.transform(np.transpose(count_mat)))
        return scipy.sparse.csr_matrix(tf_idf)
    
    if tfidf=='tfidf0':
        X_norm = sc.pp.normalize_total(adata, inplace=False)['X']
        adata.X = X_norm.copy()
    if tfidf=='tfidf1': 
        tfidf_res = tfidf1(adata.X.T).T
        adata.X = tfidf_res.copy()
    if tfidf=='tfidf2': 
        tfidf_res = tfidf2(adata.X.T).T
        adata.X = tfidf_res.copy()
    if tfidf=='tfidf3': 
        tfidf_res = tfidf3(adata.X.T).T
        adata.X = tfidf_res.copy()
        
        
        
class EarlyStopping:
    """
    Early stops the training if loss doesn't improve after a given patience.
    """
    def __init__(self, patience=10, verbose=False, checkpoint_file=''):
        """
        Parameters
        ----------
        patience 
            How long to wait after last time loss improved. Default: 30
        verbose
            If True, prints a message for each loss improvement. Default: False
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.loss_min = np.Inf
        self.checkpoint_file = checkpoint_file

    def __call__(self, loss, model):
        if np.isnan(loss):
            self.early_stop = True
        score = -loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(loss, model)
        elif score <= self.best_score:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
                model.load_model(self.checkpoint_file)
        else:
            self.best_score = score
            self.save_checkpoint(loss, model)
            self.counter = 0

    def save_checkpoint(self, loss, model):
        '''
        Saves model when loss decrease.
        '''
        if self.verbose:
            print(f'Loss decreased ({self.loss_min:.6f} --> {loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.checkpoint_file)
        self.loss_min = loss
        
class EarlyStopping_simple:
    """
    Early stops the training if loss doesn't improve after a given patience.
    """
    def __init__(self, patience=10, verbose=False):
        """
        Parameters
        ----------
        patience 
            How long to wait after last time loss improved. Default: 30
        verbose
            If True, prints a message for each loss improvement. Default: False
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.loss_min = np.Inf
       

    def __call__(self, loss, model):
        if np.isnan(loss):
            self.early_stop = True
        score = -loss

        if self.best_score is None:
            self.best_score = score
            
        elif score <= self.best_score:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
                
        else:
            self.best_score = score
           
            self.counter = 0

    def save_checkpoint(self, loss, model):
        '''
        Saves model when loss decrease.
        '''
        if self.verbose:
            print(f'Loss decreased ({self.loss_min:.6f} --> {loss:.6f}).  Saving model ...')
      
        self.loss_min = loss   
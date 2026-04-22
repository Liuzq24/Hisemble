import numpy as np
import igraph as ig
import leidenalg
from scipy.sparse import csr_matrix

def perform_leiden_clustering(net: ig.Graph, resolution: float, seed=42):
    """标准的 Leiden 社区发现算法"""
    partition = leidenalg.find_partition(
        net, 
        leidenalg.RBConfigurationVertexPartition, 
        weights=net.es['weight'], 
        resolution_parameter=resolution,
        seed=seed
    )
    return partition.membership

def matrix_to_igraph(fused_network: csr_matrix) -> ig.Graph:
    """将稀疏矩阵转化为 igraph 对象"""
    sources, targets = fused_network.nonzero()
    weights = fused_network[sources, targets].A1
    net = ig.Graph(list(zip(sources, targets)), directed=False)
    net.es['weight'] = weights
    return net

def find_best_matrix(matrix: ig.Graph, true_cell_types: int, low=0.1, high=5.0, tolerance=0.01, seed=42):
    """通过二分查找寻找最优的分辨率 (用于 Benchmark)"""
    best_resolution = None
    low_n_clusters = len(np.unique(perform_leiden_clustering(matrix, resolution=low, seed=seed)))
    high_n_clusters = len(np.unique(perform_leiden_clustering(matrix, resolution=high, seed=seed)))

    if high_n_clusters <= true_cell_types:
        best_resolution = high
    elif low_n_clusters >= true_cell_types:
        best_resolution = low
    else:
        while high - low > tolerance:
            mid = (low + high) / 2
            current_clusters = perform_leiden_clustering(matrix, resolution=mid, seed=seed)
            num_clusters = len(np.unique(current_clusters))
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
    return final_clusters
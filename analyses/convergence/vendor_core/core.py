import numpy as np
from scipy.sparse import csr_matrix
from .graph import normalize_sparse, sparsify_matrix

def hisemble_fusion(W_list: list, k_neighbors: int = 20, n_iterations: int = 20, random_seed=42, verbose=True) -> csr_matrix:
    """
    Iterative sparse cross-diffusion
    """
    num_views = len(W_list)
    n_samples = W_list[0].shape[0]

    P_list = [normalize_sparse(W) for W in W_list]
    k_retained = k_neighbors * 2
    
    for iter_num in range(n_iterations):
        if verbose:
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